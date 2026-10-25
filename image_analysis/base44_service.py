"""HTTP service wrapper for the repository's authoritative image pipeline."""

from __future__ import annotations

import hashlib
import hmac
import io
import ipaddress
import json
import logging
import math
import os
import queue
import re
import shutil
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from image_analysis import golden_match_v5 as matcher
from image_analysis import painting_crop_v8 as cropper

logger = logging.getLogger("base44_service")

MODEL_PATH = Path(os.getenv("SAM_MODEL_PATH", "models/sam_vit_b_01ec64.pth"))
MODEL_SHA256 = os.getenv(
    "SAM_MODEL_SHA256",
    "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
)
DATA_DIR = Path(os.getenv("BASE44_DATA_DIR", "/data"))
PUBLIC_BASE_URL = os.getenv("BASE44_PUBLIC_URL", "").rstrip("/")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
MAX_REFERENCES = int(os.getenv("MAX_REFERENCES", "200"))
MAX_QUEUED_JOBS = int(os.getenv("MAX_QUEUED_JOBS", "100"))
MAX_PHOTOS_PER_JOB = 10
MAX_REDIRECTS = 3
SUPPORTED_SUFFIXES = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"BM": ".bmp",
    b"II*\x00": ".tiff",
    b"MM\x00*": ".tiff",
}
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceInput(StrictModel):
    id: str
    image_url: str
    sha256: str


class VaultInput(StrictModel):
    revision: str
    references: list[ReferenceInput] = Field(min_length=1, max_length=MAX_REFERENCES)


class PhotoInput(StrictModel):
    id: str
    image_url: str
    corners: list[list[float]] | None = Field(default=None, min_length=4, max_length=4)
    sha256: str | None = None


class JobInput(StrictModel):
    vault_revision: str | None = None
    photos: list[PhotoInput] = Field(min_length=1, max_length=MAX_PHOTOS_PER_JOB)


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class DownloadError(Exception):
    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message


class ServiceState:
    def __init__(self) -> None:
        self.root = DATA_DIR
        self.jobs_dir = self.root / "jobs"
        self.vaults_dir = self.root / "vaults"
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=MAX_QUEUED_JOBS)
        self.lock = threading.RLock()
        self.worker: threading.Thread | None = None
        self.predictor: Any = None
        self.model_verified: tuple[int, int] | None = None
        self.vault_cache: dict[str, tuple[Any, dict[str, Any], dict[str, str]]] = {}
        self.active_revision: str | None = None

    def initialize(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.vaults_dir.mkdir(parents=True, exist_ok=True)
        pointer = self.root / "active.json"
        if pointer.exists():
            try:
                self.active_revision = json.loads(pointer.read_text())["revision"]
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Could not read active vault pointer")

        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
                if record.get("status") in {"queued", "running"}:
                    record["status"] = "queued"
                    record["progress"] = "Recovered after service restart"
                    self.save_job(record)
                    self.queue.put(record["job_id"])
            except (OSError, ValueError, KeyError, TypeError):
                logger.exception("Could not recover job file %s", path.name)

        self.worker = threading.Thread(target=self.worker_loop, name="analysis-worker", daemon=True)
        self.worker.start()

    def shutdown(self) -> None:
        self.queue.put(None)
        if self.worker is not None:
            self.worker.join(timeout=2)

    def job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(self.job_path(job_id).read_text())
        except FileNotFoundError:
            return None

    def save_job(self, record: dict[str, Any]) -> None:
        path = self.job_path(record["job_id"])
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, separators=(",", ":")))
        os.replace(temporary, path)

    def worker_loop(self) -> None:
        while True:
            job_id = self.queue.get()
            if job_id is None:
                return
            try:
                self.run_job(job_id)
            except Exception:
                logger.exception("Unhandled analysis error for job %s", job_id)
                record = self.load_job(job_id)
                if record is not None:
                    record.update(
                        status="failed",
                        progress="Analysis failed",
                        error={"code": "analysis_failed", "message": "Image analysis failed"},
                        updated_at=int(time.time()),
                    )
                    self.save_job(record)

    def get_vault(self, revision: str):
        with self.lock:
            cached = self.vault_cache.get(revision)
            if cached is not None:
                return cached

            revision_dir = self.vaults_dir / revision
            metadata_path = revision_dir / "manifest.json"
            index_dir = revision_dir / "index"
            if not metadata_path.is_file() or not (index_dir / matcher.VAULT_FILE).is_file():
                raise FileNotFoundError("Vault revision is unavailable")
            metadata = json.loads(metadata_path.read_text())
            data, indexes = matcher.load_vault(str(index_dir))
            id_by_path = {
                str((revision_dir / "images" / item["filename"]).resolve()): item["id"]
                for item in metadata["references"]
            }
            loaded = (data, indexes, id_by_path)
            self.vault_cache[revision] = loaded
            return loaded

    def get_predictor(self):
        with self.lock:
            if self.predictor is None:
                if not self.verify_model():
                    raise RuntimeError("SAM checkpoint is missing or failed SHA-256 verification")
                self.predictor = cropper.build_predictor(MODEL_PATH)
            return self.predictor

    def verify_model(self) -> bool:
        try:
            stat = MODEL_PATH.stat()
        except OSError:
            return False
        signature = (stat.st_size, stat.st_mtime_ns)
        if self.model_verified == signature:
            return True
        digest = hashlib.sha256()
        try:
            with MODEL_PATH.open("rb") as model_file:
                for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        if not hmac.compare_digest(digest.hexdigest(), MODEL_SHA256):
            return False
        self.model_verified = signature
        return True

    def run_job(self, job_id: str) -> None:
        record = self.load_job(job_id)
        if record is None:
            return
        record.update(status="running", progress="Downloading photos", updated_at=int(time.time()))
        self.save_job(record)
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        photos: list[dict[str, Any]] = []

        for photo in record["request"]["photos"]:
            input_path = job_dir / f"{photo['id']}.image"
            try:
                image_bytes = download_image(photo["image_url"])
                digest = hashlib.sha256(image_bytes).hexdigest()
                if photo.get("sha256") and not hmac.compare_digest(digest, photo["sha256"]):
                    raise DownloadError("image_hash_mismatch", "Image hash did not match the request")
                validate_image(image_bytes)
                input_path.write_bytes(image_bytes)
                photos.append({"photo": photo, "input_path": input_path, "sha256": digest})
            except DownloadError as error:
                record.update(
                    status="failed",
                    progress="Photo download failed",
                    error={"code": error.code, "message": error.public_message, "photo_id": photo["id"]},
                    updated_at=int(time.time()),
                )
                self.save_job(record)
                return

        record.update(status="running", progress="Cropping photos", updated_at=int(time.time()))
        self.save_job(record)
        crop_reports: list[dict[str, Any]] = []
        crop_failures: list[dict[str, Any]] = []
        for item in photos:
            photo = item["photo"]
            output_path = job_dir / f"{photo['id']}_painting_v8.tiff"
            manual = np.asarray(photo["corners"], dtype=np.float32) if photo.get("corners") else None
            try:
                predictor = None if manual is not None else self.get_predictor()
                report = cropper.crop_painting(item["input_path"], output_path, "tiff", predictor, manual)
                crop_reports.append({**item, "report": report, "crop_path": output_path})
            except (ValueError, OSError) as error:
                crop_failures.append(
                    {
                        "photo_id": photo["id"],
                        "error": {
                            "code": "invalid_corners" if manual is not None else "crop_failed",
                            "message": str(error)[:300],
                        },
                    }
                )

        if crop_failures:
            record.update(
                status="succeeded",
                progress="Corner adjustment required",
                result={"status": "needs_corners", "matched_reference_id": None, "photos": crop_failures},
                updated_at=int(time.time()),
            )
            self.save_job(record)
            return

        revision = record["vault_revision"]
        try:
            data, indexes, id_by_path = self.get_vault(revision)
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError("The pinned vault revision is unavailable") from error

        record.update(status="running", progress="Matching crops", updated_at=int(time.time()))
        self.save_job(record)
        shot_results: list[dict[str, Any]] = []
        for item in crop_reports:
            shot = matcher.search_shot(data, indexes, str(item["crop_path"]))
            top_path = str(Path(shot["golden"]).resolve())
            reference_id = id_by_path.get(top_path)
            if reference_id is None:
                raise RuntimeError("Matcher returned a reference outside the pinned vault")
            accepted_id = reference_id if shot["match"] else None
            preview_url = make_preview(item["crop_path"], job_dir, item["photo"]["id"])
            report = item["report"]
            shot_results.append(
                {
                    "photo_id": item["photo"]["id"],
                    "input_sha256": item["sha256"],
                    "crop": {
                        "method": report["method"],
                        "corners": report["corners"]["points"],
                        "bbox": report["bbox"],
                        "output_size": report["output_size"],
                        "preview_url": preview_url,
                    },
                    "match": {
                        "matched_reference_id": accepted_id,
                        "top_reference_id": reference_id,
                        "inliers": shot["inliers"],
                        "runner_up_inliers": shot["runner_up_inliers"],
                        "channel_inliers": shot["channel_inliers"],
                        "template_score_percent": shot["template_score_percent"],
                    },
                }
            )

        wins = Counter(
            shot["match"]["matched_reference_id"]
            for shot in shot_results
            if shot["match"]["matched_reference_id"]
        )
        needed = math.ceil(len(shot_results) * matcher.SHOT_FRACTION)
        consensus_id = wins.most_common(1)[0][0] if wins and wins.most_common(1)[0][1] >= needed else None
        record.update(
            status="succeeded",
            progress="Analysis complete",
            result={
                "status": "match" if consensus_id else "no_match",
                "matched_reference_id": consensus_id,
                "photos": shot_results,
            },
            updated_at=int(time.time()),
        )
        self.save_job(record)


state = ServiceState()


def configured_hosts() -> set[str]:
    return {host.strip().lower().rstrip(".") for host in os.getenv("IMAGE_URL_HOSTS", "").split(",") if host.strip()}


def validate_url(url: str) -> urllib.parse.SplitResult:
    parsed = urllib.parse.urlsplit(url)
    allowed = configured_hosts()
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise DownloadError("invalid_image_url", "Image URL must be an HTTPS URL on an allowed host")
    if not any(hostname == host or hostname.endswith("." + host) for host in allowed):
        raise DownloadError("image_host_not_allowed", "Image URL host is not allowed")
    try:
        address = ipaddress.ip_address(hostname)
        if not address.is_global:
            raise DownloadError("image_host_not_allowed", "Private and non-public image hosts are not allowed")
    except ValueError:
        pass
    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
            raise DownloadError("image_host_not_allowed", "Image host resolved to a non-public address")
    except socket.gaierror as error:
        raise DownloadError("image_host_unavailable", "Image host could not be resolved") from error
    return parsed


def download_image(url: str) -> bytes:
    opener = urllib.request.build_opener(NoRedirectHandler())
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        validate_url(current_url)
        request = urllib.request.Request(current_url, headers={"User-Agent": "base44-image-analysis/1"})
        try:
            response = opener.open(request, timeout=20)
        except urllib.error.HTTPError as error:
            if error.code in {301, 302, 303, 307, 308} and redirect_count < MAX_REDIRECTS:
                location = error.headers.get("Location")
                if not location:
                    raise DownloadError("image_download_failed", "Image URL redirect was invalid") from error
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            raise DownloadError("image_download_failed", f"Image download returned HTTP {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DownloadError("image_download_failed", "Could not download image") from error

        with response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_length = int(content_length)
                except ValueError as error:
                    raise DownloadError("invalid_image_response", "Image response had an invalid content length") from error
                if declared_length < 0 or declared_length > MAX_IMAGE_BYTES:
                    raise DownloadError("image_too_large", "Image exceeds the configured byte limit")
            chunks = bytearray()
            while True:
                chunk = response.read(min(1024 * 1024, MAX_IMAGE_BYTES + 1 - len(chunks)))
                if not chunk:
                    return bytes(chunks)
                chunks.extend(chunk)
                if len(chunks) > MAX_IMAGE_BYTES:
                    raise DownloadError("image_too_large", "Image exceeds the configured byte limit")
    raise DownloadError("too_many_redirects", "Image URL exceeded the redirect limit")


def validate_image(image_bytes: bytes) -> tuple[np.ndarray, str]:
    suffix = next((extension for magic, extension in SUPPORTED_SUFFIXES.items() if image_bytes.startswith(magic)), None)
    if suffix is None:
        raise DownloadError("unsupported_image", "Image must be a JPEG, PNG, BMP, or TIFF file")
    try:
        with Image.open(io.BytesIO(image_bytes)) as decoded:
            width, height = decoded.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise DownloadError("image_too_many_pixels", "Image exceeds the configured pixel limit")
            decoded.verify()
    except DownloadError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise DownloadError("invalid_image", "Image data could not be decoded") from error
    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise DownloadError("invalid_image", "Image data could not be decoded") from error
    if image is None:
        raise DownloadError("invalid_image", "Image data could not be decoded")
    decoded_height, decoded_width = image.shape[:2]
    if decoded_width * decoded_height > MAX_IMAGE_PIXELS:
        raise DownloadError("image_too_many_pixels", "Image exceeds the configured pixel limit")
    return image, suffix


def authenticate(authorization: str | None = Header(default=None)) -> None:
    expected = os.getenv("BASE44_SERVICE_TOKEN", "")
    if not expected:
        raise HTTPException(status_code=503, detail="Service authentication is not configured")
    supplied = authorization.removeprefix("Bearer ") if authorization and authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})


def validate_identifier(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not pattern.fullmatch(value) or value in {".", ".."}:
        raise HTTPException(status_code=422, detail=f"Invalid {label}")


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")))
    os.replace(temporary, path)


def make_preview(crop_path: Path, job_dir: Path, photo_id: str) -> str:
    image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Could not create crop preview")
    height, width = image.shape[:2]
    factor = min(1.0, 1600 / max(height, width))
    if factor < 1:
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise RuntimeError("Could not encode crop preview")
    (job_dir / f"{photo_id}_preview.jpg").write_bytes(encoded.tobytes())
    path = f"/v1/jobs/{job_dir.name}/artifacts/{photo_id}.jpg"
    return f"{PUBLIC_BASE_URL}{path}" if PUBLIC_BASE_URL else path


def config_version() -> str:
    config = {
        "crop": "painting_crop_v8",
        "matcher": "golden_match_v5",
        "min_inliers": matcher.MIN_INLIERS,
        "margin": matcher.MARGIN,
        "shot_fraction": matcher.SHOT_FRACTION,
        "work_side": matcher.WORK_SIDE,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return f"sha256:{digest}"


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.initialize()
    try:
        yield
    finally:
        state.shutdown()


app = FastAPI(title="Base44 Artwork Analysis", version="1.0.0", lifespan=lifespan)


@app.get("/health/live")
def health_live():
    return {"status": "live"}


@app.get("/health/ready", dependencies=[Depends(authenticate)])
def health_ready():
    model_ready = state.verify_model()
    revision = state.active_revision
    vault_ready = False
    if revision:
        try:
            state.get_vault(revision)
            vault_ready = True
        except Exception:
            logger.exception("Could not load vault revision %s", revision)
            vault_ready = False
    if not model_ready or not vault_ready:
        raise HTTPException(status_code=503, detail="Model or vault is not ready")
    return {
        "status": "ready",
        "service": app.version,
        "crop": "painting_crop_v8",
        "matcher": "golden_match_v5",
        "model": MODEL_PATH.name,
        "model_sha256": MODEL_SHA256,
        "config": config_version(),
        "vault_revision": revision,
    }


@app.put("/v1/vault", dependencies=[Depends(authenticate)])
def sync_vault(payload: VaultInput):
    validate_identifier(payload.revision, REVISION_PATTERN, "vault revision")
    ids = [reference.id for reference in payload.references]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Reference IDs must be unique")
    for reference in payload.references:
        validate_identifier(reference.id, ID_PATTERN, "reference ID")
        if not SHA256_PATTERN.fullmatch(reference.sha256):
            raise HTTPException(status_code=422, detail=f"Invalid SHA-256 for reference {reference.id}")
    hashes = [reference.sha256 for reference in payload.references]
    if len(set(hashes)) != len(hashes):
        raise HTTPException(status_code=422, detail="Byte-identical references must use one stable reference ID")

    manifest_digest = hashlib.sha256(
        json.dumps(
            sorted((reference.id, reference.sha256) for reference in payload.references),
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    revision_dir = state.vaults_dir / payload.revision
    manifest_path = revision_dir / "manifest.json"
    if revision_dir.exists():
        if not manifest_path.is_file():
            raise HTTPException(status_code=409, detail="Vault revision exists but is incomplete")
        existing = json.loads(manifest_path.read_text())
        if existing.get("manifest_sha256") != manifest_digest:
            raise HTTPException(status_code=409, detail="Vault revision already exists with different contents")
        try:
            state.get_vault(payload.revision)
        except (OSError, ValueError, KeyError):
            raise HTTPException(status_code=409, detail="Vault revision index is incomplete or unreadable")
        state.active_revision = payload.revision
        write_json_atomic(state.root / "active.json", {"revision": payload.revision})
        return {"revision": payload.revision, "status": "ready", "rebuilt": False}

    revision_dir.mkdir(parents=True)
    images_dir = revision_dir / "images"
    index_dir = revision_dir / "index"
    images_dir.mkdir()
    try:
        stored_references = []
        for reference in payload.references:
            image_bytes = download_image(reference.image_url)
            digest = hashlib.sha256(image_bytes).hexdigest()
            if not hmac.compare_digest(digest, reference.sha256):
                raise HTTPException(status_code=422, detail=f"Image hash mismatch for reference {reference.id}")
            _, suffix = validate_image(image_bytes)
            filename = f"{reference.id}{suffix}"
            (images_dir / filename).write_bytes(image_bytes)
            stored_references.append({"id": reference.id, "filename": filename, "sha256": digest})

        matcher.build_vault(str(images_dir), str(index_dir), progress=False)
        metadata = {
            "revision": payload.revision,
            "manifest_sha256": manifest_digest,
            "references": stored_references,
            "created_at": int(time.time()),
        }
        write_json_atomic(manifest_path, metadata)
        state.get_vault(payload.revision)
        write_json_atomic(state.root / "active.json", {"revision": payload.revision})
        state.active_revision = payload.revision
        return {"revision": payload.revision, "status": "ready", "rebuilt": True}
    except HTTPException:
        shutil.rmtree(revision_dir, ignore_errors=True)
        raise
    except Exception as error:
        shutil.rmtree(revision_dir, ignore_errors=True)
        if isinstance(error, DownloadError):
            raise HTTPException(status_code=422, detail={"code": error.code, "message": error.public_message}) from error
        logger.exception("Vault build failed for revision %s", payload.revision)
        raise HTTPException(status_code=422, detail="Vault images could not be indexed") from error


@app.post("/v1/jobs", status_code=202, dependencies=[Depends(authenticate)])
def create_job(payload: JobInput):
    ids = [photo.id for photo in payload.photos]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Photo IDs must be unique")
    for photo in payload.photos:
        validate_identifier(photo.id, ID_PATTERN, "photo ID")
        if photo.sha256 and not SHA256_PATTERN.fullmatch(photo.sha256):
            raise HTTPException(status_code=422, detail=f"Invalid SHA-256 for photo {photo.id}")
        if photo.corners is not None:
            corners = np.asarray(photo.corners, dtype=np.float32)
            if corners.shape != (4, 2) or not np.isfinite(corners).all():
                raise HTTPException(status_code=422, detail=f"Invalid corners for photo {photo.id}")
            if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
                raise HTTPException(status_code=422, detail=f"Corners must form a convex quadrilateral for photo {photo.id}")

    revision = payload.vault_revision or state.active_revision
    if not revision:
        raise HTTPException(status_code=503, detail="No vault revision is active")
    validate_identifier(revision, REVISION_PATTERN, "vault revision")
    try:
        state.get_vault(revision)
    except Exception:
        raise HTTPException(status_code=422, detail="Requested vault revision is unavailable")

    job_id = str(uuid.uuid4())
    record = {
        "job_id": job_id,
        "status": "queued",
        "progress": "Waiting for analysis worker",
        "vault_revision": revision,
        "versions": {
            "crop": "painting_crop_v8",
            "matcher": "golden_match_v5",
            "model": MODEL_PATH.name,
            "model_sha256": MODEL_SHA256,
            "config": config_version(),
        },
        "request": payload.model_dump(),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    state.save_job(record)
    try:
        state.queue.put_nowait(job_id)
    except queue.Full:
        state.job_path(job_id).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Analysis queue is full")
    return {"job_id": job_id, "status": "queued", "vault_revision": revision}


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(authenticate)])
def get_job(job_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")
    record = state.load_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {key: value for key, value in record.items() if key != "request"}


@app.get("/v1/jobs/{job_id}/artifacts/{photo_id}.jpg", dependencies=[Depends(authenticate)])
def get_preview(job_id: str, photo_id: str):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found")
    validate_identifier(photo_id, ID_PATTERN, "photo ID")
    path = state.jobs_dir / job_id / f"{photo_id}_preview.jpg"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="image/jpeg", filename=f"{photo_id}_preview.jpg")