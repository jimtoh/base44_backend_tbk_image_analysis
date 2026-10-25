import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from image_analysis import base44_service as service


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE44_SERVICE_TOKEN", "test-service-token")
    service.state.root = tmp_path
    service.state.jobs_dir = tmp_path / "jobs"
    service.state.vaults_dir = tmp_path / "vaults"
    service.state.active_revision = None
    service.state.vault_cache.clear()
    service.state.predictor = None
    service.state.model_verified = None
    with TestClient(service.app) as test_client:
        yield test_client


def test_health_route_authentication(client):
    assert client.get("/health/live").json() == {"status": "live"}
    assert client.get("/health/ready").status_code == 401
    response = client.get("/health/ready", headers={"Authorization": "Bearer test-service-token"})
    assert response.status_code == 503


def test_image_urls_require_https_and_an_allowed_host(monkeypatch):
    monkeypatch.setenv("IMAGE_URL_HOSTS", "images.example.com")
    with pytest.raises(service.DownloadError, match="HTTPS"):
        service.validate_url("http://images.example.com/photo.jpg")
    with pytest.raises(service.DownloadError, match="not allowed"):
        service.validate_url("https://attacker.example/photo.jpg")


def test_job_rejects_non_convex_manual_corners(client):
    response = client.post(
        "/v1/jobs",
        headers={"Authorization": "Bearer test-service-token"},
        json={
            "photos": [
                {
                    "id": "photo-1",
                    "image_url": "https://images.example.com/photo.png",
                    "corners": [[10, 10], [90, 70], [90, 10], [10, 70]],
                }
            ]
        },
    )
    assert response.status_code == 422


def test_image_pixel_limit_is_checked_before_opencv_decode(monkeypatch):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    monkeypatch.setattr(service, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(service.DownloadError, match="pixel limit"):
        service.validate_image(encoded.tobytes())


def test_manual_corners_bypass_sam_and_reach_original_cropper(client, tmp_path, monkeypatch):
    height, width = 80, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    image_bytes = encoded.tobytes()
    revision_dir = tmp_path / "vaults" / "revision-1"
    reference_path = revision_dir / "images" / "reference-1.png"
    reference_path.parent.mkdir(parents=True)
    reference_path.write_bytes(image_bytes)
    job_id = "7fb98776-2e91-4f61-9179-af2622129864"
    job_dir = tmp_path / "jobs"
    submitted_corners = [[10, 10], [90, 10], [90, 70], [10, 70]]
    record = {
        "job_id": job_id,
        "status": "queued",
        "vault_revision": "revision-1",
        "request": {
            "photos": [
                {
                    "id": "photo-1",
                    "image_url": "https://images.example.com/photo.png",
                    "corners": submitted_corners,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            ]
        },
    }
    service.state.save_job(record)
    service.state.vault_cache["revision-1"] = (
        object(),
        object(),
        {str(reference_path.resolve()): "reference-1"},
    )
    monkeypatch.setattr(service, "download_image", lambda _: image_bytes)

    def crop(input_path, output_path, fmt, predictor, manual_corners):
        assert predictor is None
        assert manual_corners.tolist() == submitted_corners
        assert cv2.imwrite(str(output_path), image)
        return {
            "method": "manual",
            "corners": {"points": submitted_corners},
            "bbox": {"x": 10, "y": 10, "width": 80, "height": 60},
            "output_size": {"width": width, "height": height},
        }

    monkeypatch.setattr(service.cropper, "crop_painting", crop)
    monkeypatch.setattr(service.cropper, "build_predictor", lambda _: pytest.fail("SAM should be bypassed"))
    monkeypatch.setattr(
        service.matcher,
        "search_shot",
        lambda *_: {
            "golden": str(reference_path.resolve()),
            "match": "reference-1.png",
            "inliers": 82,
            "runner_up_inliers": 20,
            "channel_inliers": {"grey": 82, "red": 20, "green": 20, "blue": 20, "chroma": 20},
            "template_score_percent": 70.0,
        },
    )

    service.state.run_job(job_id)
    result = json.loads(service.state.job_path(job_id).read_text())
    assert result["status"] == "succeeded"
    assert result["result"]["status"] == "match"
    assert result["result"]["matched_reference_id"] == "reference-1"
    assert result["result"]["photos"][0]["crop"]["corners"] == submitted_corners
    assert Path(tmp_path / "jobs" / job_id / "photo-1_preview.jpg").is_file()