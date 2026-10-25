# Base44 Artwork Analysis Backend

## Purpose

This repository provides the Python image-analysis backend for the Base44 artwork
verification app. It accepts client photographs, crops and perspective-corrects
the painting, then compares the resulting images with the app's golden-reference
vault. The result identifies visual similarity to a reference composition; it is
not proof of authorship or authenticity.

The authoritative algorithms are `image_analysis/painting_crop_v8.py` and
`image_analysis/golden_match_v5.py`. The FastAPI service in
`image_analysis/base44_service.py` wraps those functions without replacing the
matching logic with a Base44/JavaScript approximation. Base44 should own the user
experience, submission records, and private image storage, and call this service
for analysis.

Use [BASE44_BACKEND_PLAN.md](BASE44_BACKEND_PLAN.md) as the detailed API contract
and implementation reference when updating the Base44 app.

## Installation

### Docker (recommended)

Docker builds a CPU-only PyTorch image and runs one Uvicorn worker. From the repo
root:

```sh
docker build -t base44-image-analysis .
mkdir -p service_data models
```

Download the SAM ViT-B checkpoint into `models/`:

```sh
curl -fL -o models/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
sha256sum models/sam_vit_b_01ec64.pth
```

Expected SHA-256:
`ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`.

Start the service with persistent mounts, a strong secret, and the exact storage
hostnames from which it may download images:

```sh
docker run --rm --name base44-image-analysis \
  -p 8000:8000 \
  -e BASE44_SERVICE_TOKEN='set-a-long-random-secret' \
  -e IMAGE_URL_HOSTS='storage.example.com' \
  -e BASE44_PUBLIC_URL='https://analysis.example.com' \
  -v "$PWD/service_data:/data" \
  -v "$PWD/models:/models:ro" \
  base44-image-analysis
```

Set `BASE44_PUBLIC_URL` to the externally reachable service origin so preview
artifacts are returned as absolute URLs. Keep the service token server-side in
Base44; never expose it to a browser.

### Local Python

Python 3.12 is the documented runtime. Install dependencies and the checkpoint:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p models service_data
```

Download and verify the checkpoint using the commands above. Then configure and
start one worker:

```sh
export BASE44_SERVICE_TOKEN='set-a-long-random-secret'
export IMAGE_URL_HOSTS='storage.example.com'
export BASE44_PUBLIC_URL='http://127.0.0.1:8000'
export BASE44_DATA_DIR="$PWD/service_data"
.venv/bin/uvicorn image_analysis.base44_service:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

## Base44 Usage

All routes except `GET /health/live` require:

```http
Authorization: Bearer <BASE44_SERVICE_TOKEN>
```

Use this workflow from Base44:

1. Check `GET /health/ready` before accepting analysis work. It reports ready only
   when the checkpoint verifies and a vault revision is active.
2. Synchronize references with `PUT /v1/vault`. Send a stable vault revision and
   each reference's stable ID, HTTPS image URL, and SHA-256. URLs must use an
   allowlisted host. Sync only when the reference manifest or image content changes.
3. Submit `POST /v1/jobs` with one or more client photos, their stable IDs, private
   HTTPS image URLs, and optional manually adjusted corners. Corners are four
   full-resolution points in top-left, top-right, bottom-right, bottom-left order.
   Manual corners bypass SAM.
4. Poll `GET /v1/jobs/{job_id}` until `status` is `succeeded` or `failed`. Use the
   returned status, per-photo match details, crop corners, and protected preview
   URLs in the Base44 record and UI.
5. Treat `needs_corners` and processing failures distinctly from `no_match`. Never
   silently substitute another matching engine when this service is unavailable.

The service also provides `GET /v1/jobs/{job_id}/artifacts/{photo_id}.jpg` for
authenticated crop previews. Full request/response examples and the result schema
are in [BASE44_BACKEND_PLAN.md](BASE44_BACKEND_PLAN.md).

## Operational Limits

- Persist `/data` across restarts. It contains jobs, source images, crop artifacts,
  vault images, and generated indexes.
- Run exactly one service process against each data directory. Jobs are stored on
  disk and queued in-process; multiple replicas require a shared queue and storage
  implementation that this repo does not provide.
- Set `IMAGE_URL_HOSTS` narrowly. The service only downloads HTTPS URLs from those
  hosts, rejects non-public DNS results and redirects outside the allowlist, and
  enforces byte and pixel limits.
- Keep the vault revision used by a submitted job available until that job finishes.
- The template score is for display only. Inlier thresholds require calibration on
  known matches and non-matches; a visual match does not prove authenticity.
- This repo provides the service code and Docker image, not a running deployment.
  Deploy it on infrastructure capable of running OpenCV, PyTorch, and SAM, then
  configure the Base44 backend with its URL and server-side token.

## Development Checks

Install test dependencies and run the focused service suite:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

The tests cover authentication, image URL policy, image pixel limits, corner
validation, and manual-corner bypass. Validate numerical equivalence against fixed
real-image fixtures before changing algorithm behavior or claiming CLI equivalence.# Base44 image analysis

Artwork matching for client submissions. A client sends photos of a painting. The
painting is cropped and flattened, then matched against a vault of golden reference
paintings. The result is a similarity measure, not proof of authenticity.

For the planned Base44 integration, see [BASE44_BACKEND_PLAN.md](BASE44_BACKEND_PLAN.md).

## Layout

| Folder | Contents |
|---|---|
| `image_analysis/` | `painting_crop_v8.py` (crop) and `golden_match_v5.py` (match). The built index (`vault_index/`) and saved results are generated locally and are not in git. |
| `client_images/` | Client photos to process. Not in git. |
| `vault_images/` | Golden reference images (circular BMP, about 10 MB each). Not in git. |
| `models/` | `sam_vit_b_01ec64.pth`, the Segment Anything checkpoint. Not in git. |

Run commands from the repository root.

## Install

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

`requirements.txt` includes `segment-anything` from a pinned Git commit and `torch`,
which are needed only by the crop step. The matcher needs OpenCV and NumPy only.

Download the Segment Anything checkpoint used by the crop step:

```
mkdir -p models && curl -fL -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
sha256sum models/sam_vit_b_01ec64.pth
```

The expected SHA-256 is `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`
(375042383 bytes).

## Run the Base44 service

The service is a single-process FastAPI application that wraps the same crop and
matching functions used by the CLI. It keeps job and vault files in `BASE44_DATA_DIR`
(default `/data`), so mount that path on persistent storage. Run one Uvicorn worker
per data directory; the local worker queue is not shared between processes.

Set a long random `BASE44_SERVICE_TOKEN` and a comma-separated allowlist of image
storage hostnames in `IMAGE_URL_HOSTS`. Only HTTPS URLs on those hosts are accepted.
Mount the verified SAM checkpoint or set `SAM_MODEL_PATH`; readiness verifies its
SHA-256. Set `BASE44_PUBLIC_URL` to the externally reachable service origin when
Base44 needs absolute preview URLs. The queued-job limit defaults to 100 and can
be changed with `MAX_QUEUED_JOBS`. Example local run:

```
export BASE44_SERVICE_TOKEN='replace-with-a-long-random-secret'
export IMAGE_URL_HOSTS='your-storage.example.com'
export BASE44_PUBLIC_URL='https://analysis.example.com'
export BASE44_DATA_DIR="$PWD/service_data"
venv/bin/uvicorn image_analysis.base44_service:app --host 127.0.0.1 --port 8000 --workers 1
```

`GET /health/live` is unauthenticated. All other routes require the bearer token.
The service implements `GET /health/ready`, `PUT /v1/vault`, `POST /v1/jobs`,
`GET /v1/jobs/{job_id}`, and the protected crop-preview artifact route. See
[BASE44_BACKEND_PLAN.md](BASE44_BACKEND_PLAN.md) for request/response shapes and
operational limits. This filesystem-backed queue is intended for one service
instance with persistent storage; use a shared durable queue and database before
running multiple replicas.

## Step 1: Crop client photos

```
venv/bin/python3 image_analysis/painting_crop_v8.py client_images/PHOTO.jpg
```

Writes `client_images/PHOTO_painting_v8.tiff` and prints a JSON report. Options:

- `--format bmp` for BMP output instead of TIFF.
- `--model PATH` to use a different SAM checkpoint.
- `corners` as a final argument (`x,y x,y x,y x,y`, top-left, top-right, bottom-right, bottom-left) to override automatic detection.

The crop uses SAM to find the painting, then fits a quadrilateral to its outline and
rectifies it. A crop that fails the checks exits with status 1 and no output file.

## Step 2: Index the vault

Run once, and again whenever `vault_images/` changes:

# TMS: YES for indexing
```
venv/bin/python3 image_analysis/golden_match_v5.py build-vault vault_images image_analysis/vault_index --json > vault_index.json

```

Add `--json > image_analysis/vault_index.json` to save the summary and hide progress.
Building 70 goldens takes about 30 seconds and produces about 370 MB.

## Step 3: Match the crops

Search a folder of crops, or individual files:

# TMS: YES for matching
```
venv/bin/python3 image_analysis/golden_match_v5.py search image_analysis/vault_index client_images --json > result.js

```

Give the matcher the flattened crops (`*_painting_v8.tiff`), not the original photos.
Several crops of the same artwork can be searched together. A golden is reported as
the match only if it wins on at least half the shots.

Exit codes: `0` on success, including "no match". `1` when the vault is missing, no
images are found, or an image cannot be read.

## How matching works

1. Each golden and crop is resized (longer side 1000 px) and split into five channels: grey, red, green, blue, and a red chromaticity channel.
2. SIFT features are extracted from each channel. Goldens are masked to their inscribed circle; crops are not masked.
3. Each channel is matched against its vault index (FLANN), filtered by Lowe's ratio test (0.8).
4. For the goldens with the most matches, RANSAC fits a homography and counts inliers.
5. A golden's score is its best inlier count across channels. It is accepted if it has at least 50 inliers and at least 1.5 times the runner-up.
6. A template score (normalised cross-correlation of the golden inside the crop) is reported for display only. It is not a confidence measure.

The thresholds (50 inliers, 1.5 margin) were chosen on five candidates and are not calibrated against known non-matches.

## Limits

- Thresholds need calibration on known matches and known non-matches before any decision relies on them.
- A match shows the photo shows the same composition as a reference painting. It cannot show who painted the artwork. A good copy would also match. Treat a match as a prompt for expert review.
- Close-up crops give fewer features. Use them as supporting shots, not on their own.
- Rebuild the vault after any change to `vault_images/`. A stale index points at files that may no longer exist.
- The crop step needs the SAM checkpoint; the matcher does not.
