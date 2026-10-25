# Base44 image analysis

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
