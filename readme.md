# Base44 image analysis

Artwork matching for client submissions. A client sends photos of a painting. The
painting is cropped and flattened, then matched against a vault of golden reference
paintings. The result is a similarity measure, not proof of authenticity.

## Layout

| Folder | Contents |
|---|---|
| `image_analysis/` | `painting_crop_v7.py` (crop) and `golden_match_v5.py` (match). The built index (`vault_index/`) and saved results are generated locally and are not in git. |
| `client_images/` | Client photos to process. Not in git. |
| `vault_images/` | Golden reference images (circular BMP, about 10 MB each). Not in git. |
| `models/` | `sam_vit_b_01ec64.pth`, the Segment Anything checkpoint. Not in git. |

Run commands from the repository root.

## Install

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

`requirements.txt` includes `segment-anything` from GitHub and `torch`, which are
needed only by the crop step. The matcher needs OpenCV and NumPy only.

Download the Segment Anything checkpoint used by the crop step:

```
mkdir -p models && curl -fL -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
sha256sum models/sam_vit_b_01ec64.pth
```

The expected SHA-256 is `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`
(375042383 bytes).

## Step 1: Crop client photos

```
venv/bin/python3 image_analysis/painting_crop_v7.py client_images/PHOTO.jpg
```

Writes `client_images/PHOTO_painting_v7.tiff` and prints a JSON report. Options:

- `--format bmp` for BMP output instead of TIFF.
- `--model PATH` to use a different SAM checkpoint.
- `corners` as a final argument (`x,y x,y x,y x,y`, top-left, top-right, bottom-right, bottom-left) to override automatic detection.

The crop uses SAM to find the painting, then fits a quadrilateral to its outline and
rectifies it. A crop that fails the checks exits with status 1 and no output file.

## Step 2: Index the vault

Run once, and again whenever `vault_images/` changes:

```
venv/bin/python3 image_analysis/golden_match_v5.py build-vault vault_images image_analysis/vault_index
```

Add `--json > image_analysis/vault_index.json` to save the summary and hide progress.
Building 70 goldens takes about 30 seconds and produces about 370 MB.

## Step 3: Match the crops

Search a folder of crops, or individual files:

```
venv/bin/python3 image_analysis/golden_match_v5.py search image_analysis/vault_index client_images
venv/bin/python3 image_analysis/golden_match_v5.py search image_analysis/vault_index client_images --json > image_analysis/results.js
```

Give the matcher the flattened crops (`*_painting_v7.tiff`), not the original photos.
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
