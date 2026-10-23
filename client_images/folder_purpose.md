# client_images/

Client photos of paintings to be cropped and matched.

## Usage

Put the photos here, then crop each one from the repository root:

```
venv/bin/python3 image_analysis/painting_crop_v7.py client_images/PHOTO.jpg
```

Each crop is written next to its photo as `PHOTO_painting_v7.tiff`. Search the crops
with `golden_match_v5.py` (see `readme.md`).

Client images are not in git (see `.gitignore`).
