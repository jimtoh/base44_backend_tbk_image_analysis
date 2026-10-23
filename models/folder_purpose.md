# models/

Holds the Segment Anything checkpoint `sam_vit_b_01ec64.pth`. Only the crop step
(`image_analysis/painting_crop_v7.py`) uses it, to find the painting in a client photo.

## Usage

- Copy `sam_vit_b_01ec64.pth` into this folder. The crop script finds it here automatically.
- Or pass another checkpoint with `--model PATH`.

The checkpoint is not in git (see `.gitignore`).
