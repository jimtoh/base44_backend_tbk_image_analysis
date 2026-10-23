# models/

Holds the Segment Anything checkpoint `sam_vit_b_01ec64.pth`. Only the crop step
(`image_analysis/painting_crop_v7.py`) uses it, to find the painting in a client photo.

## Usage

- Copy `sam_vit_b_01ec64.pth` into this folder. The crop script finds it here automatically.
- Or pass another checkpoint with `--model PATH`.

The checkpoint is not in git (see `.gitignore`).

## Download

From the repository root, download the checkpoint directly:

```
mkdir -p models
curl -fL -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Verify it (expected SHA-256 `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`, size 375042383 bytes):

```
sha256sum models/sam_vit_b_01ec64.pth
```
