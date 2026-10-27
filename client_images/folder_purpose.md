# client_images/

Preprocessed client photos: each original in `client_pre_images/` is cropped to
the painting and perspective-flattened by `image_analysis/painting_crop_v8.py`
(Segment Anything ViT-B via SamPredictor finds the painting mask, a quadrilateral
is fitted to it, and the region is rectified to face the camera).

Files here are `<original>_painting_v8.tiff`, lossless, ready for
`golden_match_v5.py` search or the ORB pipeline.

Regenerate with:

```
python3 image_analysis/painting_crop_v8.py client_pre_images
```

Crops are written to `client_pre_images/output/`, then copied here.
Client images are not in git (see `.gitignore`).
