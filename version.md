# Version history

Current scripts live in `image_analysis/`. Earlier versions were removed from the
repository and remain in git history.

| Script | Summary |
|---|---|
| `painting_crop_v5` (removed) | Segment Anything (ViT-B) with a central box prompt, then a four-sided fit to the mask. Its logic is in `painting_crop_v7.py`. |
| `painting_crop_v6` (removed) | Ran v5 and cut optional centred squares for the template matcher. Squares were dropped because nothing uses them. |
| `image_analysis/painting_crop_v7.py` | **Current.** Standalone v5 crop. Writes `<photo>_painting_v7.<format>`. No square crops. |
| `golden_match.py`, `golden_match_v2.py`, `golden_match_v3.py`, `golden_match_v4.py` (removed) | Earlier matchers. v2 template matcher; v3 SIFT vault; v4 five-channel vault. |
| `image_analysis/golden_match_v5.py` | **Current.** Standalone v4 matcher. The template score is inlined, so it has no sibling imports. |
