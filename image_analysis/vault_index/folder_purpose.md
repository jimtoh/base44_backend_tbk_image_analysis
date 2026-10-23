# image_analysis/vault_index/

The searchable index built from `vault_images/`. The matcher reads it when searching.
Build tool: `image_analysis/golden_match_v5.py build-vault`.

## Usage

Build the index from the repository root:

```
venv/bin/python3 image_analysis/golden_match_v5.py build-vault vault_images image_analysis/vault_index

or 

.venv/bin/python3 image_analysis/golden_match_v5.py build-vault vault_images image_analysis/vault_index
```

To save the build summary as JSON and hide progress, add `--json > image_analysis/vault_index.json`.

The index is generated, so it is not in git. For 70 goldens it is about 370 MB. Keep it on
persistent storage when the service runs, and rebuild it whenever `vault_images/` changes.
