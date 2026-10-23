# vault_images/

Golden reference paintings: the circular BMP images (about 10 MB each) that client
crops are matched against. Byte-identical duplicates are skipped when the index is built.

## Usage

Add or replace golden images here, then rebuild the index from the repository root:

```
venv/bin/python3 image_analysis/golden_match_v5.py build-vault vault_images image_analysis/vault_index
```

Rebuild after any change to this folder. A stale index points at files that may no longer exist.

Golden images are not in git (see `.gitignore`).
