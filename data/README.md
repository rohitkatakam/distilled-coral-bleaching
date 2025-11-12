## Data Directory

This folder stores the datasets used for coral bleaching classification experiments.

- `raw/`: Immutable raw assets pulled from the Kaggle *Coral Bleaching Dataset* (https://www.kaggle.com/datasets/warcoder/coral-bleaching). Contains `bleached/` and `healthy/` image classes exactly as downloaded. Use Git LFS or an external bucket if tracking the raw images is not feasible in git.
  - **Total images**: 923 (485 bleached, 438 healthy)
  - **Class balance**: ~52.5% bleached, ~47.5% healthy (reasonably balanced)
- `processed/`: Intermediate artifacts produced by preprocessing or feature extraction pipelines. Keep this directory git-ignored; add metadata describing the generation steps when you populate it.
- `splits/`: Train/validation/test subsets derived from the raw data. Contains CSV manifest files (`train.csv`, `val.csv`, `test.csv`) with image paths and labels.
  - **Split ratios**: 70% train / 15% validation / 15% test
  - **Train set**: 645 images (339 bleached, 306 healthy)
  - **Validation set**: 139 images (73 bleached, 66 healthy)
  - **Test set**: 139 images (73 bleached, 66 healthy)
  - **Random seed**: 42 (for reproducibility)
  - **Created**: 2025-11-12
  - **Generation script**: `scripts/create_data_splits.py`
  - **Format**: CSV files with columns `image_path` (relative from project root), `label` (bleached/healthy)

When new datasets are added, document the source, download date, and licensing considerations in this file. Include the commands or scripts used to reproduce any processed artifacts so the data pipeline remains auditable.
