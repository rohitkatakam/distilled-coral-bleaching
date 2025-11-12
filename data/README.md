## Data Directory

This folder stores the datasets used for coral bleaching classification experiments.

- `raw/`: Immutable raw assets pulled from the Kaggle *Coral Bleaching Dataset* (https://www.kaggle.com/datasets/warcoder/coral-bleaching). Contains `bleached/` and `healthy/` image classes exactly as downloaded. Use Git LFS or an external bucket if tracking the raw images is not feasible in git.
- `processed/`: Intermediate artifacts produced by preprocessing or feature extraction pipelines. Keep this directory git-ignored; add metadata describing the generation steps when you populate it.
- `splits/`: Train/validation/test subsets derived from the raw data. Each split (`train/`, `val/`, `test/`) should mirror the class directory layout and may include manifest files describing the split logic.

When new datasets are added, document the source, download date, and licensing considerations in this file. Include the commands or scripts used to reproduce any processed artifacts so the data pipeline remains auditable.
