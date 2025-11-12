# Coral Bleaching Classifier

Knowledge distillation project for classifying coral imagery into bleached vs healthy categories. The repository is scaffolded for teacher–student training, experiment tracking, and reproducible data management.

## Data Sources

- Primary dataset: [Coral Bleaching Dataset on Kaggle](https://www.kaggle.com/datasets/warcoder/coral-bleaching). Download the archive and place the class folders inside `data/raw/` (see `data/README.md`).
- If raw assets are too large for git, prefer external storage (S3, GCS, Azure blob) or Git LFS. Keep references or download scripts under version control rather than committing large binaries.
- Store any Kaggle API credentials outside the repository. Recommended approach:
  - Set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.
  - Or place `kaggle.json` under `~/.kaggle/` with 600 permissions.

## Directory Structure

```
.
├── configs/              # YAML configs and guidelines for managing experiment variants
├── data/                 # Raw and derived datasets (see data/README.md)
├── models/               # Teacher, student, and distillation scaffolding
├── notebooks/            # Exploratory analyses and evaluation summaries
├── utils/                # Shared helper modules (dataloaders, preprocessing, metrics, viz)
├── train_teacher.py      # CLI entry point to train the teacher
├── train_student_baseline.py
├── train_student_kd.py
├── evaluate.py           # CLI entry point to run evaluations
└── requirements.txt      # Placeholder dependency list
```

We currently expose packages (`models/`, `utils/`) at the repository root to keep CLI imports straightforward (e.g., `from models.teacher import TeacherModel`). Revisit a dedicated `src/` layout if we need to publish the code as an installable package.

## Environment Setup

1. Create and activate a Python environment (3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies (versions to be finalized during implementation):
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables for data access and experiment tracking as needed (e.g., `KAGGLE_USERNAME`, `KAGGLE_KEY`, `WANDB_API_KEY`).

## Next Steps

- Implement teacher training loop in `train_teacher.py` and define the architecture in `models/teacher.py`.
- Build student baselines (`train_student_baseline.py`) and knowledge distillation pipeline (`train_student_kd.py`).
- Flesh out utilities under `utils/` for preprocessing, dataloaders, metrics, and visualization.
- Connect configuration handling (`configs/config.yaml`) to the command-line interfaces.
- Add automated tests and continuous integration once core functionality is stable.

Contributions should remain small and focused. Add or update documentation alongside code changes to keep the scaffold in sync with implementation progress.