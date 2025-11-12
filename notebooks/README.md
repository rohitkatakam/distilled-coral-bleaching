## Notebook Workspace

- Use this directory for exploratory analyses, visualizations, and experiment summaries. Follow the naming convention `NN_topic.md` or `NN_topic.ipynb` to keep notebooks ordered chronologically.
- When using Jupyter notebooks, set the kernel to the project virtual environment and ensure environment variables (e.g., `KAGGLE_USERNAME`, `KAGGLE_KEY`) are configured for data access.
- Avoid committing large intermediate outputs within notebooks. Persist necessary assets (plots, tables) to `logs/` or `reports/` and link them from the notebook instead.
- For reproducibility, capture the exact configuration (`configs/*.yaml`) and dataset version used in each session. Consider exporting executed notebooks as HTML/PDF for reporting, but store those artifacts outside of git if they become large.
