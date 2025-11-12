# Agent Onboarding Guide

Welcome! This document provides high-level context for future AI assistant sessions working on the coral bleaching classifier project.

## Project Snapshot
- **Goal**: Build a knowledge-distillation pipeline that classifies coral images into `bleached` vs `healthy`.
- **Primary Data Source**: Kaggle Coral Bleaching Dataset (see `data/README.md` for sourcing notes).
- **Code Layout**:
  - `models/`: Teacher, student, and distillation scaffolding (currently placeholders).
  - `utils/`: Planned helpers for dataloaders, preprocessing, metrics, and visualization.
  - `configs/`: YAML-based experiment settings plus guidelines for versioning.
  - Root scripts (`train_teacher.py`, `train_student_baseline.py`, `train_student_kd.py`, `evaluate.py`) expose CLI entry points.
  - `notebooks/`: Markdown stubs outlining exploratory and evaluation workflows.

## Repo Conventions
- Keep changes small and localized; add or update documentation alongside code.
- Never delete tests (once introduced). Write tests for non-trivial changes and ensure they pass.
- Large binaries (processed data, checkpoints, logs) must remain git-ignored. Prefer external storage or Git LFS for raw datasets.
- Environment setup, dataset handling, and next implementation steps are tracked in `README.md`.

## Collaboration Tips
- Favor absolute imports from root packages (`models`, `utils`) to match current script usage.
- Document new configuration files under `configs/` and update `configs/README.md` when introducing additional variants.
- When touching data pipelines, record reproducibility steps in `data/README.md`.
- Update this file if project goals or norms shift, so future agents have an accurate starting point.
