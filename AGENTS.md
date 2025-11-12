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

---

## Workflow Architecture

This project uses a **hybrid local/Colab workflow** due to GPU constraints:

### Local Environment (with Claude Code)
- **Purpose**: Code development, testing, evaluation, paper figure generation
- **Activities**:
  - Write models, training scripts, utilities, tests
  - Create and run evaluation notebooks
  - Generate visualizations and paper figures
  - Run evaluations on downloaded checkpoints (CPU-based)
- **Data**: Raw images stored locally (`data/raw/`) for quick testing
- **Git**: Primary development environment - commit and push all code here

### Google Colab Environment (without Claude)
- **Purpose**: GPU-accelerated training only
- **Activities**:
  - Clone repo from GitHub
  - Mount Google Drive for checkpoints/logs
  - Run training (teacher, student, distillation)
  - Save outputs to Drive
- **Data**: Raw images uploaded to Google Drive (read from Drive during training)
- **Important**: Colab notebook must be **simple and self-contained** - no AI assistance available, clear cell-by-cell instructions required

### Data Flow
1. **Raw Data**: Stored in both local `data/raw/` and Google Drive
2. **Data Splits**: Created locally with fixed seed, committed to Git (train/val/test manifests)
3. **Training**: Colab reads data from Drive, trains models, saves checkpoints to Drive
4. **Checkpoints**: Download from Drive to local `checkpoints/` for evaluation
5. **Logs**: Weights & Biases (wandb) for experiment tracking (accessible from both environments)

### Key Tooling
- **Experiment Tracking**: Weights & Biases (wandb)
- **Training Notebook**: Single Colab notebook with sections for teacher/student/KD training
- **Version Control**: GitHub (public repo for Colab access)

---

## Implementation Roadmap

### Phase 0: Foundation & Environment Setup
**Status**: NOT_STARTED
**Environment**: LOCAL
**Estimated Sessions**: 1-2

#### Goals
- Set up reproducible development environment
- Implement core data pipeline and utilities
- Create train/val/test splits

#### Tasks
1. **Dependencies**:
   - Create `requirements.txt` (CPU PyTorch for local development)
   - Create `requirements-colab.txt` (GPU PyTorch for Colab training)
   - Install local environment and verify

2. **Data Splitting** (LOCAL - COMMIT TO GIT):
   - Write script to split data into train/val/test (70/15/15 ratio)
   - Use fixed random seed for reproducibility
   - Generate split manifests (CSV/JSON with image paths and labels)
   - Save to `data/splits/` and commit to Git
   - Document split statistics in `data/README.md`

3. **Google Drive Setup** (USER ACTION - REQUIRED FOR COLAB TRAINING):
   - Create folder structure in Google Drive:
     ```
     /content/drive/MyDrive/coral-bleaching/
     ├── data/
     │   └── raw/
     │       ├── bleached/  (upload 485 images)
     │       └── healthy/   (upload 438 images)
     ├── checkpoints/  (create empty folder for training outputs)
     └── logs/  (optional, only if not using wandb)
     ```
   - **Upload to Google Drive**: Copy `data/raw/bleached/` and `data/raw/healthy/` folders with all images (~923 images total)
   - **Commit to Git**: Split manifests (`data/splits/*.csv`) are lightweight and go in Git, not Drive
   - **Important**: Ensure `.gitignore` allows `data/splits/*.csv` but blocks `data/raw/` (images)
   - **Path Resolution**: Split CSVs use relative paths (e.g., `data/raw/bleached/image.jpg`). In Colab, dataloaders will resolve these relative to the Drive mount point.

4. **Core Utilities**:
   - `utils/data_loader.py`: Dataset class (reads splits), dataloaders, augmentation pipeline
   - `utils/preprocessing.py`: Image transforms for train/val/test modes
   - `utils/metrics.py`: Accuracy, precision, recall, F1, confusion matrix, wandb helpers
   - `utils/visualization.py`: Training curves, confusion matrices, sample grids
   - `utils/env_utils.py`: Environment detection (Colab vs local), path resolution

5. **Testing**:
   - Write unit tests for data pipeline (test with local data)
   - Verify splits load correctly
   - Test augmentation pipeline

#### Deliverables
- [x] `requirements.txt` and `requirements-colab.txt` created
- [x] Data splits created and committed to Git
- [x] `.gitignore` updated to allow split CSVs
- [x] Google Drive folder structure created and raw images uploaded
- [ ] All utility modules implemented
- [ ] Unit tests passing
- [ ] Local data pipeline verified

#### Next Steps
→ Move to Phase 1: Teacher Model Implementation

---

### Phase 1: Teacher Model Implementation
**Status**: NOT_STARTED
**Environment**: LOCAL (code) → COLAB (training)
**Estimated Sessions**: 1-2 (local) + 1 training run (Colab)

#### Goals
- Implement teacher model (ResNet50) and training pipeline
- Create Colab setup documentation
- Train teacher model in Colab

#### Tasks
1. **Teacher Model** (LOCAL):
   - Implement `models/teacher.py`: ResNet50 with pretrained ImageNet weights
   - Support for config-driven architecture settings

2. **Training Script** (LOCAL):
   - Implement `train_teacher.py`:
     - Full training loop with wandb integration
     - Checkpoint saving (configurable path for Drive or local)
     - Early stopping based on validation loss
     - Learning rate scheduling (cosine annealing)
     - Support for resuming from checkpoint
   - Must work in both Colab (GPU) and local (CPU for testing)

3. **Colab Setup Documentation** (LOCAL):
   - Create `docs/colab_setup.md` with step-by-step instructions:
     - Clone GitHub repo
     - Install dependencies from `requirements-colab.txt`
     - Mount Google Drive
     - Set up wandb authentication
     - Configure paths for Drive-based data and checkpoints
     - Commands to run training sections
   - **Critical**: Instructions must be clear enough to follow without AI assistance

4. **Local Testing** (LOCAL):
   - Run 2-3 epoch training test locally (CPU, small subset)
   - Verify checkpoint saving and loading
   - Verify wandb logging works

5. **Training** (USER ACTION IN COLAB):
   - Clone repo and follow setup instructions
   - Run teacher training section
   - Monitor wandb dashboard
   - Save checkpoint to Google Drive: `coral-bleaching/checkpoints/teacher/best_model.pth`

#### Deliverables
- [ ] `models/teacher.py` implemented
- [ ] `train_teacher.py` implemented and tested locally
- [ ] `docs/colab_setup.md` created with clear instructions
- [ ] Code pushed to GitHub
- [ ] Teacher model trained in Colab
- [ ] Teacher checkpoint saved to Drive: `checkpoints/teacher/best_model.pth`
- [ ] Wandb run completed with training logs

#### Next Steps
→ Move to Phase 2: Teacher Evaluation & Analysis

---

### Phase 2: Teacher Evaluation & Analysis
**Status**: NOT_STARTED
**Environment**: LOCAL
**Estimated Sessions**: 1

#### Goals
- Download and evaluate teacher model locally
- Conduct exploratory data analysis
- Generate baseline results for paper

#### Tasks
1. **Checkpoint Download**:
   - Download teacher checkpoint from Drive to local `checkpoints/teacher/`
   - Verify checkpoint loads correctly

2. **Evaluation Script** (LOCAL):
   - Implement `evaluate.py`:
     - Load model checkpoint (from any path)
     - Run inference on test set (CPU-compatible)
     - Compute all metrics (accuracy, precision, recall, F1)
     - Generate confusion matrix
     - Save results to JSON/CSV for reproducibility

3. **Data Exploration Notebook** (LOCAL):
   - Create `notebooks/01_data_exploration.ipynb`:
     - Load and visualize split statistics
     - Class balance analysis
     - Sample images from both classes (bleached vs healthy)
     - Image resolution and quality assessment
     - Document dataset characteristics

4. **Teacher Evaluation Notebook** (LOCAL):
   - Create `notebooks/02_teacher_evaluation.ipynb`:
     - Load teacher evaluation results
     - Visualize training curves from wandb
     - Display confusion matrix
     - Error analysis (visualize misclassifications)
     - Per-class performance breakdown
     - (Optional) Grad-CAM visualizations if feasible on CPU

#### Deliverables
- [ ] Teacher checkpoint downloaded and verified
- [ ] `evaluate.py` implemented
- [ ] Teacher test metrics computed and saved
- [ ] `notebooks/01_data_exploration.ipynb` completed
- [ ] `notebooks/02_teacher_evaluation.ipynb` completed
- [ ] **PAPER ARTIFACT**: Teacher baseline results (accuracy, confusion matrix, training curves)

#### Paper Contributions
- Dataset statistics and characteristics
- Teacher model baseline performance
- Error analysis informing future improvements

#### Next Steps
→ Move to Phase 3: Student Baseline

---

### Phase 3: Student Baseline Implementation & Training
**Status**: NOT_STARTED
**Environment**: LOCAL (code) → COLAB (training) → LOCAL (eval)
**Estimated Sessions**: 1-2 (local code) + 1 training run (Colab) + 1 (local eval)

#### Goals
- Implement lightweight student model
- Train student independently (no distillation)
- Establish baseline student performance to measure distillation gains

#### Tasks
1. **Student Model** (LOCAL):
   - Implement `models/student.py`: MobileNetV3-Small architecture
   - Support for both baseline and distillation training modes

2. **Student Training Script** (LOCAL):
   - Implement `train_student_baseline.py`:
     - Training loop similar to teacher (for fair comparison)
     - Use same hyperparameters (epochs, batch size, optimizer)
     - Wandb logging
     - Checkpoint saving to Drive

3. **Colab Instructions Update** (LOCAL):
   - Update `docs/colab_setup.md` with student baseline training section
   - Ensure instructions are clear and standalone

4. **Training** (USER ACTION IN COLAB):
   - Run student baseline training section in Colab notebook
   - Monitor wandb
   - Save checkpoint to Drive: `checkpoints/student_baseline/best_model.pth`

5. **Evaluation** (LOCAL):
   - Download student baseline checkpoint
   - Run `evaluate.py` on student baseline
   - Create `notebooks/03_student_baseline_comparison.ipynb`:
     - Compare teacher vs student baseline metrics
     - Analyze performance gap (expected 5-10% accuracy drop)
     - Compare model sizes (parameters, disk size)
     - Compare inference speed (if possible on CPU)

#### Deliverables
- [ ] `models/student.py` implemented
- [ ] `train_student_baseline.py` implemented
- [ ] Code pushed to GitHub
- [ ] Student baseline trained in Colab
- [ ] Student baseline checkpoint saved to Drive
- [ ] Student baseline evaluated locally
- [ ] `notebooks/03_student_baseline_comparison.ipynb` completed
- [ ] **PAPER ARTIFACT**: Baseline comparison table (teacher vs student)

#### Paper Contributions
- Establish student model capacity limitations
- Quantify teacher-student performance gap
- Motivate need for knowledge distillation

#### Next Steps
→ Move to Phase 4: Knowledge Distillation

---

### Phase 4: Knowledge Distillation Implementation & Training
**Status**: NOT_STARTED
**Environment**: LOCAL (code) → COLAB (training) → LOCAL (eval)
**Estimated Sessions**: 2 (local code) + 1 training run (Colab) + 1 (local eval)

#### Goals
- Implement knowledge distillation mechanism
- Train student with teacher guidance
- Demonstrate distillation effectiveness

#### Tasks
1. **Distillation Implementation** (LOCAL):
   - Implement `models/distillation.py`:
     - KL divergence loss with temperature scaling
     - Combined loss function: `L = α * L_distill + (1-α) * L_hard`
     - Support for temperature (T) and alpha (α) from config
     - Logging for both loss components

2. **KD Training Script** (LOCAL):
   - Implement `train_student_kd.py`:
     - Load frozen teacher model from checkpoint
     - Initialize student model
     - Train with distillation loss
     - Log distillation loss and hard label loss separately to wandb
     - Save checkpoints to Drive

3. **Colab Instructions Update** (LOCAL):
   - Update `docs/colab_setup.md` with KD training section
   - Document how to specify teacher checkpoint path

4. **Training with Default Hyperparameters** (USER ACTION IN COLAB):
   - Train with T=4.0, α=0.7 (from `configs/config.yaml`)
   - Monitor both loss components in wandb
   - Save checkpoint to Drive: `checkpoints/student_kd/best_model_t4.0_a0.7.pth`

5. **Evaluation** (LOCAL):
   - Download distilled student checkpoint
   - Run `evaluate.py` on distilled student
   - Create `notebooks/04_distillation_results.ipynb`:
     - Three-way comparison: Teacher | Student Baseline | Distilled Student
     - Performance improvement from distillation
     - Analyze gap closure (target: 50-70% of teacher-student gap)

#### Deliverables
- [ ] `models/distillation.py` implemented
- [ ] `train_student_kd.py` implemented
- [ ] Code pushed to GitHub
- [ ] Distilled student trained in Colab (T=4.0, α=0.7)
- [ ] Distilled student checkpoint saved to Drive
- [ ] Distilled student evaluated locally
- [ ] `notebooks/04_distillation_results.ipynb` completed
- [ ] **PAPER ARTIFACT**: Main results table (3-model comparison)

#### Paper Contributions
- Core distillation results
- Main claim: distillation bridges performance gap
- Foundation for ablation studies

#### Next Steps
→ Move to Phase 5: Hyperparameter Ablation Studies

---

### Phase 5: Hyperparameter Ablation Studies
**Status**: NOT_STARTED
**Environment**: LOCAL (code/configs) → COLAB (multiple training runs) → LOCAL (analysis)
**Estimated Sessions**: 1 (setup) + 9 training runs (Colab) + 2 (analysis)

#### Goals
- Systematically study impact of temperature (T) and alpha (α)
- Generate ablation figures for paper
- Identify optimal hyperparameters

#### Tasks
1. **Config Variants** (LOCAL):
   - Create config variants in `configs/`:
     - Temperature ablation: T ∈ {1, 2, 4, 8, 16} (5 configs)
     - Alpha ablation: α ∈ {0.3, 0.5, 0.7, 0.9} (4 configs)
   - Document configs in `configs/README.md`

2. **Temperature Ablation** (USER ACTION IN COLAB):
   - Train 5 distilled students with different temperatures (α=0.7 fixed)
   - Save checkpoints: `checkpoints/student_kd/ablation_t{temp}_a0.7.pth`
   - Track all runs in wandb with tags

3. **Alpha Ablation** (USER ACTION IN COLAB):
   - Train 4 distilled students with different alphas (T=4.0 fixed)
   - Save checkpoints: `checkpoints/student_kd/ablation_t4.0_a{alpha}.pth`
   - Track all runs in wandb

4. **Evaluation** (LOCAL):
   - Download all ablation checkpoints
   - Run `evaluate.py` on all models
   - Create `notebooks/05_ablation_studies.ipynb`:
     - Temperature vs accuracy curve
     - Alpha vs accuracy curve
     - Loss component analysis (distillation vs hard label)
     - Identify optimal hyperparameters
     - Statistical analysis if multiple runs available

5. **Optional: Architecture Variants** (LOCAL + COLAB):
   - Implement alternative student architectures (e.g., MobileNetV3-Large, EfficientNet-B0)
   - Train with optimal hyperparameters
   - Compare architectures

#### Deliverables
- [ ] Config variants created (9 total: 5 temperature + 4 alpha)
- [ ] All ablation models trained in Colab (9 training runs)
- [ ] All checkpoints downloaded and evaluated
- [ ] `notebooks/05_ablation_studies.ipynb` completed
- [ ] **PAPER ARTIFACT**: Temperature sensitivity curve
- [ ] **PAPER ARTIFACT**: Alpha sensitivity curve
- [ ] **PAPER ARTIFACT**: (Optional) Architecture comparison table

#### Paper Contributions
- Hyperparameter sensitivity analysis
- Guidance for practitioners using distillation
- Robustness of distillation approach

#### Next Steps
→ Move to Phase 6: Deep Analysis & Visualization

---

### Phase 6: Deep Analysis & Paper Visualizations
**Status**: NOT_STARTED
**Environment**: LOCAL
**Estimated Sessions**: 2

#### Goals
- Generate comprehensive analysis for paper
- Create publication-quality figures
- Perform interpretability analysis

#### Tasks
1. **Comprehensive Evaluation Notebook** (LOCAL):
   - Create `notebooks/06_comprehensive_analysis.ipynb`:
     - Load all model checkpoints (teacher, student baseline, distilled student, ablation models)
     - Side-by-side predictions on same test samples
     - Per-class performance breakdown (bleached vs healthy)
     - Confidence distribution analysis
     - Calibration analysis (reliability diagrams)
     - Failure case analysis

2. **Interpretability Analysis** (LOCAL):
   - Generate Grad-CAM visualizations:
     - Teacher model attention
     - Student baseline attention
     - Distilled student attention
   - Compare attention patterns across models:
     - Does distilled student learn similar features to teacher?
     - Visual evidence of knowledge transfer
   - Identify cases where distillation helps vs doesn't help
   - Create qualitative comparison figure for paper

3. **Efficiency Analysis** (LOCAL):
   - Measure model efficiency metrics:
     - Model sizes (parameters, disk size)
     - Inference time (CPU-based)
     - FLOPs calculation
   - Create efficiency vs accuracy tradeoff plot
   - Discuss practical deployment considerations

#### Deliverables
- [ ] `notebooks/06_comprehensive_analysis.ipynb` completed
- [ ] **PAPER ARTIFACT**: Qualitative comparison figure (Grad-CAM visualizations)
- [ ] **PAPER ARTIFACT**: Efficiency vs accuracy tradeoff plot
- [ ] **PAPER ARTIFACT**: Per-class performance breakdown
- [ ] **PAPER ARTIFACT**: Confidence distribution plots

#### Paper Contributions
- Qualitative evidence of knowledge transfer
- Interpretability analysis
- Practical deployment considerations
- Comprehensive performance analysis

#### Next Steps
→ Move to Phase 7: Statistical Validation (if needed)

---

### Phase 7: Statistical Validation (Optional)
**Status**: NOT_STARTED
**Environment**: COLAB (training) → LOCAL (analysis)
**Estimated Sessions**: 3-5 training runs + 1 analysis

#### Goals
- Establish statistical significance of results
- Provide confidence intervals for paper

#### Tasks
1. **Multiple Runs** (USER ACTION IN COLAB):
   - Train 3-5 runs of key models with different random seeds:
     - Teacher (1 run sufficient if stable)
     - Student baseline (3-5 runs)
     - Distilled student with optimal hyperparameters (3-5 runs)
   - Save all checkpoints with run IDs

2. **Statistical Analysis** (LOCAL):
   - Evaluate all runs
   - Compute mean ± standard deviation for all metrics
   - Perform statistical significance tests (t-test, Wilcoxon)
   - Create box plots for performance distributions
   - Update all paper tables with confidence intervals

#### Deliverables
- [ ] Multiple runs completed for key models
- [ ] Statistical analysis completed
- [ ] All paper results updated with error bars/confidence intervals

#### Paper Contributions
- Statistical rigor
- Confidence in reported results
- Reproducibility evidence

#### Next Steps
→ Move to Phase 8: Reproducibility & Documentation

---

### Phase 8: Reproducibility & Final Documentation
**Status**: NOT_STARTED
**Environment**: LOCAL
**Estimated Sessions**: 1

#### Goals
- Ensure codebase is reproducible and well-documented
- Prepare for paper submission and code release

#### Tasks
1. **Testing** (LOCAL):
   - Ensure all unit tests pass
   - Add integration tests for training pipelines (1-2 epoch runs)
   - Verify all notebooks run end-to-end without errors

2. **Documentation** (LOCAL):
   - Update `AGENTS.md` with final project status
   - Document all hyperparameter choices and rationale
   - Create model cards for all trained models (architecture, performance, usage)
   - Add reproducibility checklist:
     - Random seeds used
     - Data split methodology
     - Training hyperparameters
     - Hardware specifications
     - Dependency versions

3. **Code Organization** (LOCAL):
   - Clean up checkpoint directory structure
   - Standardize naming conventions
   - Archive all training configs with checkpoints
   - Export key wandb charts for paper (as images/PDFs)

4. **README Update** (LOCAL):
   - Update main README with:
     - Project overview and results summary
     - Setup instructions (local + Colab)
     - Usage examples
     - Links to paper (when available)
     - Citation information

#### Deliverables
- [ ] All tests passing
- [ ] All notebooks verified to run end-to-end
- [ ] Complete documentation updated
- [ ] Reproducibility checklist created
- [ ] Model cards created
- [ ] README.md updated with results
- [ ] **PAPER ARTIFACT**: Camera-ready codebase for submission

#### Paper Contributions
- Code availability for reviewers
- Reproducibility for community
- Open science contribution

#### Next Steps
→ Project complete! Ready for paper writing and submission.

---

## Progress Tracking

This section is updated after each session to track overall progress and maintain continuity across sessions.

### Current Status
- **Active Phase**: Phase 0 (Foundation & Environment Setup)
- **Phase Status**: IN_PROGRESS (4/5 tasks completed)
- **Last Updated**: 2025-11-12

### Completed Tasks (Phase 0)
- ✅ Created `requirements.txt` and `requirements-colab.txt`
- ✅ Installed local environment (all dependencies working)
- ✅ Implemented data splitting script (`scripts/create_data_splits.py`)
- ✅ Generated train/val/test splits (645/139/139 images, random seed 42)
- ✅ Committed split manifests to Git (`data/splits/*.csv`)
- ✅ Updated `data/README.md` with split statistics
- ✅ Updated `.gitignore` to allow split CSVs in Git
- ✅ Uploaded raw images to Google Drive (923 images in bleached/healthy folders)
- ✅ Created checkpoint directory structure in Google Drive

### Completed Training Runs
None yet.

### Available Checkpoints (in Google Drive)
None yet.

### Available Checkpoints (downloaded locally)
None yet.

### Current Blockers
None.

### Next Immediate Action
Implement core utilities for Phase 0:
- `utils/data_loader.py`: Dataset class (reads split CSVs), dataloaders, augmentation pipeline
- `utils/preprocessing.py`: Image transforms for train/val/test modes
- `utils/metrics.py`: Accuracy, precision, recall, F1, confusion matrix, wandb helpers
- `utils/visualization.py`: Training curves, confusion matrices, sample grids
- `utils/env_utils.py`: Environment detection (Colab vs local), path resolution

Then write unit tests for data pipeline to complete Phase 0.

### Notes
- Project roadmap finalized with hybrid local/Colab workflow
- Workflow optimized for no-GPU local environment + Colab training
- Emphasis on simple, self-contained Colab notebook (no AI assistance needed)
- **Google Drive documentation added**: See Phase 0, Task 3 for upload instructions
- Data splits use relative paths for portability between local and Colab environments

---

## Quick Reference: Workflow Summary

**LOCAL (with Claude Code):**
1. Write/edit code (models, utils, scripts, notebooks)
2. Write unit tests
3. Push to GitHub
4. Download checkpoints from Drive
5. Run evaluations (CPU-based)
6. Generate paper figures

**COLAB (without Claude):**
1. Clone repo from GitHub
2. Mount Google Drive
3. Run training section in notebook
4. Monitor wandb dashboard
5. Checkpoints auto-save to Drive

**Handoff Points:**
- LOCAL → COLAB: After pushing code updates to GitHub
- COLAB → LOCAL: After training completes and checkpoints are saved to Drive

**Key Paths:**
- **Local repo**: `/Users/rohitkatakam/projects/distilled-coral-bleaching/`
- **Google Drive** (Colab): `/content/drive/MyDrive/coral-bleaching/`
- **Checkpoints** (Drive): `/content/drive/MyDrive/coral-bleaching/checkpoints/`
- **Logs** (wandb): Cloud-based, accessible from both environments
