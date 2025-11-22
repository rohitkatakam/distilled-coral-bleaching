# Agent Onboarding Guide

Welcome! This document provides high-level context for future AI assistant sessions working on the coral bleaching classifier project.

## Project Snapshot
- **Goal**: Build a knowledge-distillation pipeline that classifies coral images into `bleached` vs `healthy`.
- **Primary Data Source**: Kaggle Coral Bleaching Dataset (see `data/README.md` for sourcing notes).
- **Code Layout**:
  - `models/`: Teacher, student, and distillation scaffolding.
  - `utils/`: Helpers for dataloaders, preprocessing, metrics, and visualization.
  - `configs/`: YAML-based experiment settings plus guidelines for versioning.
  - Root scripts (`train_teacher.py`, `train_student_baseline.py`, `train_student_kd.py`) expose training CLI entry points.
  - `scripts/`: Utility scripts for data processing, evaluation, and analysis.

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
- **Local Development**: Python scripts in `scripts/` directory for evaluation and analysis

---

## Implementation Roadmap

### Phase 0: Foundation & Environment Setup
**Status**: ✅ COMPLETE
**Environment**: LOCAL
**Completed**: 2025-11-13 (2 sessions)

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

4. **Core Utilities** (implement and test one at a time):

   a. `utils/env_utils.py`: Environment detection (Colab vs local), path resolution
      - `is_colab()` - detect Colab environment
      - `get_project_root()` - return project root path
      - `resolve_data_path()` - resolve relative paths from CSVs to actual files
      - `resolve_checkpoint_path()` - handle checkpoint paths for Drive vs local

   b. `utils/preprocessing.py`: Image transforms for train/val/test modes
      - `get_train_transforms(config)` - augmentation pipeline
      - `get_val_transforms(config)` - validation transforms
      - `get_test_transforms(config)` - test transforms

   c. `utils/data_loader.py`: Dataset class (reads splits), dataloaders, augmentation pipeline
      - `CoralDataset` class - read CSVs, load images, apply transforms
      - `build_dataloaders(config, split)` - create DataLoaders from config

   d. `utils/metrics.py`: Accuracy, precision, recall, F1, confusion matrix, wandb helpers
      - `compute_accuracy()`, `compute_confusion_matrix()`, `compute_classification_metrics()`
      - `log_metrics_to_wandb()` - wandb logging helper

   e. `utils/visualization.py`: Training curves, confusion matrices, sample grids
      - `plot_training_curves()`, `plot_confusion_matrix()`, `plot_sample_grid()`

5. **Testing** (write tests alongside each utility):
   - Create `tests/` directory structure
   - Write unit tests for each utility module
   - Verify splits load correctly
   - Test augmentation pipeline
   - Run complete test suite with `pytest`

#### Deliverables
- [x] `requirements.txt` and `requirements-colab.txt` created
- [x] Data splits created and committed to Git
- [x] `.gitignore` updated to allow split CSVs
- [x] Google Drive folder structure created and raw images uploaded
- [x] `utils/env_utils.py` implemented and tested (26 tests passing)
- [x] `utils/preprocessing.py` implemented and tested (32 tests passing, including 3 real coral image integration tests)
- [x] `utils/data_loader.py` implemented and tested (31 tests passing)
- [x] `utils/metrics.py` implemented and tested (45 tests passing, including 3 real wandb/model integration tests)
- [x] `utils/visualization.py` implemented and tested (23 tests passing, including 4 real plotting integration tests)
- [x] All unit tests passing (161 total tests)
- [x] Local data pipeline verified (4 end-to-end pipeline tests)

#### Next Steps
→ Move to Phase 1: Teacher Model Implementation

---

### Phase 1: Teacher Model Implementation
**Status**: ✅ COMPLETE
**Environment**: LOCAL (code) → COLAB (training)
**Completed**: 2025-11-14 (local code), 2025-11-16 (Colab training)

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
- [x] `models/teacher.py` implemented (25 tests passing)
- [x] `train_teacher.py` implemented and tested locally (25 tests passing)
- [x] `docs/colab_setup.md` created with clear instructions
- [x] Code pushed to GitHub
- [x] Teacher model trained in Colab (19/50 epochs, early stopping)
- [x] Teacher checkpoint saved to Drive: `checkpoints/teacher/best_model.pth`
- [x] W&B run completed with training logs (run ID: lfidb03f)

#### Next Steps
→ Move to Phase 2: Teacher Evaluation & Analysis

---

### Phase 2: Teacher Evaluation & Analysis
**Status**: ✅ COMPLETE
**Environment**: LOCAL
**Completed**: 2025-11-21 (1 session)

#### Goals
- Download and evaluate teacher model locally
- Conduct exploratory data analysis
- Generate baseline results for paper

#### Tasks
1. **Checkpoint Download**:
   - Download teacher checkpoint from Drive to local `checkpoints/teacher/`
   - Verify checkpoint loads correctly

2. **Evaluation Script** (LOCAL):
   - Implement `scripts/evaluate.py`:
     - Load model checkpoint (from any path)
     - Run inference on test set (CPU-compatible)
     - Compute all metrics (accuracy, precision, recall, F1)
     - Generate confusion matrix
     - Save results to JSON in `scripts/results/{model_name}/`

3. **Data Exploration Script** (LOCAL):
   - Create `scripts/explore_data.py` (Python script with optional `# %%` cell markers for interactive development):
     - Load and visualize split statistics
     - Class balance analysis
     - Sample images from both classes (bleached vs healthy)
     - Image resolution and quality assessment
     - Document dataset characteristics
     - Save plots to `scripts/results/data_exploration/`

4. **Teacher Evaluation Script** (LOCAL):
   - Create `scripts/evaluate_teacher.py` (Python script with optional `# %%` cell markers):
     - Load teacher evaluation results from `scripts/results/teacher/`
     - Visualize training curves from wandb
     - Display confusion matrix
     - Error analysis (visualize misclassifications)
     - Per-class performance breakdown
     - (Optional) Grad-CAM visualizations if feasible on CPU
     - Save plots to `scripts/results/teacher/`

#### Deliverables
- [x] Teacher checkpoint downloaded and verified
- [x] `scripts/evaluate.py` implemented
- [x] Teacher test metrics computed and saved
- [x] `scripts/explore_data.py` completed with plots saved
- [x] `scripts/evaluate_teacher.py` completed with plots saved
- [x] **PAPER ARTIFACT**: Teacher baseline results (accuracy, confusion matrix, training curves)

#### Paper Contributions
- Dataset statistics and characteristics
- Teacher model baseline performance
- Error analysis informing future improvements

#### Next Steps
→ Move to Phase 3: Student Baseline

---

### Phase 3: Student Baseline Implementation & Training
**Status**: ✅ LOCAL COMPLETE - Ready for Colab Training
**Environment**: LOCAL (code) → COLAB (training) → LOCAL (eval)
**Completed**: 2025-11-22 (local implementation)

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
   - Run `scripts/evaluate.py` on student baseline
   - Create `scripts/compare_student_baseline.py`:
     - Compare teacher vs student baseline metrics
     - Analyze performance gap (expected 5-10% accuracy drop)
     - Compare model sizes (parameters, disk size)
     - Compare inference speed (if possible on CPU)
     - Save plots to `scripts/results/student_baseline/`

#### Deliverables
- [x] `models/student.py` implemented (25 tests passing)
- [x] `train_student_baseline.py` implemented (28 tests passing)
- [x] `scripts/evaluate.py` updated with --model-type parameter (13 tests passing)
- [x] `docs/colab_setup.md` updated with Part 8: Student Baseline Training
- [x] Code pushed to GitHub (ready for Colab)
- [ ] Student baseline trained in Colab (USER ACTION REQUIRED)
- [ ] Student baseline checkpoint saved to Drive (awaiting training)
- [ ] Student baseline evaluated locally (awaiting checkpoint download)
- [x] `tests/test_evaluate.py` implemented
- [ ] `scripts/compare_baseline.py` completed (will create after evaluation)
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
     - **IMPORTANT - Course Requirement**: Implement KL divergence **from scratch** (NOT using `torch.nn.functional.kl_div` directly as a black box)
       - Manually compute temperature-scaled softmax: `p = softmax(logits / T)`
       - Manually compute KL divergence: `KL(p_teacher || p_student) = sum(p_teacher * log(p_teacher / p_student))`
       - Scale by T² (explain why in docstring: gradient magnitude correction)
       - Document the full mathematical derivation in comments
       - Show each computational step explicitly to demonstrate understanding of the underlying math
     - Combined loss function: `L = α * L_distill + (1-α) * L_hard`
     - Support for temperature (T) and alpha (α) from config
     - Logging for both loss components separately
     - Include comprehensive docstrings explaining the mathematics
     - **Rationale**: Addresses course feedback requiring lower-level numerical implementation (not just plug-and-play library calls)

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
   - Run `scripts/evaluate.py` on distilled student
   - Create `scripts/compare_distillation.py`:
     - Three-way comparison: Teacher | Student Baseline | Distilled Student
     - Performance improvement from distillation
     - Analyze gap closure (target: 50-70% of teacher-student gap)
     - Save plots to `scripts/results/distillation/`

#### Deliverables
- [ ] `models/distillation.py` implemented
- [ ] `train_student_kd.py` implemented
- [ ] Code pushed to GitHub
- [ ] Distilled student trained in Colab (T=4.0, α=0.7)
- [ ] Distilled student checkpoint saved to Drive
- [ ] Distilled student evaluated locally
- [ ] `scripts/compare_distillation.py` completed with plots saved
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
   - Run `scripts/evaluate.py` on all models
   - Create `scripts/analyze_ablations.py`:
     - Temperature vs accuracy curve
     - Alpha vs accuracy curve
     - **Temperature effect visualization**: Demonstrate how different T values "soften" probability distributions
       - Plot softmax outputs for same logits with T ∈ {1, 2, 4, 8, 16}
       - Show how higher T creates more uniform distributions (knowledge transfer mechanism)
       - Helps explain WHY temperature scaling works (addresses course feedback on understanding fundamentals)
     - Loss component analysis (distillation vs hard label)
     - Identify optimal hyperparameters
     - Statistical analysis if multiple runs available
     - Save plots to `scripts/results/ablations/`

5. **Optional: Architecture Variants** (LOCAL + COLAB):
   - Implement alternative student architectures (e.g., MobileNetV3-Large, EfficientNet-B0)
   - Train with optimal hyperparameters
   - Compare architectures

#### Deliverables
- [ ] Config variants created (9 total: 5 temperature + 4 alpha)
- [ ] All ablation models trained in Colab (9 training runs)
- [ ] All checkpoints downloaded and evaluated
- [ ] `scripts/analyze_ablations.py` completed with plots saved
- [ ] **PAPER ARTIFACT**: Temperature sensitivity curve (accuracy vs T)
- [ ] **PAPER ARTIFACT**: Temperature effect visualization (probability distribution softening)
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
1. **Comprehensive Evaluation Script** (LOCAL):
   - Create `scripts/comprehensive_analysis.py`:
     - Load all model checkpoints (teacher, student baseline, distilled student, ablation models)
     - Side-by-side predictions on same test samples
     - Per-class performance breakdown (bleached vs healthy)
     - Confidence distribution analysis
     - Calibration analysis (reliability diagrams)
     - Failure case analysis
     - Save plots to `scripts/results/comprehensive/`

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
- [ ] `scripts/comprehensive_analysis.py` completed with plots saved
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
   - Verify all analysis scripts run end-to-end without errors

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
- [ ] All scripts verified to run end-to-end
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
- **Active Phase**: Phase 3 (Student Baseline Implementation & Training)
- **Phase Status**: 🚀 COLAB TRAINING READY - Local implementation complete (281 tests passing)
- **Last Updated**: 2025-11-22

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
- ✅ Implemented `utils/env_utils.py` (26 tests passing)
- ✅ Implemented `utils/preprocessing.py` (32 tests passing, including 3 real coral image integration tests)
- ✅ Implemented `utils/data_loader.py` (31 tests passing)
- ✅ Implemented `utils/metrics.py` (45 tests passing, including 3 real wandb/model integration tests)
- ✅ Implemented `utils/visualization.py` (23 tests passing, including 4 real plotting integration tests)
- ✅ Created `tests/test_end_to_end_pipeline.py` (4 comprehensive pipeline tests)
- ✅ All unit tests passing (161 total tests: 128 unit + 33 integration)
- ✅ Local data pipeline verified with real coral images

### Completed Tasks (Phase 1)
- ✅ Implemented `models/teacher.py` (ResNet50-based teacher model)
  - Pretrained ImageNet weights support
  - Configurable architecture (num_classes, dropout)
  - Freeze/unfreeze backbone functionality
  - Parameter counting utilities
  - 25 tests passing (models/test_teacher.py)
- ✅ Implemented `train_teacher.py` (full training pipeline)
  - Config-driven training with CLI overrides
  - W&B integration (online/offline/disabled modes)
  - Checkpoint saving/loading with resume support
  - Learning rate scheduling (Cosine/Step)
  - Early stopping (patience = 10 epochs)
  - Comprehensive metrics logging
  - 25 tests passing (tests/test_train_teacher.py)
- ✅ Created `docs/colab_setup.md` (comprehensive Colab guide)
  - Step-by-step setup instructions
  - Drive mounting and verification
  - W&B authentication
  - Training commands with examples
  - Troubleshooting section
  - Checkpoint management guide
- ✅ Fixed critical bug in `train_teacher.py` (2025-11-16)
  - Corrected `build_dataloaders()` function call (wrong parameters)
  - Fixed: `split='train'` → `splits=['train', 'val']`
  - Fixed: Removed invalid `batch_size` parameter
  - Fixed: Proper dict unpacking of returned dataloaders
- ✅ Added 4 integration tests for `build_dataloaders()` usage (2025-11-16)
  - Test correct parameter passing and return type
  - Test batch size configuration
  - Test main() integration pattern
  - Total test count: 211 → 215 tests passing
- ✅ Completed teacher training in Colab (2025-11-16)
  - 19/50 epochs (early stopping after epoch 9)
  - Best validation accuracy: 83%
  - W&B run: lfidb03f
  - Checkpoints saved to Google Drive
  - Training time: ~2 hours on T4 GPU

### Completed Tasks (Phase 2)
- ✅ Restructured project from notebooks/ to scripts/ workflow (2025-11-18)
  - Deleted misleading `notebooks/` folder (referenced non-existent `analysis/` directory)
  - Updated all documentation (README.md, AGENTS.md, docs/colab_setup.md, .gitignore)
  - Created `scripts/results/` directory structure
- ✅ Downloaded teacher checkpoint from Google Drive (2025-11-18)
  - File: `checkpoints/teacher/best_model.pth` (~97 MB)
  - Epoch 8, 82.73% validation accuracy
- ✅ Implemented `scripts/evaluate.py` (2025-11-18)
  - Universal evaluation script for any model checkpoint
  - CLI interface with argparse
  - CPU-compatible inference with progress bar
  - Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)
  - Per-class performance breakdown
  - JSON output with full metadata
  - Fixed SSL certificate issue (pretrained=False for checkpoint loading)
  - Fixed metrics computation (correct function signatures and argument order)
- ✅ Evaluated teacher model on test set (2025-11-18)
  - Test accuracy: 77.70% (5% drop from validation - slight overfitting but reasonable)
  - Healthy: 80.0% precision, 76.7% recall, 78.3% F1 (73 samples)
  - Bleached: 75.4% precision, 78.8% recall, 77.0% F1 (66 samples)
  - Confusion matrix: 56/73 healthy correct, 52/66 bleached correct
  - Results saved to `scripts/results/teacher/test_results.json`
- ✅ Implemented `scripts/explore_data.py` (2025-11-21)
  - Dataset exploration and visualization script
  - Analyzes train/val/test split statistics (923 total images)
  - Computes image statistics (avg 295x222px, 27KB JPEG)
  - Generates visualizations: class distribution, sample grid, statistics summary
  - Outputs: `class_distribution.png`, `sample_images.png`, `dataset_stats.txt`
- ✅ Implemented `scripts/evaluate_teacher.py` (2025-11-21)
  - Teacher model analysis and visualization script
  - Loads evaluation results and attempts W&B fetch (graceful fallback)
  - Generates comprehensive analysis: confusion matrices, per-class metrics, error analysis
  - Outputs: 5 visualizations (confusion matrices raw/normalized, per-class metrics, error analysis, summary)
- ✅ Generated all paper artifacts for Phase 2 (2025-11-21)
  - Data Exploration: 3 files (class distribution, sample images, dataset stats)
  - Teacher Evaluation: 5 files (2 confusion matrices, per-class metrics, error analysis, summary)
  - Total: 8 publication-ready artifacts for paper Dataset and Results sections

### Completed Tasks (Phase 3 - Local Implementation)
- ✅ Implemented `models/student.py` (2025-11-22)
  - MobileNetV3-Small architecture (~1.52M parameters, 9x smaller than teacher)
  - API consistent with teacher model (freeze/unfreeze, param counting)
  - 25 tests passing (tests/test_student.py)
- ✅ Implemented `train_student_baseline.py` (2025-11-22)
  - Full training pipeline (W&B, checkpointing, early stopping, LR scheduling)
  - Identical structure to teacher for fair comparison
  - 28 tests passing (tests/test_train_student_baseline.py)
- ✅ Updated `scripts/evaluate.py` with model type support (2025-11-22)
  - Added --model-type parameter (teacher/student)
  - Unified evaluation infrastructure for all models
  - 13 tests passing (tests/test_evaluate.py)
- ✅ Updated `docs/colab_setup.md` (2025-11-22)
  - Added Part 8: Student Baseline Training section
  - Step-by-step Colab training instructions
  - Expected performance metrics and timings
- ✅ Full test suite verified (2025-11-22)
  - Total: 281 tests passing (66 new tests added for Phase 3)
  - Zero failures, 4 skipped (network-requiring tests)
  - Codebase ready for Colab training

### Completed Training Runs
- **Teacher Model** (2025-11-16, Colab T4 GPU)
  - Epochs: 19/50 (early stopping after 10 epochs without improvement)
  - Best validation accuracy: 83% (epoch 9)
  - Final training accuracy: 87%
  - W&B run: https://wandb.ai/rohitkatakam-northwestern-university/coral-bleaching/runs/lfidb03f
  - Checkpoint: Google Drive `coral-bleaching/checkpoints/teacher/best_model.pth`

### Available Checkpoints (in Google Drive)
- **teacher/best_model.pth** (2025-11-16, epoch 9)
  - Validation accuracy: 83%
  - Training accuracy: 78%
  - W&B run ID: lfidb03f
  - Notes: Early stopping triggered, best model saved before overfitting

### Available Checkpoints (downloaded locally)
- **checkpoints/teacher/best_model.pth** (2025-11-18, epoch 8)
  - Validation accuracy: 82.73%
  - Test accuracy: 77.70%
  - Model parameters: 23.5M
  - File size: ~97 MB
  - W&B run ID: lfidb03f
  - Notes: 5% generalization gap (val→test) indicates slight overfitting

### Current Blockers
None.

### Next Immediate Action
**Phase 3 LOCAL COMPLETE** ✅ - All implementation and testing done. Ready for Colab training.

**Phase 3 Local Accomplishments (2025-11-22):**
- ✅ Implemented student model (MobileNetV3-Small, ~1.52M params)
- ✅ Implemented student training script (identical pipeline to teacher)
- ✅ Updated evaluation infrastructure (supports teacher & student models)
- ✅ Comprehensive test coverage: 281 tests passing (66 new tests)
- ✅ Updated Colab documentation with student training guide
- ✅ Code pushed to GitHub and ready for Colab

**Next User Action - Train in Google Colab:**

1. **Clone repo in Colab** (use latest code from GitHub)
2. **Mount Google Drive** (ensure data is uploaded)
3. **Run student baseline training**:
   ```python
   !python train_student_baseline.py \
       --config configs/config.yaml \
       --output-dir /content/drive/MyDrive/coral-bleaching/checkpoints/student_baseline \
       --wandb-project coral-bleaching \
       --wandb-mode online \
       --device cuda
   ```
4. **Expected results**:
   - Training time: ~1-1.5 hours on T4 GPU
   - Val accuracy: ~72-75%
   - Checkpoint: `student_baseline/best_model.pth` (~10 MB)

**After Colab Training:**
1. Download checkpoint to local: `checkpoints/student_baseline/best_model.pth`
2. Run evaluation: `python3 scripts/evaluate.py --checkpoint checkpoints/student_baseline/best_model.pth --model-type student`
3. Create comparison script: `scripts/compare_baseline.py`
4. Generate comparison visualizations
5. Update AGENTS.md with training results

### Notes
- Project roadmap finalized with hybrid local/Colab workflow
- Workflow optimized for no-GPU local environment + Colab training
- Emphasis on simple, self-contained Colab notebook (no AI assistance needed)
- **Google Drive documentation added**: See Phase 0, Task 3 for upload instructions
- Data splits use relative paths for portability between local and Colab environments
- **Phase 0 Complete (2025-11-13)**: All utilities implemented with comprehensive test coverage (161 tests)
- **Phase 1 Local Complete (2025-11-14)**: Teacher model and training script implemented (50 new tests, 211 total)
  - Teacher model: ResNet50 with 23-25M parameters
  - Training script: Full pipeline with W&B, checkpointing, early stopping, LR scheduling
  - Colab documentation: Comprehensive step-by-step guide with troubleshooting
  - Test coverage: 25 tests for teacher model, 25 tests for training script
- **Phase 1 Colab Complete (2025-11-16)**: Teacher model trained successfully (4 new tests, 215 total)
  - Fixed critical dataloader bug in train_teacher.py (parameter mismatch)
  - Added integration tests to prevent regression
  - Completed teacher training: 83% val accuracy, early stopping at epoch 19
  - W&B tracking verified, checkpoints saved to Google Drive
  - Ready for Phase 2 evaluation
- **Phase 2 Restructuring (2025-11-18)**: Simplified workflow from notebooks/ to scripts/
  - Deleted `notebooks/` folder (referenced non-existent `analysis/` directory - misleading)
  - Moved to simpler `scripts/` workflow (all evaluation/analysis code in one place)
  - Rationale: notebooks/ referenced a complex `analysis/outputs/` structure that was never created
  - Result: Cleaner, more maintainable structure aligned with actual codebase
- **Phase 2 Evaluation (2025-11-18)**: Implemented and tested evaluation infrastructure
  - Created `scripts/evaluate.py` - universal evaluation tool (works, tested)
  - Fixed SSL cert issue: set pretrained=False when loading from checkpoint
  - Fixed metrics computation: correct function signatures and argument order
  - Evaluated teacher: 77.7% test accuracy (reasonable 5% drop from validation)
- **Phase 2 Complete (2025-11-21)**: Analysis scripts and paper artifacts generated
  - Implemented `scripts/explore_data.py` - dataset exploration with 3 artifacts
  - Implemented `scripts/evaluate_teacher.py` - teacher analysis with 5 artifacts
  - Fixed plot_sample_grid usage (convert labels to indices, remove invalid grid_size param)
  - Fixed plot_confusion_matrix usage (remove invalid title param)
  - Fixed config path for normalization (augmentations.normalization vs preprocessing.normalize)
  - W&B fetch gracefully skips if not logged in (training curves optional)
  - All 8 paper artifacts ready: dataset stats, confusion matrices, metrics, error analysis
  - Ready for Phase 3: Student Baseline Implementation
- **Course Feedback Integration (2025-11-21)**: Updated roadmap to address instructor feedback
  - Phase 4 now explicitly requires implementing KL divergence from scratch (not using torch.nn.functional.kl_div)
  - Added detailed mathematical documentation requirements for distillation loss
  - Phase 5 enhanced with temperature effect visualization (demonstrate softmax softening)
  - Ensures project demonstrates lower-level numerical understanding (not just high-level library usage)
  - Aligns with feedback: "build in numerical/lower-level components" and "explore how performance changes"
- **Phase 3 Local Complete (2025-11-22)**: Student baseline implementation ready for Colab training
  - Implemented student model: MobileNetV3-Small with ~1.52M parameters (9x compression)
  - Implemented training script: Full pipeline matching teacher for fair comparison
  - Updated evaluation infrastructure: Universal script supports teacher/student/distilled models
  - Updated Colab documentation: Part 8 with step-by-step student training instructions
  - Test coverage: 281 tests passing (66 new tests: 25 student model + 28 training + 13 evaluation)
  - Codebase verified: All tests pass, ready for GitHub push and Colab clone
  - **Status**: Awaiting Colab training (user action) - expected ~1-1.5 hours on T4 GPU
- Test quality: ~85% real testing (minimal mocking), includes integration tests with real data, real models, real wandb offline logging

---

## Quick Reference: Workflow Summary

**LOCAL (with Claude Code):**
1. Write/edit code (models, utils, scripts)
2. Write unit tests
3. Push to GitHub
4. Download checkpoints from Drive
5. Run evaluations (CPU-based) using `scripts/`
6. Generate paper figures using analysis scripts

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
