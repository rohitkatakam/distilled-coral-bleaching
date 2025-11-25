# Paper Figures Guide for Coral Bleaching Knowledge Distillation Project

**Last Updated**: 2025-11-25
**Project Phase**: Phase 6 Complete (Paper-Ready)
**Report Type**: 4-6 page course project report

This guide provides recommendations for selecting and organizing figures for your paper.

---

## Quick Start: Recommended Figure Set

For a **4-6 page course report**, use these **5-7 figures**:

### MUST INCLUDE (5 figures):
1. **Figure 1**: Main results (2×2 multi-panel) - `paper_figures/main_results.png`
2. **Figure 2**: Dataset overview - `data_exploration/class_distribution.png`
3. **Figure 3**: Confusion matrices - `distillation/confusion_matrices_comparison.png`
4. **Figure 4**: KD effectiveness - `distillation/kd_effectiveness.png`
5. **Figure 5**: Confidence analysis - `paper_figures/confidence_by_correctness.png`

### OPTIONAL (add 1-2 if space allows):
6. **Figure 6**: Hyperparameter sensitivity - `distillation/hyperparameter_sensitivity.png`
7. **Figure 7**: Per-class metrics - `distillation/per_class_metrics_comparison.png`

---

## Complete Figure Inventory

### NEW FIGURES (Phase 6 - Generated Today)

#### 1. Main Results Figure (2×2 Multi-Panel)
- **File**: `paper_figures/main_results.png`
- **Size**: 14×11 inches, 150 DPI
- **Panels**:
  - (A) Test accuracy comparison (6 models: Teacher, Baseline, 4 KD variants)
  - (B) Model size comparison (23.5M vs 1.52M parameters)
  - (C) Accuracy vs efficiency scatter plot
  - (D) KD effectiveness (Δ from baseline)
- **Use in**: **Results section (primary figure)**
- **Key Finding**: Best KD (T=2.0, α=0.5) achieves 79.14% with 15.5× compression
- **Priority**: **CRITICAL - This is your Figure 1**

#### 2. Confidence Distribution Histograms
- **File**: `paper_figures/confidence_histograms.png`
- **Size**: 12×6 inches, 150 DPI
- **Content**: Overlapping histograms of prediction confidence for all 6 models
- **Use in**: Discussion section (confidence analysis)
- **Key Finding**: Models show different confidence distributions; baseline student most confident but not most accurate
- **Priority**: OPTIONAL (nice-to-have)

#### 3. Confidence by Correctness
- **File**: `paper_figures/confidence_by_correctness.png`
- **Size**: 14×6 inches, 150 DPI
- **Content**: Grouped bar chart showing avg confidence for correct vs incorrect predictions
- **Use in**: Discussion section (model calibration)
- **Key Finding**: Best KD (T=2.0, α=0.5) has highest confidence gap (0.153), suggesting better calibration
- **Priority**: **RECOMMENDED (demonstrates KD benefit beyond accuracy)**

---

### EXISTING FIGURES (Phases 2-4 - Already Generated)

#### Data Exploration (Phase 2)

##### 4. Class Distribution
- **File**: `data_exploration/class_distribution.png`
- **Content**: Bar chart showing class balance across train/val/test splits
- **Use in**: Methodology - Dataset section
- **Stats**: 923 total images, 47.5% healthy, 52.5% bleached
- **Priority**: **MUST INCLUDE (shows dataset balance)**

##### 5. Sample Images
- **File**: `data_exploration/sample_images.png`
- **Content**: Grid of representative coral images (bleached vs healthy)
- **Use in**: Introduction or Methodology (visual examples)
- **Priority**: OPTIONAL (nice visual but takes space)

##### 6. Dataset Statistics
- **File**: `data_exploration/dataset_stats.txt`
- **Content**: Text summary (total samples, split ratios, image resolution stats)
- **Use in**: Methodology section (as inline text, not figure)
- **Priority**: USE AS TEXT REFERENCE

---

#### Teacher Baseline (Phase 2)

##### 7. Teacher Confusion Matrix (Raw)
- **File**: `teacher/confusion_matrix.png`
- **Content**: Raw counts confusion matrix (77.70% test accuracy)
- **Use in**: Results section (if showing individual model performance)
- **Priority**: LOW (use distillation comparison instead)

##### 8. Teacher Confusion Matrix (Normalized)
- **File**: `teacher/confusion_matrix_normalized.png`
- **Content**: Percentage-based confusion matrix
- **Use in**: Results section (alternative to raw counts)
- **Priority**: LOW (use distillation comparison instead)

##### 9. Teacher Per-Class Metrics
- **File**: `teacher/per_class_metrics.png`
- **Content**: Bar chart of precision/recall/F1 by class
- **Use in**: Results section (detailed analysis)
- **Priority**: LOW (use distillation comparison instead)

##### 10. Teacher Error Analysis
- **File**: `teacher/error_analysis.png`
- **Content**: Breakdown of TP/TN/FP/FN counts
- **Use in**: Discussion section
- **Priority**: LOW

##### 11. Teacher Evaluation Summary
- **File**: `teacher/evaluation_summary.txt`
- **Content**: Text summary of test results
- **Use in**: Reference for writing results section
- **Priority**: USE AS TEXT REFERENCE

---

#### Student Baseline Comparison (Phase 3)

##### 12. Baseline Accuracy Comparison
- **File**: `student_baseline/accuracy_comparison.png`
- **Content**: Teacher (77.70%) vs Student Baseline (78.42%)
- **Use in**: Results section (motivates KD)
- **Priority**: LOW (included in main_results.png Panel A)

##### 13. Baseline Model Efficiency
- **File**: `student_baseline/model_efficiency.png`
- **Content**: Table showing parameters and checkpoint size (15.5× compression)
- **Use in**: Results or Discussion section
- **Priority**: LOW (included in main_results.png Panel B)

##### 14. Baseline Confusion Matrices Comparison
- **File**: `student_baseline/confusion_matrices_comparison.png`
- **Content**: Side-by-side confusion matrices (teacher vs baseline)
- **Use in**: Results section (decision pattern comparison)
- **Priority**: LOW (use 6-model comparison instead)

##### 15. Baseline Per-Class Metrics
- **File**: `student_baseline/per_class_metrics_comparison.png`
- **Content**: Precision/recall/F1 for teacher vs baseline
- **Use in**: Results section
- **Priority**: LOW (use 6-model comparison instead)

##### 16. Baseline Error Analysis
- **File**: `student_baseline/error_analysis.png`
- **Content**: Error pattern comparison
- **Use in**: Discussion section
- **Priority**: LOW

##### 17. Baseline Performance vs Efficiency
- **File**: `student_baseline/performance_vs_efficiency.png`
- **Content**: Scatter plot (2 points: teacher and baseline)
- **Use in**: Results section
- **Priority**: LOW (included in main_results.png Panel C)

##### 18. Baseline Comparison Summary
- **File**: `student_baseline/comparison_summary.txt`
- **Content**: Detailed text comparison
- **Use in**: Reference for writing
- **Priority**: USE AS TEXT REFERENCE

---

#### Distillation Results (Phase 4)

##### 19. Distillation Accuracy Comparison
- **File**: `distillation/accuracy_comparison.png`
- **Content**: Bar chart of all 6 models (77.70% → 79.14%)
- **Use in**: Results section
- **Priority**: **HIGH (but redundant with main_results.png Panel A)**
- **Note**: Use main_results.png instead for cleaner presentation

##### 20. Model Efficiency (6 models)
- **File**: `distillation/model_efficiency.png`
- **Content**: Table showing parameters for all 6 models
- **Use in**: Results or Discussion section
- **Priority**: MEDIUM (redundant with main_results.png Panel B)

##### 21. Confusion Matrices Comparison (6 models)
- **File**: `distillation/confusion_matrices_comparison.png`
- **Content**: 2×3 grid of confusion matrices (all 6 models)
- **Use in**: Results section (detailed model behavior)
- **Priority**: **HIGH - Shows error patterns across all models**
- **Recommendation**: Use as Figure 3 in paper

##### 22. Per-Class Metrics Comparison (6 models)
- **File**: `distillation/per_class_metrics_comparison.png`
- **Content**: Precision/recall/F1 for healthy and bleached classes (all 6 models)
- **Use in**: Results section (class-level analysis)
- **Priority**: MEDIUM (good for discussion of class imbalances)

##### 23. Performance vs Efficiency (6 models)
- **File**: `distillation/performance_vs_efficiency.png`
- **Content**: Scatter plot showing Pareto frontier
- **Use in**: Discussion section
- **Priority**: LOW (redundant with main_results.png Panel C)

##### 24. Hyperparameter Sensitivity
- **File**: `distillation/hyperparameter_sensitivity.png`
- **Content**: 2D visualization of temperature (T) and alpha (α) effects
- **Use in**: Results or Discussion section (KD hyperparameter analysis)
- **Priority**: **MEDIUM-HIGH - Demonstrates systematic exploration**
- **Recommendation**: Use as Figure 6 if space allows

##### 25. KD Effectiveness
- **File**: `distillation/kd_effectiveness.png`
- **Content**: Bar chart showing Δ accuracy from baseline for each KD config
- **Use in**: Results section (directly answers "does KD work?")
- **Priority**: **HIGH - Clear demonstration of KD impact**
- **Recommendation**: Use as Figure 4 in paper

##### 26. Error Analysis (6 models)
- **File**: `distillation/error_analysis.png`
- **Content**: Error type breakdown (TP/TN/FP/FN)
- **Use in**: Discussion section
- **Priority**: LOW

##### 27. Distillation Comparison Summary
- **File**: `distillation/comparison_summary.txt`
- **Content**: Comprehensive text summary with all metrics
- **Use in**: **PRIMARY REFERENCE for Results section**
- **Priority**: **CRITICAL - Use this for your results table**

---

## Recommended Figure Organization by Paper Section

### 1. Introduction (0.5 pages)
- **No figures** (text only)
- Optional: Reference sample_images.png if you want to show coral bleaching examples

### 2. Related Work (0.5 pages)
- **No figures** (text only)

### 3. Methodology - Dataset (0.5 pages)
- **Figure 1**: Class distribution (`data_exploration/class_distribution.png`)
- **Text reference**: Dataset stats from `data_exploration/dataset_stats.txt`
  - 923 images: 438 healthy (47.5%), 485 bleached (52.5%)
  - Split: 645 train, 139 val, 139 test
  - Average resolution: 295×222 pixels

### 4. Methodology - Models & Training (0.5 pages)
- **No figures** (describe architectures in text)
- Mention: Teacher (ResNet50, 23.5M params), Student (MobileNetV3-Small, 1.52M params)

### 5. Results (2 pages)

**Must Include**:
- **Figure 2**: Main results multi-panel (`paper_figures/main_results.png`)
  - This is your centerpiece figure - tells the complete story in one page
  - Panel A: Accuracy comparison
  - Panel B: Model size
  - Panel C: Efficiency tradeoff
  - Panel D: KD effectiveness

- **Figure 3**: Confusion matrices comparison (`distillation/confusion_matrices_comparison.png`)
  - Shows error patterns across all 6 models
  - Demonstrates where each model succeeds/fails

- **Figure 4**: KD effectiveness (`distillation/kd_effectiveness.png`)
  - Direct answer: "Did KD work?" → Yes, +0.72% improvement
  - Shows variability across hyperparameter configs

**Optional (if space)**:
- **Figure 5**: Hyperparameter sensitivity (`distillation/hyperparameter_sensitivity.png`)
  - Demonstrates systematic exploration of T and α
  - Explains why T=2.0, α=0.5 is optimal

**Text reference**:
- Use `distillation/comparison_summary.txt` to create your results table

### 6. Discussion (1 page)

**Recommended**:
- **Figure 5** (or 6): Confidence by correctness (`paper_figures/confidence_by_correctness.png`)
  - Shows KD improves prediction calibration, not just accuracy
  - Best KD has highest confidence gap (0.153)

**Optional**:
- **Figure 6** (or 7): Per-class metrics (`distillation/per_class_metrics_comparison.png`)
  - Discuss class-level performance differences

**Text discussion**:
- Why did baseline student outperform teacher? (overparameterization)
- Why did T=2.0, α=0.5 work best? (sharper distributions, balanced weighting)
- Limitations: small dataset, limited model architectures explored
- Future work: more aggressive compression, other distillation techniques

### 7. Conclusion (0.5 pages)
- **No figures** (text summary)
- Reference key finding: 79.14% accuracy with 15.5× compression

---

## Figure Quality Notes

### Already Publication-Ready
- All PNG files are 150 DPI (sufficient for course reports)
- All plots have clear labels, legends, and titles
- Color scheme is consistent across related plots

### If You Need Higher Resolution
- For conference submission (not needed for course report), regenerate with:
  - DPI=300 or 400
  - Larger font sizes (add +2 to all fontsize parameters)
  - Save as PDF instead of PNG for vector graphics

---

## File Paths Summary

All figures are in: `/Users/rohitkatakam/projects/distilled-coral-bleaching/scripts/results/`

**New figures** (Phase 6):
```
paper_figures/
├── main_results.png                   # Figure 1 (2×2 multi-panel)
├── confidence_histograms.png          # Optional
└── confidence_by_correctness.png      # Figure 5
```

**Existing figures** (Phases 2-4):
```
data_exploration/
├── class_distribution.png             # Figure 2 (dataset)
├── sample_images.png                  # Optional
└── dataset_stats.txt                  # Text reference

distillation/
├── confusion_matrices_comparison.png  # Figure 3 (6 models)
├── kd_effectiveness.png               # Figure 4 (KD impact)
├── hyperparameter_sensitivity.png     # Figure 6 (optional)
├── per_class_metrics_comparison.png   # Figure 7 (optional)
└── comparison_summary.txt             # Text reference (CRITICAL)
```

---

## Checklist for Paper Writing

- [ ] Read `distillation/comparison_summary.txt` - this is your results table source
- [ ] Use `main_results.png` as Figure 1 (centerpiece)
- [ ] Include `class_distribution.png` in Methodology
- [ ] Include `confusion_matrices_comparison.png` in Results
- [ ] Include `kd_effectiveness.png` in Results
- [ ] Include `confidence_by_correctness.png` in Discussion
- [ ] Optional: Add `hyperparameter_sensitivity.png` if space allows
- [ ] Reference `dataset_stats.txt` for dataset description
- [ ] Cite key findings: 79.14% accuracy, 15.5× compression, +0.72% improvement

---

## Contact & Notes

**Project Status**: Ready for paper writing!

**Total Artifacts**: 27 files (3 new + 24 existing)
- 21 PNG plots
- 3 TXT summaries (use as text references)
- 3 JSON metadata files (backup data)

**Recommended Total Figures in Paper**: 5-7 (depending on page limit)

**Next Steps**: Write paper independently using this guide

---

## Quick Reference: Top 7 Figures

| # | Figure | File | Section | Priority |
|---|--------|------|---------|----------|
| 1 | Main results (2×2) | `paper_figures/main_results.png` | Results | **CRITICAL** |
| 2 | Class distribution | `data_exploration/class_distribution.png` | Methodology | **MUST** |
| 3 | Confusion matrices | `distillation/confusion_matrices_comparison.png` | Results | **HIGH** |
| 4 | KD effectiveness | `distillation/kd_effectiveness.png` | Results | **HIGH** |
| 5 | Confidence by correctness | `paper_figures/confidence_by_correctness.png` | Discussion | **RECOMMENDED** |
| 6 | Hyperparameter sensitivity | `distillation/hyperparameter_sensitivity.png` | Results/Discussion | MEDIUM |
| 7 | Per-class metrics | `distillation/per_class_metrics_comparison.png` | Discussion | MEDIUM |

Use Figures 1-5 for **5-page report**, add 6-7 if you have space for **6+ pages**.

---

**Good luck with paper writing!** 🎓
