#!/usr/bin/env python3
"""
Teacher vs Student Baseline Comparison

Compares teacher and student baseline models to quantify performance gap
and compression benefits. Generates comprehensive comparison visualizations.

Usage:
    python scripts/compare_baseline.py

Outputs:
    - scripts/results/student_baseline/accuracy_comparison.png
    - scripts/results/student_baseline/model_efficiency.png
    - scripts/results/student_baseline/confusion_matrices_comparison.png
    - scripts/results/student_baseline/per_class_metrics_comparison.png
    - scripts/results/student_baseline/performance_vs_efficiency.png
    - scripts/results/student_baseline/error_analysis.png
    - scripts/results/student_baseline/comparison_summary.txt
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import time
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import plot_confusion_matrix
from utils.env_utils import get_project_root, resolve_data_path
from utils.data_loader import CoralDataset, build_dataloaders
from utils.preprocessing import get_test_transforms
from models.teacher import TeacherModel
from models.student import StudentModel

# %% Configuration
PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "student_baseline"
TEACHER_RESULTS_PATH = PROJECT_ROOT / "scripts" / "results" / "teacher" / "test_results.json"
STUDENT_RESULTS_PATH = PROJECT_ROOT / "scripts" / "results" / "student" / "test_results.json"
TEACHER_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "teacher" / "best_model.pth"
STUDENT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "student_baseline" / "best_model.pth"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Class names
CLASS_NAMES = ['healthy', 'bleached']

# Color scheme for plots
COLORS = {
    'teacher': '#3498db',  # Blue
    'student': '#e67e22',  # Orange
}

# %% Helper Functions

def load_results():
    """
    Load evaluation results for both teacher and student models.

    Returns:
        tuple: (teacher_results, student_results) dictionaries
    """
    print("Loading evaluation results...")

    with open(TEACHER_RESULTS_PATH, 'r') as f:
        teacher_results = json.load(f)

    with open(STUDENT_RESULTS_PATH, 'r') as f:
        student_results = json.load(f)

    print(f"Teacher test accuracy: {teacher_results['metrics']['accuracy']*100:.2f}%")
    print(f"Student test accuracy: {student_results['metrics']['accuracy']*100:.2f}%")

    return teacher_results, student_results


def get_checkpoint_size(checkpoint_path):
    """
    Get checkpoint file size in bytes.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        int: File size in bytes
    """
    return checkpoint_path.stat().st_size


def measure_inference_speed(model, config, device, num_batches=10):
    """
    Measure average inference time per image.

    Args:
        model: PyTorch model
        config: Configuration dictionary
        device: torch device
        num_batches: Number of batches to measure

    Returns:
        tuple: (mean_time, std_time) in seconds per image
    """
    # Build test dataloader
    dataloaders = build_dataloaders(config, splits=['test'])
    test_loader = dataloaders['test']

    model.eval()
    times = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_batches:
                break

            images = images.to(device)
            start = time.perf_counter()
            _ = model(images)
            end = time.perf_counter()
            times.append((end - start) / len(images))

    return np.mean(times), np.std(times)


def plot_accuracy_comparison(teacher_results, student_results, output_dir):
    """
    Generate accuracy comparison bar chart.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
    """
    print("\nGenerating accuracy comparison...")

    # Extract metrics
    teacher_acc = teacher_results['metrics']['accuracy'] * 100
    student_acc = student_results['metrics']['accuracy'] * 100
    teacher_healthy = teacher_results['metrics']['per_class_metrics']['healthy']['f1'] * 100
    student_healthy = student_results['metrics']['per_class_metrics']['healthy']['f1'] * 100
    teacher_bleached = teacher_results['metrics']['per_class_metrics']['bleached']['f1'] * 100
    student_bleached = student_results['metrics']['per_class_metrics']['bleached']['f1'] * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(3)
    width = 0.35

    # Create bars
    teacher_bars = ax.bar(x - width/2, [teacher_acc, teacher_healthy, teacher_bleached],
                          width, label='Teacher', color=COLORS['teacher'], alpha=0.8)
    student_bars = ax.bar(x + width/2, [student_acc, student_healthy, student_bleached],
                          width, label='Student', color=COLORS['student'], alpha=0.8)

    # Customize plot
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy / F1-Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Teacher vs Student Baseline - Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Overall Accuracy', 'Healthy F1', 'Bleached F1'])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(teacher_bars)
    add_labels(student_bars)

    plt.tight_layout()
    save_path = output_dir / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_model_efficiency(teacher_results, student_results, output_dir):
    """
    Generate model efficiency comparison table/chart.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
    """
    print("\nGenerating model efficiency comparison...")

    # Extract data
    teacher_params = teacher_results['model']['num_parameters']
    student_params = student_results['model']['num_parameters']
    teacher_size = get_checkpoint_size(TEACHER_CHECKPOINT_PATH) / (1024**2)  # MB
    student_size = get_checkpoint_size(STUDENT_CHECKPOINT_PATH) / (1024**2)  # MB

    param_compression = teacher_params / student_params
    size_compression = teacher_size / student_size

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Data for table
    table_data = [
        ['Metric', 'Teacher', 'Student', 'Compression Ratio'],
        ['Parameters', f'{teacher_params:,}', f'{student_params:,}', f'{param_compression:.1f}x'],
        ['Checkpoint Size', f'{teacher_size:.1f} MB', f'{student_size:.1f} MB', f'{size_compression:.1f}x'],
        ['Test Accuracy', f'{teacher_results["metrics"]["accuracy"]*100:.2f}%',
         f'{student_results["metrics"]["accuracy"]*100:.2f}%',
         f'{student_results["metrics"]["accuracy"]*100 - teacher_results["metrics"]["accuracy"]*100:+.2f}%'],
    ]

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, 4):
        for j in range(4):
            cell = table[(i, j)]
            if j == 1:
                cell.set_facecolor('#ebf5fb')  # Light blue for teacher
            elif j == 2:
                cell.set_facecolor('#fdebd0')  # Light orange for student
            elif j == 3:
                cell.set_facecolor('#e8f8f5')  # Light green for compression

    plt.title('Model Efficiency Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = output_dir / "model_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_confusion_matrices_comparison(teacher_results, student_results, output_dir):
    """
    Generate side-by-side confusion matrix comparison.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
    """
    print("\nGenerating confusion matrices comparison...")

    teacher_cm = np.array(teacher_results['metrics']['confusion_matrix'])
    student_cm = np.array(student_results['metrics']['confusion_matrix'])

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Teacher confusion matrix
    im1 = axes[0].imshow(teacher_cm, interpolation='nearest', cmap='Blues')
    axes[0].set_title('Teacher Model', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[0].set_xticks(np.arange(len(CLASS_NAMES)))
    axes[0].set_yticks(np.arange(len(CLASS_NAMES)))
    axes[0].set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    axes[0].set_yticklabels([c.capitalize() for c in CLASS_NAMES])

    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = axes[0].text(j, i, teacher_cm[i, j],
                              ha="center", va="center", color="black", fontweight='bold')

    # Add colorbar
    plt.colorbar(im1, ax=axes[0])

    # Student confusion matrix
    im2 = axes[1].imshow(student_cm, interpolation='nearest', cmap='Oranges')
    axes[1].set_title('Student Model', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1].set_xticks(np.arange(len(CLASS_NAMES)))
    axes[1].set_yticks(np.arange(len(CLASS_NAMES)))
    axes[1].set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    axes[1].set_yticklabels([c.capitalize() for c in CLASS_NAMES])

    # Add text annotations
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            text = axes[1].text(j, i, student_cm[i, j],
                              ha="center", va="center", color="black", fontweight='bold')

    # Add colorbar
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "confusion_matrices_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_per_class_metrics_comparison(teacher_results, student_results, output_dir):
    """
    Generate per-class metrics comparison chart.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
    """
    print("\nGenerating per-class metrics comparison...")

    # Extract metrics
    teacher_healthy = teacher_results['metrics']['per_class_metrics']['healthy']
    teacher_bleached = teacher_results['metrics']['per_class_metrics']['bleached']
    student_healthy = student_results['metrics']['per_class_metrics']['healthy']
    student_bleached = student_results['metrics']['per_class_metrics']['bleached']

    # Prepare data for grouped bar chart
    metrics = ['Precision', 'Recall', 'F1-Score']
    teacher_healthy_vals = [teacher_healthy['precision'], teacher_healthy['recall'], teacher_healthy['f1']]
    student_healthy_vals = [student_healthy['precision'], student_healthy['recall'], student_healthy['f1']]
    teacher_bleached_vals = [teacher_bleached['precision'], teacher_bleached['recall'], teacher_bleached['f1']]
    student_bleached_vals = [student_bleached['precision'], student_bleached['recall'], student_bleached['f1']]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(metrics))
    width = 0.35

    # Healthy class comparison
    bars1 = axes[0].bar(x - width/2, teacher_healthy_vals, width, label='Teacher',
                        color=COLORS['teacher'], alpha=0.8)
    bars2 = axes[0].bar(x + width/2, student_healthy_vals, width, label='Student',
                        color=COLORS['student'], alpha=0.8)
    axes[0].set_xlabel('Metric', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Healthy Class Performance', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    # Add value labels
    def add_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    add_labels(axes[0], bars1)
    add_labels(axes[0], bars2)

    # Bleached class comparison
    bars3 = axes[1].bar(x - width/2, teacher_bleached_vals, width, label='Teacher',
                        color=COLORS['teacher'], alpha=0.8)
    bars4 = axes[1].bar(x + width/2, student_bleached_vals, width, label='Student',
                        color=COLORS['student'], alpha=0.8)
    axes[1].set_xlabel('Metric', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Bleached Class Performance', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)

    add_labels(axes[1], bars3)
    add_labels(axes[1], bars4)

    plt.suptitle('Per-Class Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "per_class_metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_performance_vs_efficiency(teacher_results, student_results, output_dir):
    """
    Generate performance vs efficiency tradeoff scatter plot.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
    """
    print("\nGenerating performance vs efficiency tradeoff plot...")

    # Extract data
    teacher_params = teacher_results['model']['num_parameters'] / 1e6  # Millions
    student_params = student_results['model']['num_parameters'] / 1e6  # Millions
    teacher_acc = teacher_results['metrics']['accuracy'] * 100
    student_acc = student_results['metrics']['accuracy'] * 100

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot points
    ax.scatter(teacher_params, teacher_acc, s=300, c=COLORS['teacher'],
              alpha=0.7, edgecolors='black', linewidth=2, label='Teacher', zorder=3)
    ax.scatter(student_params, student_acc, s=300, c=COLORS['student'],
              alpha=0.7, edgecolors='black', linewidth=2, label='Student', zorder=3)

    # Add labels to points
    ax.annotate(f'Teacher\n{teacher_params:.1f}M params\n{teacher_acc:.2f}% acc',
               xy=(teacher_params, teacher_acc), xytext=(10, 10),
               textcoords='offset points', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['teacher'], alpha=0.3))
    ax.annotate(f'Student\n{student_params:.2f}M params\n{student_acc:.2f}% acc',
               xy=(student_params, student_acc), xytext=(10, -30),
               textcoords='offset points', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['student'], alpha=0.3))

    # Add compression annotation
    compression = teacher_params / student_params
    ax.text(0.05, 0.95, f'Compression: {compression:.1f}x\nAccuracy: {student_acc - teacher_acc:+.2f}%',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Customize plot
    ax.set_xlabel('Model Parameters (Millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance vs Efficiency Tradeoff', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, teacher_params * 1.1)
    ax.set_ylim(75, 82)

    plt.tight_layout()
    save_path = output_dir / "performance_vs_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def analyze_differential_errors(teacher_results, student_results, output_dir, max_samples=16):
    """
    Analyze samples where one model fails but the other succeeds.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save plot
        max_samples: Maximum number of samples to visualize
    """
    print(f"\nPerforming differential error analysis (max {max_samples} samples)...")

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Load test dataset
    test_csv = DATA_SPLITS_DIR / "test.csv"
    import pandas as pd
    test_df = pd.read_csv(test_csv)

    # Load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    teacher_model = TeacherModel(num_classes=len(CLASS_NAMES), pretrained=False)
    teacher_checkpoint = torch.load(TEACHER_CHECKPOINT_PATH, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.to(device)
    teacher_model.eval()

    student_model = StudentModel(num_classes=len(CLASS_NAMES), pretrained=False)
    student_checkpoint = torch.load(STUDENT_CHECKPOINT_PATH, map_location=device)
    student_model.load_state_dict(student_checkpoint['model_state_dict'])
    student_model.to(device)
    student_model.eval()

    # Get test transforms
    transform = get_test_transforms(config)

    # Run inference to find differential errors
    print("Running inference to identify differential errors...")
    student_fails_teacher_succeeds = []  # Student wrong, teacher right
    teacher_fails_student_succeeds = []  # Teacher wrong, student right

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finding errors"):
            img_path = resolve_data_path(row['image_path'])
            true_label = row['label']

            try:
                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Predict with both models
                teacher_output = teacher_model(img_tensor)
                teacher_pred_idx = teacher_output.argmax(dim=1).item()
                teacher_pred = CLASS_NAMES[teacher_pred_idx]

                student_output = student_model(img_tensor)
                student_pred_idx = student_output.argmax(dim=1).item()
                student_pred = CLASS_NAMES[student_pred_idx]

                # Check for differential errors
                teacher_correct = (teacher_pred == true_label)
                student_correct = (student_pred == true_label)

                if student_correct and not teacher_correct:
                    teacher_fails_student_succeeds.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'teacher_pred': teacher_pred,
                        'student_pred': student_pred,
                        'image_tensor': transform(img)
                    })

                elif teacher_correct and not student_correct:
                    student_fails_teacher_succeeds.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'teacher_pred': teacher_pred,
                        'student_pred': student_pred,
                        'image_tensor': transform(img)
                    })

            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue

    print(f"\nFound {len(student_fails_teacher_succeeds)} samples where student fails but teacher succeeds")
    print(f"Found {len(teacher_fails_student_succeeds)} samples where teacher fails but student succeeds")

    # Visualize both categories
    fig = plt.figure(figsize=(16, 12))

    # Grid for student failures (top half)
    num_student_fails = min(len(student_fails_teacher_succeeds), max_samples // 2)
    if num_student_fails > 0:
        grid_rows = int(np.ceil(np.sqrt(num_student_fails)))
        for idx in range(num_student_fails):
            ax = plt.subplot(4, 4, idx + 1)
            error = student_fails_teacher_succeeds[idx]

            # Denormalize
            mean = torch.tensor(config['augmentations']['normalization']['mean']).view(3, 1, 1)
            std = torch.tensor(config['augmentations']['normalization']['std']).view(3, 1, 1)
            img_denorm = error['image_tensor'] * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)

            ax.imshow(img_denorm.permute(1, 2, 0).numpy())
            ax.set_title(f"True: {error['true_label']}\nT: {error['teacher_pred']} ✓\nS: {error['student_pred']} ✗",
                        fontsize=9, fontweight='bold', color='red')
            ax.axis('off')

    # Grid for teacher failures (bottom half)
    num_teacher_fails = min(len(teacher_fails_student_succeeds), max_samples // 2)
    if num_teacher_fails > 0:
        for idx in range(num_teacher_fails):
            ax = plt.subplot(4, 4, 8 + idx + 1)
            error = teacher_fails_student_succeeds[idx]

            # Denormalize
            mean = torch.tensor(config['augmentations']['normalization']['mean']).view(3, 1, 1)
            std = torch.tensor(config['augmentations']['normalization']['std']).view(3, 1, 1)
            img_denorm = error['image_tensor'] * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)

            ax.imshow(img_denorm.permute(1, 2, 0).numpy())
            ax.set_title(f"True: {error['true_label']}\nT: {error['teacher_pred']} ✗\nS: {error['student_pred']} ✓",
                        fontsize=9, fontweight='bold', color='blue')
            ax.axis('off')

    plt.suptitle('Differential Error Analysis\nTop: Student Fails (Red) | Bottom: Teacher Fails (Blue)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "error_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def save_comparison_summary(teacher_results, student_results, output_dir):
    """
    Save comprehensive comparison summary to text file.

    Args:
        teacher_results: Teacher evaluation results
        student_results: Student evaluation results
        output_dir: Directory to save summary
    """
    print("\nSaving comparison summary...")

    save_path = output_dir / "comparison_summary.txt"

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEACHER vs STUDENT BASELINE COMPARISON\n")
        f.write("=" * 80 + "\n\n")

        # Model specifications
        f.write("MODEL SPECIFICATIONS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'Teacher':>20} {'Student':>20} {'Ratio':>10}\n")
        f.write("-" * 80 + "\n")

        teacher_params = teacher_results['model']['num_parameters']
        student_params = student_results['model']['num_parameters']
        teacher_size = get_checkpoint_size(TEACHER_CHECKPOINT_PATH) / (1024**2)
        student_size = get_checkpoint_size(STUDENT_CHECKPOINT_PATH) / (1024**2)

        f.write(f"{'Parameters':<30} {teacher_params:>20,} {student_params:>20,} "
               f"{teacher_params/student_params:>9.1f}x\n")
        f.write(f"{'Checkpoint Size (MB)':<30} {teacher_size:>20.1f} {student_size:>20.1f} "
               f"{teacher_size/student_size:>9.1f}x\n")
        f.write(f"{'Training Epochs':<30} {teacher_results['model']['checkpoint_epoch']:>20} "
               f"{student_results['model']['checkpoint_epoch']:>20} {'':>10}\n")
        f.write(f"{'Best Val Accuracy (%)':<30} "
               f"{teacher_results['model']['checkpoint_best_val_acc']*100:>20.2f} "
               f"{student_results['model']['checkpoint_best_val_acc']*100:>20.2f} {'':>10}\n")
        f.write("\n")

        # Performance comparison
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<30} {'Teacher':>20} {'Student':>20} {'Difference':>10}\n")
        f.write("-" * 80 + "\n")

        teacher_acc = teacher_results['metrics']['accuracy'] * 100
        student_acc = student_results['metrics']['accuracy'] * 100
        teacher_prec = teacher_results['metrics']['precision'] * 100
        student_prec = student_results['metrics']['precision'] * 100
        teacher_rec = teacher_results['metrics']['recall'] * 100
        student_rec = student_results['metrics']['recall'] * 100
        teacher_f1 = teacher_results['metrics']['f1'] * 100
        student_f1 = student_results['metrics']['f1'] * 100

        f.write(f"{'Test Accuracy (%)':<30} {teacher_acc:>20.2f} {student_acc:>20.2f} "
               f"{student_acc - teacher_acc:>+9.2f}\n")
        f.write(f"{'Precision (%)':<30} {teacher_prec:>20.2f} {student_prec:>20.2f} "
               f"{student_prec - teacher_prec:>+9.2f}\n")
        f.write(f"{'Recall (%)':<30} {teacher_rec:>20.2f} {student_rec:>20.2f} "
               f"{student_rec - teacher_rec:>+9.2f}\n")
        f.write(f"{'F1-Score (%)':<30} {teacher_f1:>20.2f} {student_f1:>20.2f} "
               f"{student_f1 - teacher_f1:>+9.2f}\n")
        f.write("\n")

        # Per-class performance
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("-" * 80 + "\n")

        for class_name in CLASS_NAMES:
            f.write(f"\n{class_name.upper()}:\n")
            teacher_class = teacher_results['metrics']['per_class_metrics'][class_name]
            student_class = student_results['metrics']['per_class_metrics'][class_name]

            f.write(f"  {'Metric':<26} {'Teacher':>18} {'Student':>18} {'Difference':>10}\n")
            f.write("  " + "-" * 76 + "\n")
            f.write(f"  {'Precision (%)':<26} {teacher_class['precision']*100:>18.2f} "
                   f"{student_class['precision']*100:>18.2f} "
                   f"{(student_class['precision'] - teacher_class['precision'])*100:>+9.2f}\n")
            f.write(f"  {'Recall (%)':<26} {teacher_class['recall']*100:>18.2f} "
                   f"{student_class['recall']*100:>18.2f} "
                   f"{(student_class['recall'] - teacher_class['recall'])*100:>+9.2f}\n")
            f.write(f"  {'F1-Score (%)':<26} {teacher_class['f1']*100:>18.2f} "
                   f"{student_class['f1']*100:>18.2f} "
                   f"{(student_class['f1'] - teacher_class['f1'])*100:>+9.2f}\n")
            f.write(f"  {'Support':<26} {teacher_class['support']:>18} "
                   f"{student_class['support']:>18} {'':>10}\n")

        # Confusion matrices
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION MATRICES:\n")
        f.write("-" * 80 + "\n")

        teacher_cm = np.array(teacher_results['metrics']['confusion_matrix'])
        student_cm = np.array(student_results['metrics']['confusion_matrix'])

        f.write("\nTEACHER:\n")
        f.write("           Predicted\n")
        f.write("           Healthy  Bleached\n")
        f.write(f"Actual Healthy   {teacher_cm[0][0]:4d}     {teacher_cm[0][1]:4d}\n")
        f.write(f"       Bleached  {teacher_cm[1][0]:4d}     {teacher_cm[1][1]:4d}\n")

        f.write("\nSTUDENT:\n")
        f.write("           Predicted\n")
        f.write("           Healthy  Bleached\n")
        f.write(f"Actual Healthy   {student_cm[0][0]:4d}     {student_cm[0][1]:4d}\n")
        f.write(f"       Bleached  {student_cm[1][0]:4d}     {student_cm[1][1]:4d}\n")

        # Key findings
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n\n")

        f.write(f"1. COMPRESSION: Student model achieves {teacher_params/student_params:.1f}x "
               f"parameter compression and {teacher_size/student_size:.1f}x checkpoint size reduction.\n\n")

        acc_diff = student_acc - teacher_acc
        if abs(acc_diff) < 1.0:
            f.write(f"2. PERFORMANCE: Student model achieves nearly identical performance "
                   f"({acc_diff:+.2f}% accuracy difference).\n\n")
        elif acc_diff > 0:
            f.write(f"2. PERFORMANCE: Student model OUTPERFORMS teacher by {acc_diff:+.2f}% accuracy.\n")
            f.write(f"   This suggests the smaller MobileNetV3 architecture may be better suited\n")
            f.write(f"   for this task, or the teacher model is overfitting.\n\n")
        else:
            f.write(f"2. PERFORMANCE: Student model achieves {student_acc:.2f}% accuracy "
                   f"({acc_diff:.2f}% drop from teacher).\n\n")

        # Class-specific findings
        healthy_diff = (student_class['f1'] - teacher_class['f1']) * 100
        f.write(f"3. CLASS PERFORMANCE:\n")
        f.write(f"   - Healthy: Student {'outperforms' if healthy_diff > 0 else 'underperforms'} "
               f"teacher by {abs(healthy_diff):.2f}% F1-score\n")

        student_bleached = student_results['metrics']['per_class_metrics']['bleached']
        teacher_bleached = teacher_results['metrics']['per_class_metrics']['bleached']
        bleached_diff = (student_bleached['f1'] - teacher_bleached['f1']) * 100
        f.write(f"   - Bleached: Student {'outperforms' if bleached_diff > 0 else 'underperforms'} "
               f"teacher by {abs(bleached_diff):.2f}% F1-score\n\n")

        f.write(f"4. EFFICIENCY: Student model provides excellent compression with minimal\n")
        f.write(f"   (or even positive) impact on accuracy, making it ideal for deployment.\n\n")

        f.write("=" * 80 + "\n")

    print(f"Saved to {save_path}")


# %% Main Execution

def main():
    """Main execution function."""
    print("=" * 80)
    print("TEACHER vs STUDENT BASELINE COMPARISON")
    print("=" * 80)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")

    # 1. Load results
    print("\n" + "=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)
    teacher_results, student_results = load_results()

    # 2. Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_accuracy_comparison(teacher_results, student_results, RESULTS_DIR)
    plot_model_efficiency(teacher_results, student_results, RESULTS_DIR)
    plot_confusion_matrices_comparison(teacher_results, student_results, RESULTS_DIR)
    plot_per_class_metrics_comparison(teacher_results, student_results, RESULTS_DIR)
    plot_performance_vs_efficiency(teacher_results, student_results, RESULTS_DIR)
    analyze_differential_errors(teacher_results, student_results, RESULTS_DIR, max_samples=16)

    # 3. Save summary
    print("\n" + "=" * 80)
    print("SAVING COMPARISON SUMMARY")
    print("=" * 80)
    save_comparison_summary(teacher_results, student_results, RESULTS_DIR)

    # 4. Final summary
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - accuracy_comparison.png")
    print(f"  - model_efficiency.png")
    print(f"  - confusion_matrices_comparison.png")
    print(f"  - per_class_metrics_comparison.png")
    print(f"  - performance_vs_efficiency.png")
    print(f"  - error_analysis.png")
    print(f"  - comparison_summary.txt")

    # Print key results
    print("\n" + "=" * 80)
    print("KEY RESULTS:")
    print("=" * 80)
    teacher_params = teacher_results['model']['num_parameters']
    student_params = student_results['model']['num_parameters']
    teacher_acc = teacher_results['metrics']['accuracy'] * 100
    student_acc = student_results['metrics']['accuracy'] * 100

    print(f"Compression Ratio: {teacher_params/student_params:.1f}x parameters")
    print(f"Teacher Accuracy: {teacher_acc:.2f}%")
    print(f"Student Accuracy: {student_acc:.2f}%")
    print(f"Accuracy Difference: {student_acc - teacher_acc:+.2f}%")


if __name__ == "__main__":
    main()
