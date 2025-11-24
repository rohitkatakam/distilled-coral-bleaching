#!/usr/bin/env python3
"""
Knowledge Distillation Comparison Analysis

Compares teacher, student baseline, and 4 distilled student models to evaluate
the effectiveness of knowledge distillation with different hyperparameters.

Usage:
    python scripts/compare_distillation.py

Outputs:
    - scripts/results/distillation/accuracy_comparison.png
    - scripts/results/distillation/model_efficiency.png
    - scripts/results/distillation/confusion_matrices_comparison.png
    - scripts/results/distillation/per_class_metrics_comparison.png
    - scripts/results/distillation/performance_vs_efficiency.png
    - scripts/results/distillation/hyperparameter_sensitivity.png
    - scripts/results/distillation/kd_effectiveness.png
    - scripts/results/distillation/error_analysis.png
    - scripts/results/distillation/comparison_summary.txt
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
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "distillation"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Model results paths
RESULTS_PATHS = {
    'Teacher': PROJECT_ROOT / "scripts" / "results" / "teacher" / "test_results.json",
    'Student Baseline': PROJECT_ROOT / "scripts" / "results" / "student" / "test_results.json",
    'KD (T=2.0, α=0.5)': PROJECT_ROOT / "scripts" / "results" / "kd_t2.0_a0.5" / "student" / "test_results.json",
    'KD (T=4.0, α=0.3)': PROJECT_ROOT / "scripts" / "results" / "kd_t4.0_a0.3" / "student" / "test_results.json",
    'KD (T=4.0, α=0.7)': PROJECT_ROOT / "scripts" / "results" / "kd_t4.0_a0.7" / "student" / "test_results.json",
    'KD (T=8.0, α=0.9)': PROJECT_ROOT / "scripts" / "results" / "kd_t8.0_a0.9" / "student" / "test_results.json",
}

# Checkpoint paths (for error analysis)
CHECKPOINT_PATHS = {
    'Teacher': PROJECT_ROOT / "checkpoints" / "teacher" / "best_model.pth",
    'Student Baseline': PROJECT_ROOT / "checkpoints" / "student_baseline" / "best_model.pth",
    'KD (T=2.0, α=0.5)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t2.0_a0.5.pth",
    'KD (T=4.0, α=0.3)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t4.0_a0.3.pth",
    'KD (T=4.0, α=0.7)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t4.0_a0.7.pth",
    'KD (T=8.0, α=0.9)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t8.0_a0.9.pth",
}

# Class names
CLASS_NAMES = ['healthy', 'bleached']

# Color scheme for plots
COLORS = {
    'Teacher': '#3498db',           # Blue
    'Student Baseline': '#e67e22',  # Orange
    'KD (T=2.0, α=0.5)': '#27ae60', # Green
    'KD (T=4.0, α=0.3)': '#9b59b6', # Purple
    'KD (T=4.0, α=0.7)': '#e74c3c', # Red
    'KD (T=8.0, α=0.9)': '#f39c12', # Yellow
}

# %% Helper Functions

def load_all_results():
    """
    Load evaluation results for all models.

    Returns:
        dict: Dictionary mapping model names to results dictionaries
    """
    print("Loading evaluation results...")

    all_results = {}
    for model_name, results_path in RESULTS_PATHS.items():
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")

        with open(results_path, 'r') as f:
            all_results[model_name] = json.load(f)

        acc = all_results[model_name]['metrics']['accuracy'] * 100
        print(f"  {model_name}: {acc:.2f}% test accuracy")

    return all_results


def extract_hyperparameters(model_name):
    """
    Extract temperature (T) and alpha (α) from KD model name.

    Args:
        model_name: Model name string

    Returns:
        tuple: (T, alpha) or (None, None) if not a KD model
    """
    if not model_name.startswith('KD'):
        return None, None

    # Parse "KD (T=X.X, α=Y.Y)"
    import re
    match = re.search(r'T=(\d+\.\d+).*α=(\d+\.\d+)', model_name)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def identify_best_kd_model(all_results):
    """
    Identify the best performing KD configuration.

    Args:
        all_results: Dictionary of all results

    Returns:
        str: Name of best KD model
    """
    kd_models = {k: v for k, v in all_results.items() if k.startswith('KD')}
    best_model = max(kd_models.items(), key=lambda x: x[1]['metrics']['accuracy'])
    return best_model[0]


def get_checkpoint_size(checkpoint_path):
    """
    Get checkpoint file size in bytes.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        int: File size in bytes
    """
    return checkpoint_path.stat().st_size


# %% Visualization Functions

def plot_accuracy_comparison(all_results, output_dir):
    """
    Generate accuracy comparison bar chart for all 6 models.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating accuracy comparison...")

    # Prepare data
    model_names = list(all_results.keys())
    accuracies = [all_results[m]['metrics']['accuracy'] * 100 for m in model_names]
    colors_list = [COLORS[m] for m in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(model_names))
    bars = ax.bar(x, accuracies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Knowledge Distillation Comparison - Test Accuracy', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11)
    ax.set_ylim(70, 82)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add horizontal reference lines
    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'Baseline: {baseline_acc:.2f}%')
    ax.legend(fontsize=11, loc='lower right')

    plt.tight_layout()
    save_path = output_dir / "accuracy_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_model_efficiency(all_results, output_dir):
    """
    Generate model efficiency comparison table.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating model efficiency comparison...")

    # Prepare data
    model_names = list(all_results.keys())

    # Build table data
    table_data = [['Model', 'Parameters', 'Size (MB)', 'Test Acc', 'Δ Teacher', 'Δ Baseline']]

    teacher_acc = all_results['Teacher']['metrics']['accuracy'] * 100
    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100

    for model_name in model_names:
        results = all_results[model_name]
        params = results['model']['num_parameters']

        # Get checkpoint size
        if model_name in CHECKPOINT_PATHS:
            size_mb = get_checkpoint_size(CHECKPOINT_PATHS[model_name]) / (1024**2)
        else:
            size_mb = 0

        acc = results['metrics']['accuracy'] * 100
        delta_teacher = acc - teacher_acc
        delta_baseline = acc - baseline_acc

        table_data.append([
            model_name,
            f'{params:,}',
            f'{size_mb:.1f}',
            f'{acc:.2f}%',
            f'{delta_teacher:+.2f}%',
            f'{delta_baseline:+.2f}%'
        ])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.20, 0.15, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    for i in range(1, len(table_data)):
        for j in range(6):
            cell = table[(i, j)]
            # Alternate row colors
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            # Highlight best accuracy in green
            if j == 3:  # Test Acc column
                acc_val = float(table_data[i][3].strip('%'))
                if acc_val == max(float(row[3].strip('%')) for row in table_data[1:]):
                    cell.set_facecolor('#abebc6')
                    cell.set_text_props(weight='bold')

    plt.title('Model Efficiency and Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = output_dir / "model_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_confusion_matrices_comparison(all_results, output_dir):
    """
    Generate 2×3 grid of confusion matrices for all models.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating confusion matrices comparison...")

    model_names = list(all_results.keys())

    # Create 2×3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        cm = np.array(all_results[model_name]['metrics']['confusion_matrix'])
        acc = all_results[model_name]['metrics']['accuracy'] * 100

        ax = axes[idx]

        # Choose colormap based on model type
        if model_name == 'Teacher':
            cmap = 'Blues'
        elif model_name == 'Student Baseline':
            cmap = 'Oranges'
        else:
            cmap = 'Greens'

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(f'{model_name}\nAcc: {acc:.2f}%', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(len(CLASS_NAMES)))
        ax.set_yticks(np.arange(len(CLASS_NAMES)))
        ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
        ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES])

        # Add text annotations
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                text = ax.text(j, i, cm[i, j],
                              ha="center", va="center", color="black", fontweight='bold', fontsize=11)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('Confusion Matrix Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "confusion_matrices_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_per_class_metrics_comparison(all_results, output_dir):
    """
    Generate per-class metrics comparison with grouped bars.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating per-class metrics comparison...")

    model_names = list(all_results.keys())

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    metrics = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.13  # Bar width for 6 models

    for class_idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[class_idx]

        # Plot grouped bars for each model
        for model_idx, model_name in enumerate(model_names):
            class_metrics = all_results[model_name]['metrics']['per_class_metrics'][class_name]
            values = [class_metrics['precision'], class_metrics['recall'], class_metrics['f1']]

            offset = (model_idx - 2.5) * width  # Center the groups
            bars = ax.bar(x + offset, values, width, label=model_name,
                         color=COLORS[model_name], alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{class_name.capitalize()} Class Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Per-Class Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "per_class_metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_performance_vs_efficiency(all_results, output_dir):
    """
    Generate performance vs efficiency tradeoff scatter plot.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating performance vs efficiency tradeoff plot...")

    model_names = list(all_results.keys())

    # Extract data
    params_list = []
    acc_list = []
    colors_list = []
    labels_list = []

    for model_name in model_names:
        params = all_results[model_name]['model']['num_parameters'] / 1e6  # Millions
        acc = all_results[model_name]['metrics']['accuracy'] * 100
        params_list.append(params)
        acc_list.append(acc)
        colors_list.append(COLORS[model_name])
        labels_list.append(model_name)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot points
    for params, acc, color, label in zip(params_list, acc_list, colors_list, labels_list):
        ax.scatter(params, acc, s=300, c=color, alpha=0.7, edgecolors='black',
                  linewidth=2, label=label, zorder=3)

    # Add labels to points
    for params, acc, label in zip(params_list, acc_list, labels_list):
        ax.annotate(f'{label}\n{params:.2f}M\n{acc:.2f}%',
                   xy=(params, acc), xytext=(10, 10),
                   textcoords='offset points', fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Customize plot
    ax.set_xlabel('Model Parameters (Millions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Performance vs Efficiency Tradeoff', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(params_list) * 1.1)
    ax.set_ylim(75, 82)

    plt.tight_layout()
    save_path = output_dir / "performance_vs_efficiency.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_hyperparameter_sensitivity(all_results, output_dir):
    """
    Generate hyperparameter sensitivity analysis plot.
    Shows how accuracy varies with temperature (T) and alpha (α).

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating hyperparameter sensitivity analysis...")

    # Extract KD model data
    kd_data = []
    for model_name, results in all_results.items():
        if model_name.startswith('KD'):
            T, alpha = extract_hyperparameters(model_name)
            acc = results['metrics']['accuracy'] * 100
            kd_data.append({'T': T, 'alpha': alpha, 'acc': acc, 'name': model_name})

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Accuracy vs Temperature (grouped by alpha)
    ax = axes[0]
    alpha_groups = {}
    for d in kd_data:
        if d['alpha'] not in alpha_groups:
            alpha_groups[d['alpha']] = []
        alpha_groups[d['alpha']].append((d['T'], d['acc']))

    for alpha, points in alpha_groups.items():
        points = sorted(points)  # Sort by T
        T_vals, acc_vals = zip(*points)
        ax.plot(T_vals, acc_vals, 'o-', linewidth=2, markersize=10,
               label=f'α={alpha}', alpha=0.8)

    ax.set_xlabel('Temperature (T)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Temperature Sensitivity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(75, 82)

    # Subplot 2: Accuracy vs Alpha (grouped by T)
    ax = axes[1]
    T_groups = {}
    for d in kd_data:
        if d['T'] not in T_groups:
            T_groups[d['T']] = []
        T_groups[d['T']].append((d['alpha'], d['acc']))

    for T, points in T_groups.items():
        points = sorted(points)  # Sort by alpha
        alpha_vals, acc_vals = zip(*points)
        ax.plot(alpha_vals, acc_vals, 's-', linewidth=2, markersize=10,
               label=f'T={T}', alpha=0.8)

    ax.set_xlabel('Alpha (α)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Alpha Sensitivity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(75, 82)

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "hyperparameter_sensitivity.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def plot_kd_effectiveness(all_results, output_dir):
    """
    Generate KD effectiveness plot showing Δ accuracy from baseline.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
    """
    print("\nGenerating KD effectiveness plot...")

    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100

    # Extract KD model deltas
    kd_models = [k for k in all_results.keys() if k.startswith('KD')]
    deltas = []
    for model_name in kd_models:
        acc = all_results[model_name]['metrics']['accuracy'] * 100
        delta = acc - baseline_acc
        deltas.append((model_name, delta))

    # Sort by delta (best first)
    deltas.sort(key=lambda x: x[1], reverse=True)
    model_names, delta_values = zip(*deltas)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    colors_bars = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in delta_values]
    x = np.arange(len(model_names))
    bars = ax.bar(x, delta_values, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_xlabel('KD Configuration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Δ Accuracy from Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('Knowledge Distillation Effectiveness', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, delta in zip(bars, delta_values):
        height = bar.get_height()
        y_pos = height + 0.1 if height > 0 else height - 0.3
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{delta:+.2f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = output_dir / "kd_effectiveness.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def analyze_differential_errors(all_results, output_dir, max_samples=8):
    """
    Analyze samples where baseline fails but best KD succeeds (and vice versa).

    Note: Simplified version - just compares baseline vs best KD model.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save plot
        max_samples: Maximum number of samples to visualize
    """
    print(f"\nPerforming differential error analysis (simplified: baseline vs best KD)...")

    # Identify best KD model
    best_kd = identify_best_kd_model(all_results)
    print(f"Best KD model: {best_kd}")

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

    baseline_model = StudentModel(num_classes=len(CLASS_NAMES), pretrained=False)
    baseline_checkpoint = torch.load(CHECKPOINT_PATHS['Student Baseline'],
                                    map_location=device, weights_only=False)
    baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
    baseline_model.to(device)
    baseline_model.eval()

    kd_model = StudentModel(num_classes=len(CLASS_NAMES), pretrained=False)
    kd_checkpoint = torch.load(CHECKPOINT_PATHS[best_kd],
                              map_location=device, weights_only=False)
    kd_model.load_state_dict(kd_checkpoint['model_state_dict'])
    kd_model.to(device)
    kd_model.eval()

    # Get test transforms
    transform = get_test_transforms(config)

    # Run inference to find differential errors
    print("Running inference to identify differential errors...")
    baseline_fails_kd_succeeds = []
    kd_fails_baseline_succeeds = []

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finding errors"):
            img_path = resolve_data_path(row['image_path'])
            true_label = row['label']

            try:
                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Predict with both models
                baseline_output = baseline_model(img_tensor)
                baseline_pred_idx = baseline_output.argmax(dim=1).item()
                baseline_pred = CLASS_NAMES[baseline_pred_idx]

                kd_output = kd_model(img_tensor)
                kd_pred_idx = kd_output.argmax(dim=1).item()
                kd_pred = CLASS_NAMES[kd_pred_idx]

                # Check for differential errors
                baseline_correct = (baseline_pred == true_label)
                kd_correct = (kd_pred == true_label)

                if kd_correct and not baseline_correct:
                    baseline_fails_kd_succeeds.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'baseline_pred': baseline_pred,
                        'kd_pred': kd_pred,
                        'image_tensor': transform(img)
                    })

                elif baseline_correct and not kd_correct:
                    kd_fails_baseline_succeeds.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'baseline_pred': baseline_pred,
                        'kd_pred': kd_pred,
                        'image_tensor': transform(img)
                    })

            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue

    print(f"\nFound {len(baseline_fails_kd_succeeds)} samples where baseline fails but KD succeeds")
    print(f"Found {len(kd_fails_baseline_succeeds)} samples where KD fails but baseline succeeds")

    # Visualize
    fig = plt.figure(figsize=(16, 10))

    # Grid for baseline failures (top half)
    num_baseline_fails = min(len(baseline_fails_kd_succeeds), max_samples // 2)
    if num_baseline_fails > 0:
        for idx in range(num_baseline_fails):
            ax = plt.subplot(4, 4, idx + 1)
            error = baseline_fails_kd_succeeds[idx]

            # Denormalize
            mean = torch.tensor(config['augmentations']['normalization']['mean']).view(3, 1, 1)
            std = torch.tensor(config['augmentations']['normalization']['std']).view(3, 1, 1)
            img_denorm = error['image_tensor'] * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)

            ax.imshow(img_denorm.permute(1, 2, 0).numpy())
            ax.set_title(f"True: {error['true_label']}\nBase: {error['baseline_pred']} ✗\nKD: {error['kd_pred']} ✓",
                        fontsize=9, fontweight='bold', color='green')
            ax.axis('off')

    # Grid for KD failures (bottom half)
    num_kd_fails = min(len(kd_fails_baseline_succeeds), max_samples // 2)
    if num_kd_fails > 0:
        for idx in range(num_kd_fails):
            ax = plt.subplot(4, 4, 8 + idx + 1)
            error = kd_fails_baseline_succeeds[idx]

            # Denormalize
            mean = torch.tensor(config['augmentations']['normalization']['mean']).view(3, 1, 1)
            std = torch.tensor(config['augmentations']['normalization']['std']).view(3, 1, 1)
            img_denorm = error['image_tensor'] * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)

            ax.imshow(img_denorm.permute(1, 2, 0).numpy())
            ax.set_title(f"True: {error['true_label']}\nBase: {error['baseline_pred']} ✓\nKD: {error['kd_pred']} ✗",
                        fontsize=9, fontweight='bold', color='red')
            ax.axis('off')

    plt.suptitle(f'Differential Error Analysis: Baseline vs {best_kd}\nTop: KD Fixes Errors (Green) | Bottom: KD Introduces Errors (Red)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = output_dir / "error_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()


def save_comparison_summary(all_results, output_dir):
    """
    Save comprehensive comparison summary to text file.

    Args:
        all_results: Dictionary of all results
        output_dir: Directory to save summary
    """
    print("\nSaving comparison summary...")

    save_path = output_dir / "comparison_summary.txt"
    model_names = list(all_results.keys())

    with open(save_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("KNOWLEDGE DISTILLATION COMPARISON SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        # Overall comparison table
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Model':<25} {'Test Acc':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Parameters':>15}\n")
        f.write("-" * 100 + "\n")

        for model_name in model_names:
            results = all_results[model_name]
            acc = results['metrics']['accuracy'] * 100
            prec = results['metrics']['precision'] * 100
            rec = results['metrics']['recall'] * 100
            f1 = results['metrics']['f1'] * 100
            params = results['model']['num_parameters']

            f.write(f"{model_name:<25} {acc:>11.2f}% {prec:>11.2f}% {rec:>11.2f}% {f1:>11.2f}% {params:>15,}\n")

        f.write("\n")

        # KD Effectiveness
        f.write("KNOWLEDGE DISTILLATION EFFECTIVENESS:\n")
        f.write("-" * 100 + "\n")
        baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100
        teacher_acc = all_results['Teacher']['metrics']['accuracy'] * 100

        f.write(f"{'Model':<25} {'Test Acc':>12} {'Δ from Baseline':>18} {'Δ from Teacher':>18}\n")
        f.write("-" * 100 + "\n")

        kd_models = [k for k in model_names if k.startswith('KD')]
        for model_name in kd_models:
            acc = all_results[model_name]['metrics']['accuracy'] * 100
            delta_baseline = acc - baseline_acc
            delta_teacher = acc - teacher_acc

            f.write(f"{model_name:<25} {acc:>11.2f}% {delta_baseline:>+17.2f}% {delta_teacher:>+17.2f}%\n")

        f.write("\n")

        # Best configuration
        best_kd = identify_best_kd_model(all_results)
        best_acc = all_results[best_kd]['metrics']['accuracy'] * 100
        best_delta = best_acc - baseline_acc

        f.write("=" * 100 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 100 + "\n\n")

        f.write(f"1. BEST KD CONFIGURATION: {best_kd}\n")
        f.write(f"   Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"   Improvement over baseline: {best_delta:+.2f}%\n\n")

        # Hyperparameter insights
        T_best, alpha_best = extract_hyperparameters(best_kd)
        f.write(f"2. OPTIMAL HYPERPARAMETERS:\n")
        f.write(f"   Temperature (T): {T_best}\n")
        f.write(f"   Alpha (α): {alpha_best}\n\n")

        f.write(f"3. COMPRESSION:\n")
        teacher_params = all_results['Teacher']['model']['num_parameters']
        student_params = all_results['Student Baseline']['model']['num_parameters']
        compression = teacher_params / student_params
        f.write(f"   Student achieves {compression:.1f}x parameter compression\n")
        f.write(f"   Teacher: {teacher_params:,} parameters\n")
        f.write(f"   Student: {student_params:,} parameters\n\n")

        f.write(f"4. PERFORMANCE SUMMARY:\n")
        if best_delta > 0:
            f.write(f"   ✓ Knowledge distillation improves over baseline by {best_delta:.2f}%\n")
        elif abs(best_delta) < 0.5:
            f.write(f"   ≈ Knowledge distillation matches baseline performance\n")
        else:
            f.write(f"   ✗ Knowledge distillation underperforms baseline by {abs(best_delta):.2f}%\n")

        f.write("\n")
        f.write("=" * 100 + "\n")

    print(f"Saved to {save_path}")


# %% Main Execution

def main():
    """Main execution function."""
    print("=" * 100)
    print("KNOWLEDGE DISTILLATION COMPARISON ANALYSIS")
    print("=" * 100)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")

    # 1. Load all results
    print("\n" + "=" * 100)
    print("LOADING RESULTS")
    print("=" * 100)
    all_results = load_all_results()

    # 2. Generate visualizations
    print("\n" + "=" * 100)
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)

    plot_accuracy_comparison(all_results, RESULTS_DIR)
    plot_model_efficiency(all_results, RESULTS_DIR)
    plot_confusion_matrices_comparison(all_results, RESULTS_DIR)
    plot_per_class_metrics_comparison(all_results, RESULTS_DIR)
    plot_performance_vs_efficiency(all_results, RESULTS_DIR)
    plot_hyperparameter_sensitivity(all_results, RESULTS_DIR)
    plot_kd_effectiveness(all_results, RESULTS_DIR)
    analyze_differential_errors(all_results, RESULTS_DIR, max_samples=8)

    # 3. Save summary
    print("\n" + "=" * 100)
    print("SAVING COMPARISON SUMMARY")
    print("=" * 100)
    save_comparison_summary(all_results, RESULTS_DIR)

    # 4. Final summary
    print("\n" + "=" * 100)
    print("COMPARISON ANALYSIS COMPLETE!")
    print("=" * 100)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - accuracy_comparison.png")
    print(f"  - model_efficiency.png")
    print(f"  - confusion_matrices_comparison.png")
    print(f"  - per_class_metrics_comparison.png")
    print(f"  - performance_vs_efficiency.png")
    print(f"  - hyperparameter_sensitivity.png")
    print(f"  - kd_effectiveness.png")
    print(f"  - error_analysis.png")
    print(f"  - comparison_summary.txt")

    # Print key results
    print("\n" + "=" * 100)
    print("KEY RESULTS:")
    print("=" * 100)

    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100
    best_kd = identify_best_kd_model(all_results)
    best_acc = all_results[best_kd]['metrics']['accuracy'] * 100

    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"Best KD Model: {best_kd}")
    print(f"Best KD Accuracy: {best_acc:.2f}%")
    print(f"Improvement: {best_acc - baseline_acc:+.2f}%")


if __name__ == "__main__":
    main()
