#!/usr/bin/env python3
"""
Main Results Figure Generator for Course Paper

Generates a single 2×2 multi-panel figure combining key findings:
- Panel A (top-left): Test accuracy comparison (6 models)
- Panel B (top-right): Parameter count comparison
- Panel C (bottom-left): Performance vs efficiency scatter
- Panel D (bottom-right): KD effectiveness (Δ from baseline)

Usage:
    python scripts/generate_main_figure.py

Outputs:
    - scripts/results/paper_figures/main_results.png
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.env_utils import get_project_root

# Configuration
PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "paper_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model results paths
RESULTS_PATHS = {
    'Teacher': PROJECT_ROOT / "scripts" / "results" / "teacher" / "test_results.json",
    'Student Baseline': PROJECT_ROOT / "scripts" / "results" / "student" / "test_results.json",
    'KD (T=2.0, α=0.5)': PROJECT_ROOT / "scripts" / "results" / "kd_t2.0_a0.5" / "student" / "test_results.json",
    'KD (T=4.0, α=0.3)': PROJECT_ROOT / "scripts" / "results" / "kd_t4.0_a0.3" / "student" / "test_results.json",
    'KD (T=4.0, α=0.7)': PROJECT_ROOT / "scripts" / "results" / "kd_t4.0_a0.7" / "student" / "test_results.json",
    'KD (T=8.0, α=0.9)': PROJECT_ROOT / "scripts" / "results" / "kd_t8.0_a0.9" / "student" / "test_results.json",
}

# Model parameters (from training logs)
MODEL_PARAMS = {
    'Teacher': 23.5e6,  # 23.5M parameters
    'Student Baseline': 1.52e6,  # 1.52M parameters
    'KD (T=2.0, α=0.5)': 1.52e6,
    'KD (T=4.0, α=0.3)': 1.52e6,
    'KD (T=4.0, α=0.7)': 1.52e6,
    'KD (T=8.0, α=0.9)': 1.52e6,
}

# Color scheme
COLORS = {
    'Teacher': '#3498db',           # Blue
    'Student Baseline': '#e67e22',  # Orange
    'KD (T=2.0, α=0.5)': '#27ae60', # Green (best)
    'KD (T=4.0, α=0.3)': '#9b59b6', # Purple
    'KD (T=4.0, α=0.7)': '#e74c3c', # Red
    'KD (T=8.0, α=0.9)': '#f39c12', # Yellow
}


def load_all_results():
    """
    Load evaluation results for all models.

    Returns:
        dict: Dictionary mapping model names to results dictionaries
    """
    print("Loading evaluation results for 6 models...")

    all_results = {}
    for model_name, results_path in RESULTS_PATHS.items():
        if not results_path.exists():
            raise FileNotFoundError(f"Results not found: {results_path}")

        with open(results_path, 'r') as f:
            all_results[model_name] = json.load(f)

        acc = all_results[model_name]['metrics']['accuracy'] * 100
        print(f"  {model_name}: {acc:.2f}% test accuracy")

    return all_results


def plot_accuracy_comparison(ax, all_results):
    """
    Panel A: Test accuracy bar chart (6 models).

    Args:
        ax: Matplotlib axis
        all_results: Dictionary of all results
    """
    # Prepare data
    model_names = list(all_results.keys())
    accuracies = [all_results[m]['metrics']['accuracy'] * 100 for m in model_names]
    colors_list = [COLORS[m] for m in model_names]

    # Create bar chart
    x = np.arange(len(model_names))
    bars = ax.bar(x, accuracies, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_xlabel('Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('(A) Test Accuracy Comparison', fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
    ax.set_ylim(75, 80.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'{acc:.2f}%',
               ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Add baseline reference line
    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
              label=f'Baseline ({baseline_acc:.2f}%)')
    ax.legend(loc='lower right', fontsize=8)


def plot_parameter_comparison(ax, all_results):
    """
    Panel B: Parameter count comparison.

    Args:
        ax: Matplotlib axis
        all_results: Dictionary of all results
    """
    # Prepare data (group by unique param counts)
    # Teacher: 23.5M, All students: 1.52M
    model_types = ['Teacher', 'All Students']
    params = [23.5, 1.52]  # In millions
    colors = ['#3498db', '#27ae60']

    # Create bar chart
    bars = ax.bar(model_types, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_ylabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax.set_title('(B) Model Size Comparison', fontsize=12, fontweight='bold', loc='left')
    ax.set_ylim(0, 26)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
               f'{param:.2f}M',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add compression ratio annotation
    compression = params[0] / params[1]
    ax.text(0.5, 12, f'15.5× compression', ha='center', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))


def plot_efficiency_scatter(ax, all_results):
    """
    Panel C: Performance vs efficiency scatter plot.

    Args:
        ax: Matplotlib axis
        all_results: Dictionary of all results
    """
    # Prepare data
    model_names = list(all_results.keys())
    accuracies = [all_results[m]['metrics']['accuracy'] * 100 for m in model_names]
    params = [MODEL_PARAMS[m] / 1e6 for m in model_names]  # Convert to millions
    colors_list = [COLORS[m] for m in model_names]

    # Create scatter plot
    for i, (name, acc, param, color) in enumerate(zip(model_names, accuracies, params, colors_list)):
        ax.scatter(param, acc, s=150, color=color, alpha=0.8, edgecolor='black', linewidth=1.2, zorder=3)

        # Add labels with smart positioning
        offset_x = 1.5 if param > 10 else 0.3
        offset_y = 0.3 if i % 2 == 0 else -0.3
        ax.text(param + offset_x, acc + offset_y, name.replace('KD ', ''),
               fontsize=7, ha='left' if param > 10 else 'left')

    # Customize
    ax.set_xlabel('Parameters (Millions)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('(C) Accuracy vs Model Size', fontsize=12, fontweight='bold', loc='left')
    ax.set_xlim(-1, 26)
    ax.set_ylim(75.5, 79.8)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)


def plot_kd_effectiveness(ax, all_results):
    """
    Panel D: KD effectiveness (Δ accuracy from baseline).

    Args:
        ax: Matplotlib axis
        all_results: Dictionary of all results
    """
    # Calculate deltas from baseline
    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100

    kd_models = [k for k in all_results.keys() if k.startswith('KD')]
    deltas = []
    colors_list = []

    for model in kd_models:
        acc = all_results[model]['metrics']['accuracy'] * 100
        delta = acc - baseline_acc
        deltas.append(delta)
        colors_list.append(COLORS[model])

    # Create bar chart
    x = np.arange(len(kd_models))
    bars = ax.bar(x, deltas, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_xlabel('KD Configuration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Δ Accuracy vs Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_title('(D) KD Effectiveness', fontsize=12, fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('KD ', '') for m in kd_models], rotation=20, ha='right', fontsize=8)
    ax.set_ylim(-3, 1.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7, label='Baseline')

    # Add value labels on bars
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        y_pos = height + 0.15 if height > 0 else height - 0.25
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{delta:+.2f}%',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=8, fontweight='bold')

    ax.legend(loc='upper right', fontsize=8)


def create_main_figure():
    """
    Create 2×2 multi-panel figure combining all key results.
    """
    print("\n" + "="*60)
    print("GENERATING MAIN RESULTS FIGURE (2×2 LAYOUT)")
    print("="*60)

    # Load all results
    all_results = load_all_results()

    # Create figure with 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Knowledge Distillation for Coral Bleaching Classification: Main Results',
                fontsize=15, fontweight='bold', y=0.995)

    # Generate each panel
    print("\nGenerating Panel A: Accuracy comparison...")
    plot_accuracy_comparison(axes[0, 0], all_results)

    print("Generating Panel B: Parameter comparison...")
    plot_parameter_comparison(axes[0, 1], all_results)

    print("Generating Panel C: Efficiency scatter...")
    plot_efficiency_scatter(axes[1, 0], all_results)

    print("Generating Panel D: KD effectiveness...")
    plot_kd_effectiveness(axes[1, 1], all_results)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save figure
    output_path = RESULTS_DIR / "main_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Main results figure saved to: {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("FIGURE SUMMARY")
    print("="*60)
    print(f"Layout: 2×2 grid (4 subpanels)")
    print(f"Models: 6 (Teacher, Student Baseline, 4 KD variants)")
    print(f"Best KD: {max([(k, v['metrics']['accuracy']*100) for k, v in all_results.items() if k.startswith('KD')], key=lambda x: x[1])[0]}")
    best_kd_name = max([(k, v['metrics']['accuracy']*100) for k, v in all_results.items() if k.startswith('KD')], key=lambda x: x[1])[0]
    best_kd_acc = all_results[best_kd_name]['metrics']['accuracy'] * 100
    baseline_acc = all_results['Student Baseline']['metrics']['accuracy'] * 100
    improvement = best_kd_acc - baseline_acc
    print(f"Improvement: +{improvement:.2f}% over baseline ({baseline_acc:.2f}% → {best_kd_acc:.2f}%)")
    print(f"Compression: 15.5× (23.5M → 1.52M parameters)")
    print("="*60)

    plt.close()


if __name__ == "__main__":
    create_main_figure()
    print("\n✓ Main figure generation complete!")
