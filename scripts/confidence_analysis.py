#!/usr/bin/env python3
"""
Confidence Distribution Analysis for Knowledge Distillation

Analyzes prediction confidence across teacher, student baseline, and distilled models
to demonstrate that KD improves not just accuracy but also prediction certainty.

Generates two plots:
1. Histogram of maximum prediction confidence for each model
2. Average confidence for correct vs incorrect predictions

Usage:
    python scripts/confidence_analysis.py

Outputs:
    - scripts/results/paper_figures/confidence_distributions.png
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import TeacherModel
from models.student import StudentModel
from utils.data_loader import build_dataloaders
from utils.env_utils import get_project_root

# Configuration
PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "paper_figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Model checkpoints
CHECKPOINT_PATHS = {
    'Teacher': PROJECT_ROOT / "checkpoints" / "teacher" / "best_model.pth",
    'Student Baseline': PROJECT_ROOT / "checkpoints" / "student_baseline" / "best_model.pth",
    'KD (T=2.0, α=0.5)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t2.0_a0.5.pth",
    'KD (T=4.0, α=0.3)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t4.0_a0.3.pth",
    'KD (T=4.0, α=0.7)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t4.0_a0.7.pth",
    'KD (T=8.0, α=0.9)': PROJECT_ROOT / "checkpoints" / "student_kd" / "best_model_t8.0_a0.9.pth",
}

# Model types
MODEL_TYPES = {
    'Teacher': 'teacher',
    'Student Baseline': 'student',
    'KD (T=2.0, α=0.5)': 'student',
    'KD (T=4.0, α=0.3)': 'student',
    'KD (T=4.0, α=0.7)': 'student',
    'KD (T=8.0, α=0.9)': 'student',
}

# Color scheme
COLORS = {
    'Teacher': '#3498db',
    'Student Baseline': '#e67e22',
    'KD (T=2.0, α=0.5)': '#27ae60',
    'KD (T=4.0, α=0.3)': '#9b59b6',
    'KD (T=4.0, α=0.7)': '#e74c3c',
    'KD (T=8.0, α=0.9)': '#f39c12',
}


def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, model_type, config, device):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_type: Model type ('teacher' or 'student')
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model
    if model_type == 'teacher':
        model = TeacherModel(
            num_classes=config['model'].get('num_classes', 2),
            pretrained=False,
            dropout=config['model'].get('dropout', None)
        )
    elif model_type == 'student':
        model = StudentModel(
            num_classes=config['model'].get('num_classes', 2),
            pretrained=False,
            dropout=config['model'].get('dropout', None)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model


def get_predictions_with_confidence(model, dataloader, device):
    """
    Run inference and collect prediction probabilities.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader for test set
        device: Device to run inference on

    Returns:
        all_probs: Array of prediction probabilities (N x num_classes)
        all_labels: Array of ground truth labels (N,)
        all_correct: Boolean array indicating correct predictions (N,)
    """
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Running inference", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            # Collect probabilities and labels
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate batches
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Determine correctness
    preds = np.argmax(all_probs, axis=1)
    all_correct = (preds == all_labels)

    return all_probs, all_labels, all_correct


def plot_confidence_histograms(all_model_data, output_dir):
    """
    Plot overlapping histograms of maximum prediction confidence for each model.

    Args:
        all_model_data: Dictionary mapping model names to (probs, labels, correct) tuples
        output_dir: Directory to save plot
    """
    print("\nGenerating confidence distribution histograms...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # For each model, plot histogram of max confidence
    for model_name, (probs, labels, correct) in all_model_data.items():
        # Get max confidence for each prediction
        max_confidences = np.max(probs, axis=1)

        # Plot histogram
        ax.hist(max_confidences, bins=30, alpha=0.5, label=model_name,
               color=COLORS[model_name], edgecolor='black', linewidth=0.8)

    # Customize
    ax.set_xlabel('Maximum Prediction Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distributions Across Models', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    # Add vertical line at 0.5 (random baseline)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Random (50%)')

    plt.tight_layout()
    output_path = output_dir / "confidence_histograms.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def plot_confidence_by_correctness(all_model_data, output_dir):
    """
    Plot average confidence for correct vs incorrect predictions (grouped bar chart).

    Args:
        all_model_data: Dictionary mapping model names to (probs, labels, correct) tuples
        output_dir: Directory to save plot
    """
    print("\nGenerating confidence by correctness comparison...")

    model_names = list(all_model_data.keys())
    correct_confidences = []
    incorrect_confidences = []

    # Calculate average confidence for correct/incorrect predictions
    for model_name in model_names:
        probs, labels, correct = all_model_data[model_name]
        max_confidences = np.max(probs, axis=1)

        # Average confidence for correct predictions
        if np.any(correct):
            avg_correct = np.mean(max_confidences[correct])
        else:
            avg_correct = 0.0

        # Average confidence for incorrect predictions
        if np.any(~correct):
            avg_incorrect = np.mean(max_confidences[~correct])
        else:
            avg_incorrect = 0.0

        correct_confidences.append(avg_correct)
        incorrect_confidences.append(avg_incorrect)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, correct_confidences, width, label='Correct Predictions',
                  color='green', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, incorrect_confidences, width, label='Incorrect Predictions',
                  color='red', alpha=0.7, edgecolor='black', linewidth=1.2)

    # Customize
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Average Prediction Confidence: Correct vs Incorrect', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=7, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "confidence_by_correctness.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    plt.close()


def main():
    """Main confidence analysis workflow."""
    print("\n" + "="*70)
    print("CONFIDENCE DISTRIBUTION ANALYSIS")
    print("="*70)

    # Load configuration
    config = load_config()
    device = torch.device('cpu')  # CPU-based evaluation

    # Load test dataloader
    print("\nLoading test dataset...")
    dataloaders_dict = build_dataloaders(config, splits=['test'])
    test_loader = dataloaders_dict['test']
    print(f"  ✓ Test set: {len(test_loader.dataset)} samples")

    # Load all models and collect predictions
    print("\nLoading models and collecting predictions...")
    all_model_data = {}

    for model_name, checkpoint_path in CHECKPOINT_PATHS.items():
        print(f"\n  {model_name}:")
        model_type = MODEL_TYPES[model_name]

        # Load model
        model = load_model(checkpoint_path, model_type, config, device)
        print(f"    ✓ Model loaded")

        # Get predictions with confidence
        probs, labels, correct = get_predictions_with_confidence(model, test_loader, device)
        accuracy = np.mean(correct) * 100
        avg_confidence = np.mean(np.max(probs, axis=1))

        print(f"    ✓ Accuracy: {accuracy:.2f}%")
        print(f"    ✓ Avg confidence: {avg_confidence:.3f}")

        all_model_data[model_name] = (probs, labels, correct)

    # Generate plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)

    plot_confidence_histograms(all_model_data, RESULTS_DIR)
    plot_confidence_by_correctness(all_model_data, RESULTS_DIR)

    # Print summary statistics
    print("\n" + "="*70)
    print("CONFIDENCE ANALYSIS SUMMARY")
    print("="*70)

    for model_name, (probs, labels, correct) in all_model_data.items():
        max_confidences = np.max(probs, axis=1)
        avg_conf_all = np.mean(max_confidences)
        avg_conf_correct = np.mean(max_confidences[correct]) if np.any(correct) else 0
        avg_conf_incorrect = np.mean(max_confidences[~correct]) if np.any(~correct) else 0

        print(f"\n{model_name}:")
        print(f"  Overall avg confidence: {avg_conf_all:.3f}")
        print(f"  Correct predictions:   {avg_conf_correct:.3f}")
        print(f"  Incorrect predictions: {avg_conf_incorrect:.3f}")
        print(f"  Confidence gap:        {avg_conf_correct - avg_conf_incorrect:.3f}")

    print("\n" + "="*70)
    print("✓ Confidence analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
