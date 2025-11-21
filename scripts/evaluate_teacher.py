#!/usr/bin/env python3
"""
Teacher Model Evaluation and Analysis

Analyzes teacher model performance and generates comprehensive evaluation visualizations.
This script provides detailed analysis of the teacher model for the paper.

Usage:
    python scripts/evaluate_teacher.py

Outputs:
    - scripts/results/teacher/confusion_matrix.png
    - scripts/results/teacher/confusion_matrix_normalized.png
    - scripts/results/teacher/training_curves.png (if W&B accessible)
    - scripts/results/teacher/per_class_metrics.png
    - scripts/results/teacher/error_analysis.png
    - scripts/results/teacher/evaluation_summary.txt
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import plot_confusion_matrix, plot_training_curves, plot_sample_grid
from utils.env_utils import get_project_root, resolve_data_path
from utils.data_loader import CoralDataset
from utils.preprocessing import get_test_transforms
from models.teacher import TeacherModel
import yaml

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed, W&B features will be skipped")

# %% Configuration
PROJECT_ROOT = get_project_root()
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results" / "teacher"
EVAL_RESULTS_PATH = RESULTS_DIR / "test_results.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "teacher" / "best_model.pth"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# W&B configuration
WANDB_PROJECT = "coral-bleaching"
WANDB_RUN_ID = "lfidb03f"

# Class names
CLASS_NAMES = ['healthy', 'bleached']

# %% Helper Functions

def load_evaluation_results():
    """
    Load evaluation results from JSON file.

    Returns:
        dict: Evaluation results including metrics, confusion matrix, etc.
    """
    print(f"Loading evaluation results from {EVAL_RESULTS_PATH}...")

    if not EVAL_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Evaluation results not found at {EVAL_RESULTS_PATH}")

    with open(EVAL_RESULTS_PATH, 'r') as f:
        results = json.load(f)

    print(f"Loaded results for {results['dataset']['split']} split "
          f"({results['dataset']['num_samples']} samples)")
    print(f"Test Accuracy: {results['metrics']['accuracy']*100:.2f}%")

    return results


def fetch_wandb_history():
    """
    Fetch training history from Weights & Biases.

    Returns:
        dict or None: Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
                     Returns None if W&B is not available or run cannot be accessed.
    """
    if not WANDB_AVAILABLE:
        print("\nW&B not available, skipping training curves...")
        return None

    try:
        print(f"\nAttempting to fetch W&B run history (run ID: {WANDB_RUN_ID})...")

        # Initialize W&B API
        api = wandb.Api()

        # Try to get the run (may need entity/username)
        try:
            run = api.run(f"{WANDB_PROJECT}/{WANDB_RUN_ID}")
        except Exception:
            # Try with username if needed
            print("Trying to locate run with entity prefix...")
            # Get current user
            try:
                # Get all runs and find matching ID
                runs = api.runs(WANDB_PROJECT)
                run = None
                for r in runs:
                    if r.id == WANDB_RUN_ID:
                        run = r
                        break
                if run is None:
                    raise ValueError(f"Run {WANDB_RUN_ID} not found in project {WANDB_PROJECT}")
            except Exception as e:
                print(f"Could not locate W&B run: {e}")
                return None

        print(f"Successfully accessed W&B run: {run.name}")

        # Fetch history
        history = run.history()

        # Extract metrics
        train_losses = history['train/loss'].dropna().tolist()
        val_losses = history['val/loss'].dropna().tolist()
        train_accs = history['train/accuracy'].dropna().tolist()
        val_accs = history['val/accuracy'].dropna().tolist()

        print(f"Fetched {len(train_losses)} epochs of training data")

        return {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_acc': train_accs,
            'val_acc': val_accs
        }

    except Exception as e:
        print(f"\nWarning: Could not fetch W&B history: {e}")
        print("Skipping training curves visualization...")
        return None


def visualize_confusion_matrices(results, output_dir):
    """
    Generate confusion matrix visualizations (raw and normalized).

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    print("\nGenerating confusion matrix visualizations...")

    cm = np.array(results['metrics']['confusion_matrix'])

    # Raw confusion matrix
    fig1 = plot_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        normalize=False,
        save_path=output_dir / "confusion_matrix.png"
    )
    print(f"Saved raw confusion matrix to {output_dir / 'confusion_matrix.png'}")

    # Normalized confusion matrix
    fig2 = plot_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        normalize=True,
        save_path=output_dir / "confusion_matrix_normalized.png"
    )
    print(f"Saved normalized confusion matrix to {output_dir / 'confusion_matrix_normalized.png'}")


def visualize_training_curves(wandb_history, output_dir):
    """
    Generate training curves visualization from W&B history.

    Args:
        wandb_history: Dictionary with training history or None
        output_dir: Directory to save plots
    """
    if wandb_history is None:
        print("\nSkipping training curves (W&B history not available)")
        return

    print("\nGenerating training curves...")

    fig = plot_training_curves(
        train_losses=wandb_history['train_loss'],
        val_losses=wandb_history['val_loss'],
        train_accs=wandb_history['train_acc'],
        val_accs=wandb_history['val_acc'],
        save_path=output_dir / "training_curves.png"
    )
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")


def plot_per_class_metrics(results, output_dir):
    """
    Generate per-class performance comparison bar chart.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
    """
    print("\nGenerating per-class metrics comparison...")

    per_class = results['metrics']['per_class_metrics']

    # Extract metrics for each class
    classes = CLASS_NAMES
    precision = [per_class['healthy']['precision'], per_class['bleached']['precision']]
    recall = [per_class['healthy']['recall'], per_class['bleached']['recall']]
    f1 = [per_class['healthy']['f1'], per_class['bleached']['f1']]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(classes))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    # Customize plot
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Model - Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes])
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    save_path = output_dir / "per_class_metrics.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-class metrics to {save_path}")
    plt.close()


def analyze_errors(results, output_dir, max_samples=16):
    """
    Perform error analysis by visualizing misclassified test samples.

    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots
        max_samples: Maximum number of error samples to display
    """
    print(f"\nPerforming error analysis (showing up to {max_samples} misclassifications)...")

    # Load config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Load test dataset
    test_csv = DATA_SPLITS_DIR / "test.csv"
    import pandas as pd
    test_df = pd.read_csv(test_csv)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model without pretrained weights (we'll load from checkpoint)
    model = TeacherModel(num_classes=len(CLASS_NAMES), pretrained=False)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get test transforms
    transform = get_test_transforms(config)

    # Run inference to find errors
    print("Running inference on test set to identify errors...")
    errors = []  # List of (image_path, true_label, pred_label, image_tensor)

    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Finding errors"):
            img_path = resolve_data_path(row['image_path'])
            true_label = row['label']

            try:
                # Load and transform image
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Predict
                output = model(img_tensor)
                pred_idx = output.argmax(dim=1).item()
                pred_label = CLASS_NAMES[pred_idx]

                # Check if misclassified
                if pred_label != true_label:
                    errors.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'image_tensor': transform(img)  # Store transformed tensor
                    })

                    # Stop if we have enough errors
                    if len(errors) >= max_samples:
                        break

            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue

    print(f"Found {len(errors)} misclassifications (showing {min(len(errors), max_samples)})")

    if len(errors) == 0:
        print("No errors found! Model is perfect on test set.")
        return

    # Visualize errors using plot_sample_grid
    error_images = [e['image_tensor'] for e in errors]
    error_true_labels = [e['true_label'] for e in errors]
    error_pred_labels = [e['pred_label'] for e in errors]

    # Create custom labels showing true vs predicted
    error_labels = [f"True: {true}\nPred: {pred}"
                   for true, pred in zip(error_true_labels, error_pred_labels)]

    # Plot using custom matplotlib (since plot_sample_grid expects class indices)
    num_errors = len(error_images)
    grid_size = int(np.ceil(np.sqrt(num_errors)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 16))
    axes = axes.flatten() if num_errors > 1 else [axes]

    for idx, (img_tensor, label) in enumerate(zip(error_images, error_labels)):
        # Denormalize for display
        mean = torch.tensor(config['augmentations']['normalization']['mean']).view(3, 1, 1)
        std = torch.tensor(config['augmentations']['normalization']['std']).view(3, 1, 1)
        img_denorm = img_tensor * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)

        # Display
        axes[idx].imshow(img_denorm.permute(1, 2, 0).numpy())
        axes[idx].set_title(label, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_errors, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Teacher Model - Misclassified Test Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / "error_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved error analysis to {save_path}")
    plt.close()


def save_evaluation_summary(results, wandb_history, output_dir):
    """
    Save evaluation summary to text file.

    Args:
        results: Evaluation results dictionary
        wandb_history: W&B history or None
        output_dir: Directory to save summary
    """
    save_path = output_dir / "evaluation_summary.txt"

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEACHER MODEL EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Model info
        f.write("MODEL INFORMATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Type: {results['model']['type']}\n")
        f.write(f"Checkpoint: {results['model']['checkpoint']}\n")
        f.write(f"Parameters: {results['model']['num_parameters']:,}\n")
        f.write(f"Trainable Parameters: {results['model']['num_trainable_parameters']:,}\n")
        f.write(f"Best Validation Accuracy: {results['model']['checkpoint_best_val_acc']*100:.2f}%\n")
        f.write(f"Training Epoch: {results['model']['checkpoint_epoch']}\n\n")

        # Dataset info
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Split: {results['dataset']['split']}\n")
        f.write(f"Number of Samples: {results['dataset']['num_samples']}\n")
        f.write(f"Class Distribution:\n")
        for class_name, count in results['dataset']['class_distribution'].items():
            f.write(f"  {class_name.capitalize()}: {count}\n")
        f.write("\n")

        # Overall metrics
        f.write("OVERALL PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {results['metrics']['accuracy']*100:.2f}%\n")
        f.write(f"Precision (weighted): {results['metrics']['precision']*100:.2f}%\n")
        f.write(f"Recall (weighted): {results['metrics']['recall']*100:.2f}%\n")
        f.write(f"F1-Score (weighted): {results['metrics']['f1']*100:.2f}%\n\n")

        # Per-class metrics
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        for class_name in CLASS_NAMES:
            metrics = results['metrics']['per_class_metrics'][class_name]
            f.write(f"\n{class_name.upper()}:\n")
            f.write(f"  Precision: {metrics['precision']*100:.2f}%\n")
            f.write(f"  Recall:    {metrics['recall']*100:.2f}%\n")
            f.write(f"  F1-Score:  {metrics['f1']*100:.2f}%\n")
            f.write(f"  Support:   {metrics['support']} samples\n")

        # Confusion matrix
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 80 + "\n")
        cm = np.array(results['metrics']['confusion_matrix'])
        f.write("           Predicted\n")
        f.write("           Healthy  Bleached\n")
        f.write(f"Actual Healthy   {cm[0][0]:4d}     {cm[0][1]:4d}\n")
        f.write(f"       Bleached  {cm[1][0]:4d}     {cm[1][1]:4d}\n\n")

        # Training info (if available)
        if wandb_history is not None:
            f.write("=" * 80 + "\n")
            f.write("TRAINING INFORMATION:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Epochs: {len(wandb_history['train_loss'])}\n")
            f.write(f"Final Train Loss: {wandb_history['train_loss'][-1]:.4f}\n")
            f.write(f"Final Val Loss: {wandb_history['val_loss'][-1]:.4f}\n")
            f.write(f"Final Train Acc: {wandb_history['train_acc'][-1]*100:.2f}%\n")
            f.write(f"Final Val Acc: {wandb_history['val_acc'][-1]*100:.2f}%\n")
            f.write(f"Best Val Acc: {max(wandb_history['val_acc'])*100:.2f}%\n\n")

        f.write("=" * 80 + "\n")
        f.write(f"Evaluation completed: {results['timestamp']}\n")
        f.write("=" * 80 + "\n")

    print(f"Saved evaluation summary to {save_path}")


# %% Main Execution

def main():
    """Main execution function."""
    print("=" * 80)
    print("TEACHER MODEL EVALUATION AND ANALYSIS")
    print("=" * 80)

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")

    # 1. Load evaluation results
    print("\n" + "=" * 80)
    print("LOADING EVALUATION RESULTS")
    print("=" * 80)
    results = load_evaluation_results()

    # 2. Fetch W&B history
    print("\n" + "=" * 80)
    print("FETCHING TRAINING HISTORY FROM WANDB")
    print("=" * 80)
    wandb_history = fetch_wandb_history()

    # 3. Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    visualize_confusion_matrices(results, RESULTS_DIR)
    visualize_training_curves(wandb_history, RESULTS_DIR)
    plot_per_class_metrics(results, RESULTS_DIR)
    analyze_errors(results, RESULTS_DIR, max_samples=16)

    # 4. Save summary
    print("\n" + "=" * 80)
    print("SAVING EVALUATION SUMMARY")
    print("=" * 80)
    save_evaluation_summary(results, wandb_history, RESULTS_DIR)

    # 5. Final summary
    print("\n" + "=" * 80)
    print("EVALUATION ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print(f"  - confusion_matrix.png")
    print(f"  - confusion_matrix_normalized.png")
    if wandb_history is not None:
        print(f"  - training_curves.png")
    print(f"  - per_class_metrics.png")
    print(f"  - error_analysis.png")
    print(f"  - evaluation_summary.txt")

    # Print key metrics
    print("\n" + "=" * 80)
    print("KEY METRICS:")
    print("=" * 80)
    print(f"Test Accuracy: {results['metrics']['accuracy']*100:.2f}%")
    print(f"Healthy - Precision: {results['metrics']['per_class_metrics']['healthy']['precision']*100:.2f}%, "
          f"Recall: {results['metrics']['per_class_metrics']['healthy']['recall']*100:.2f}%")
    print(f"Bleached - Precision: {results['metrics']['per_class_metrics']['bleached']['precision']*100:.2f}%, "
          f"Recall: {results['metrics']['per_class_metrics']['bleached']['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
