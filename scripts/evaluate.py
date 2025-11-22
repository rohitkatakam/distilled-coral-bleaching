#!/usr/bin/env python3
"""
Universal Evaluation Script for Coral Bleaching Classifiers

Evaluates any model checkpoint (teacher, student baseline, distilled student) on a specified dataset split.
Computes comprehensive metrics and saves results to JSON.

Usage:
    # Evaluate teacher model
    python scripts/evaluate.py --checkpoint checkpoints/teacher/best_model.pth --model-type teacher

    # Evaluate student baseline
    python scripts/evaluate.py --checkpoint checkpoints/student_baseline/best_model.pth --model-type student

    # Evaluate distilled student (Phase 4)
    python scripts/evaluate.py --checkpoint checkpoints/student_kd/best_model.pth --model-type student
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher import TeacherModel
from models.student import StudentModel
from utils.data_loader import build_dataloaders
from utils.metrics import (
    compute_accuracy,
    compute_classification_metrics,
    compute_confusion_matrix
)


def load_checkpoint(checkpoint_path):
    """Load checkpoint from path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        checkpoint: Checkpoint dictionary
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def load_model(checkpoint_path, config, model_type, device):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        model_type: Model type ('teacher' or 'student')
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
        checkpoint: Checkpoint dictionary (for metadata)
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Initialize model based on type
    # NOTE: Set pretrained=False since we're loading weights from checkpoint
    if model_type == 'teacher':
        model = TeacherModel(
            num_classes=config['model'].get('num_classes', 2),
            pretrained=False,  # Don't download ImageNet weights - we're loading from checkpoint
            dropout=config['model'].get('dropout', None)
        )
    elif model_type == 'student':
        model = StudentModel(
            num_classes=config['model'].get('num_classes', 2),
            pretrained=False,  # Don't download ImageNet weights - we're loading from checkpoint
            dropout=config['model'].get('dropout', None)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded {model_type} model from {checkpoint_path}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    best_val_acc = checkpoint.get('best_val_acc', None)
    if best_val_acc is not None:
        # Checkpoint stores accuracy as decimal (0.0-1.0), convert to percentage
        print(f"  Best val acc: {best_val_acc * 100:.2f}%")
    else:
        print(f"  Best val acc: N/A")

    return model, checkpoint


def count_parameters(model):
    """Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        total_params: Total parameter count
        trainable_params: Trainable parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def run_inference(model, dataloader, device):
    """Run inference and collect predictions.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader for evaluation
        device: Device to run inference on

    Returns:
        all_preds: List of predicted class indices
        all_labels: List of ground truth class indices
        all_probs: List of prediction probabilities (softmax outputs)
    """
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(preds, labels, probs):
    """Compute comprehensive evaluation metrics.

    Args:
        preds: Predicted class indices
        labels: Ground truth class indices
        probs: Prediction probabilities

    Returns:
        metrics: Dictionary of metrics
    """
    # Overall metrics (all functions expect y_true, y_pred order)
    accuracy = compute_accuracy(labels, preds)
    clf_metrics = compute_classification_metrics(labels, preds)
    cm = compute_confusion_matrix(labels, preds)

    # Per-class metrics
    per_class = {}
    class_names = ['healthy', 'bleached']
    for i, class_name in enumerate(class_names):
        # Count support from labels
        support = int((labels == i).sum())
        per_class[class_name] = {
            'precision': float(clf_metrics['precision_per_class'][i]),
            'recall': float(clf_metrics['recall_per_class'][i]),
            'f1': float(clf_metrics['f1_per_class'][i]),
            'support': support
        }

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(clf_metrics['precision_macro']),
        'recall': float(clf_metrics['recall_macro']),
        'f1': float(clf_metrics['f1_macro']),
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': per_class
    }

    return metrics


def save_results(results, output_path):
    """Save results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


def print_summary(results):
    """Print evaluation results summary.

    Args:
        results: Results dictionary
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nModel: {results['model']['type']}")
    print(f"Checkpoint: {results['model']['checkpoint']}")
    print(f"Parameters: {results['model']['num_parameters']:,}")

    print(f"\nDataset: {results['dataset']['split']}")
    print(f"Samples: {results['dataset']['num_samples']}")
    print(f"Class distribution: {results['dataset']['class_distribution']}")

    metrics = results['metrics']
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    print(f"\nPer-Class Metrics:")
    for class_name, class_metrics in metrics['per_class_metrics'].items():
        print(f"  {class_name.capitalize()}:")
        print(f"    Precision: {class_metrics['precision']:.4f}")
        print(f"    Recall:    {class_metrics['recall']:.4f}")
        print(f"    F1 Score:  {class_metrics['f1']:.4f}")
        print(f"    Support:   {class_metrics['support']}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Healthy  Bleached")
    print(f"  Actual")
    print(f"    Healthy     {cm[0][0]:3d}      {cm[0][1]:3d}")
    print(f"    Bleached    {cm[1][0]:3d}      {cm[1][1]:3d}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate coral bleaching classifier")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='scripts/results',
                        help='Directory to save results')
    parser.add_argument('--model-type', type=str, default='teacher',
                        choices=['teacher', 'student'],
                        help='Type of model to evaluate')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use for inference')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override batch size if specified
    if args.batch_size != 32:
        config['training']['batch_size'] = args.batch_size

    # Set device
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading {args.model_type} model...")
    model, checkpoint = load_model(args.checkpoint, config, args.model_type, device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataloaders = build_dataloaders(config, splits=[args.split])
    dataloader = dataloaders[args.split]
    print(f"  Batches: {len(dataloader)}")
    print(f"  Samples: {len(dataloader.dataset)}")

    # Run inference
    print(f"\nRunning inference on {args.split} set...")
    preds, labels, probs = run_inference(model, dataloader, device)

    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(preds, labels, probs)

    # Prepare results
    results = {
        'model': {
            'type': args.model_type,
            'checkpoint': args.checkpoint,
            'num_parameters': total_params,
            'num_trainable_parameters': trainable_params,
            'checkpoint_epoch': checkpoint.get('epoch', None),
            'checkpoint_best_val_acc': checkpoint.get('best_val_acc', None)
        },
        'dataset': {
            'split': args.split,
            'num_samples': len(labels),
            'class_distribution': {
                'healthy': int((labels == 0).sum()),
                'bleached': int((labels == 1).sum())
            }
        },
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'batch_size': args.batch_size,
            'device': str(device)
        }
    }

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path(args.output_dir) / args.model_type / f'{args.split}_results.json'
    save_results(results, output_path)

    print(f"\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
