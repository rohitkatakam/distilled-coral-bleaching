"""CLI entry point to train the student model with knowledge distillation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml

from models.student import StudentModel
from models.teacher import TeacherModel
from models.distillation import DistillationLoss
from utils.data_loader import build_dataloaders
from utils.env_utils import is_colab, resolve_checkpoint_path, get_project_root
from utils.metrics import compute_accuracy, compute_classification_metrics


def parse_args(args=None) -> argparse.Namespace:
    """Parse command-line arguments for knowledge distillation training."""
    parser = argparse.ArgumentParser(description="Train student model with knowledge distillation.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to the training config.")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to trained teacher checkpoint (REQUIRED).")
    parser.add_argument("--output-dir", type=str, default="checkpoints/student_kd", help="Directory to store KD student checkpoints.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config).")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--temperature", type=float, default=None, help="Distillation temperature (overrides config).")
    parser.add_argument("--alpha", type=float, default=None, help="Distillation alpha weight (overrides config).")
    parser.add_argument("--wandb-project", type=str, default="coral-bleaching", help="W&B project name.")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"], help="W&B logging mode.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, overrides config).")
    parser.add_argument("--no-pretrained", action="store_true", help="Don't use pretrained weights for student.")
    return parser.parse_args(args)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict, args: argparse.Namespace) -> torch.device:
    """Setup device for training (CPU or CUDA)."""
    if args.device:
        device_str = args.device
    else:
        device_str = config.get('training', {}).get('device', 'cuda')

    # Use CPU if CUDA is not available
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device_str = 'cpu'

    return torch.device(device_str)


def load_teacher_model(checkpoint_path: str, device: torch.device, num_classes: int = 2) -> TeacherModel:
    """Load and freeze teacher model from checkpoint.

    Args:
        checkpoint_path: Path to teacher checkpoint file.
        device: Device to load model onto.
        num_classes: Number of output classes.

    Returns:
        Frozen teacher model in eval mode.
    """
    print(f"\nLoading teacher model from {checkpoint_path}...")

    # Instantiate teacher (no pretrained since loading from checkpoint)
    teacher = TeacherModel(num_classes=num_classes, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    teacher.load_state_dict(checkpoint['model_state_dict'])

    # Move to device
    teacher = teacher.to(device)

    # Freeze all parameters
    for param in teacher.parameters():
        param.requires_grad = False

    # Set to evaluation mode
    teacher.eval()

    print(f"Teacher loaded successfully (params: {teacher.get_num_total_params():,})")
    print(f"Teacher frozen: all parameters have requires_grad=False")

    return teacher


def setup_wandb(args: argparse.Namespace, config: dict, temperature: float, alpha: float, run_name: str = None):
    """Initialize Weights & Biases logging."""
    wandb_config = {
        'model': 'student_kd',
        'architecture': config.get('model', {}).get('student', {}).get('name', 'mobilenet_v3_small'),
        'teacher_checkpoint': args.teacher_checkpoint,
        'temperature': temperature,
        'alpha': alpha,
        'epochs': config.get('training', {}).get('epochs', 50),
        'batch_size': config.get('training', {}).get('batch_size', 32),
        'learning_rate': config.get('training', {}).get('learning_rate', 0.001),
        'optimizer': config.get('training', {}).get('optimizer', 'adam'),
        'scheduler': config.get('training', {}).get('scheduler', 'cosine'),
    }

    # Override with CLI args
    if args.epochs:
        wandb_config['epochs'] = args.epochs
    if args.batch_size:
        wandb_config['batch_size'] = args.batch_size
    if args.lr:
        wandb_config['learning_rate'] = args.lr

    wandb.init(
        project=args.wandb_project,
        name=run_name or f"student-kd-t{temperature}-a{alpha}",
        config=wandb_config,
        mode=args.wandb_mode
    )


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, scheduler, epoch: int,
                   best_val_acc: float, checkpoint_path: str, temperature: float, alpha: float,
                   teacher_checkpoint: str, metrics: dict = None):
    """Save model checkpoint with KD metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_acc': best_val_acc,
        'temperature': temperature,
        'alpha': alpha,
        'teacher_checkpoint': teacher_checkpoint,
        'metrics': metrics or {}
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None,
                   scheduler=None) -> tuple:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    temperature = checkpoint.get('temperature', 4.0)
    alpha = checkpoint.get('alpha', 0.7)

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Best Val Acc: {best_val_acc:.4f}")
    print(f"  Temperature: {temperature}, Alpha: {alpha}")

    return epoch, best_val_acc, temperature, alpha


def train_one_epoch_kd(teacher: nn.Module, student: nn.Module, train_loader, kd_criterion: DistillationLoss,
                       optimizer, device: torch.device, epoch: int) -> dict:
    """Train student for one epoch with knowledge distillation.

    Args:
        teacher: Frozen teacher model.
        student: Student model being trained.
        train_loader: Training data loader.
        kd_criterion: DistillationLoss module.
        optimizer: Optimizer for student.
        device: Training device.
        epoch: Current epoch number.

    Returns:
        Dictionary of training metrics.
    """
    teacher.eval()  # Ensure teacher stays in eval mode
    student.train()

    running_loss = 0.0
    running_kd_loss = 0.0
    running_hard_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass through teacher (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher(images)

        # Forward pass through student
        optimizer.zero_grad()
        student_outputs = student(images)

        # Compute distillation loss
        total_loss, loss_metrics = kd_criterion(
            student_logits=student_outputs,
            teacher_logits=teacher_outputs,
            hard_labels=labels,
            include_hard_loss=True
        )

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += total_loss.item()
        running_kd_loss += loss_metrics['kd_loss'].item()
        running_hard_loss += loss_metrics['hard_loss'].item()

        _, predicted = student_outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log batch metrics (every 10 batches)
        if (batch_idx + 1) % 10 == 0:
            batch_acc = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            avg_kd = running_kd_loss / (batch_idx + 1)
            avg_hard = running_hard_loss / (batch_idx + 1)
            print(f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {avg_loss:.4f} (KD: {avg_kd:.4f}, Hard: {avg_hard:.4f}) "
                  f"Acc: {batch_acc:.2f}%")

    # Epoch metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_kd_loss = running_kd_loss / len(train_loader)
    epoch_hard_loss = running_hard_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    metrics = {
        'train/loss': epoch_loss,
        'train/kd_loss': epoch_kd_loss,
        'train/hard_loss': epoch_hard_loss,
        'train/accuracy': epoch_acc,
        'epoch': epoch
    }

    return metrics


def validate_kd(teacher: nn.Module, student: nn.Module, val_loader, kd_criterion: DistillationLoss,
                ce_criterion, device: torch.device) -> dict:
    """Validate student model with both CE and KD losses.

    Args:
        teacher: Frozen teacher model.
        student: Student model being validated.
        val_loader: Validation data loader.
        kd_criterion: DistillationLoss module.
        ce_criterion: CrossEntropy loss for standard validation.
        device: Device for computation.

    Returns:
        Dictionary of validation metrics.
    """
    teacher.eval()
    student.eval()

    running_ce_loss = 0.0
    running_kd_loss = 0.0
    running_hard_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward passes
            teacher_outputs = teacher(images)
            student_outputs = student(images)

            # Compute CE loss (for early stopping)
            ce_loss = ce_criterion(student_outputs, labels)
            running_ce_loss += ce_loss.item()

            # Compute KD loss (for monitoring)
            total_kd_loss, loss_metrics = kd_criterion(
                student_logits=student_outputs,
                teacher_logits=teacher_outputs,
                hard_labels=labels,
                include_hard_loss=True
            )
            running_kd_loss += loss_metrics['kd_loss'].item()
            running_hard_loss += loss_metrics['hard_loss'].item()

            # Track predictions
            _, predicted = student_outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    val_ce_loss = running_ce_loss / len(val_loader)
    val_kd_loss = running_kd_loss / len(val_loader)
    val_hard_loss = running_hard_loss / len(val_loader)
    val_acc = compute_accuracy(all_labels, all_predictions)
    classification_metrics = compute_classification_metrics(all_labels, all_predictions)

    metrics = {
        'val/loss': val_ce_loss,  # Use CE for early stopping
        'val/kd_loss': val_kd_loss,
        'val/hard_loss': val_hard_loss,
        'val/accuracy': val_acc,
        'val/precision': classification_metrics['precision_macro'],
        'val/recall': classification_metrics['recall_macro'],
        'val/f1': classification_metrics['f1_macro']
    }

    return metrics


def main(args=None) -> None:
    """Execute knowledge distillation training workflow."""
    # Parse arguments
    if args is None:
        args = parse_args()
    elif isinstance(args, list):
        args = parse_args(args)

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = setup_device(config, args)
    print(f"Using device: {device}")
    print(f"Running in {'Colab' if is_colab() else 'local'} environment")

    # Get distillation hyperparameters (CLI args override config)
    temperature = args.temperature if args.temperature else config['model']['distillation']['temperature']
    alpha = args.alpha if args.alpha else config['model']['distillation']['alpha']

    print(f"\nDistillation hyperparameters:")
    print(f"  Temperature (T): {temperature}")
    print(f"  Alpha (α): {alpha}")
    print(f"  Hard weight (1-α): {1-alpha}")

    # Load frozen teacher model
    teacher_checkpoint_path = resolve_checkpoint_path(args.teacher_checkpoint)
    if not os.path.exists(teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at {teacher_checkpoint_path}")

    teacher = load_teacher_model(teacher_checkpoint_path, device)

    # Setup W&B
    setup_wandb(args, config, temperature, alpha)

    # Get training hyperparameters (CLI args override config)
    epochs = args.epochs if args.epochs else config['training']['epochs']
    batch_size = args.batch_size if args.batch_size else config['training']['batch_size']
    learning_rate = args.lr if args.lr else config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)

    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")

    # Create dataloaders (apply CLI batch_size override to config)
    print("\nLoading datasets...")
    config['training']['batch_size'] = batch_size
    dataloaders = build_dataloaders(config, splits=['train', 'val'])
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize student model
    print("\nInitializing student model...")
    use_pretrained = not args.no_pretrained
    student = StudentModel(num_classes=2, pretrained=use_pretrained)
    student = student.to(device)
    print(f"Total parameters: {student.get_num_total_params():,}")
    print(f"Trainable parameters: {student.get_num_trainable_params():,}")
    print(f"Pretrained weights: {'Yes' if use_pretrained else 'No'}")

    # Loss functions
    kd_criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    ce_criterion = nn.CrossEntropyLoss()  # For validation

    # Optimizer
    optimizer_name = config['training'].get('optimizer', 'adam').lower()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(student.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler_name = config['training'].get('scheduler', 'cosine').lower()
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        resume_path = resolve_checkpoint_path(args.resume)
        if os.path.exists(resume_path):
            start_epoch, best_val_acc, resume_temp, resume_alpha = load_checkpoint(
                resume_path, student, optimizer, scheduler
            )
            start_epoch += 1  # Start from next epoch
            # Verify hyperparameters match
            if resume_temp != temperature or resume_alpha != alpha:
                print(f"WARNING: Resumed checkpoint has different hyperparameters!")
                print(f"  Checkpoint: T={resume_temp}, α={resume_alpha}")
                print(f"  Current: T={temperature}, α={alpha}")
        else:
            print(f"Warning: Resume checkpoint not found at {resume_path}, starting from scratch")

    # Setup checkpoint directory
    checkpoint_dir = resolve_checkpoint_path(args.output_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    print("\n" + "="*60)
    print("Starting knowledge distillation training...")
    print("="*60 + "\n")

    patience = 10
    patience_counter = 0

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 40)

        # Train with distillation
        train_metrics = train_one_epoch_kd(
            teacher, student, train_loader, kd_criterion, optimizer, device, epoch + 1
        )

        # Validate
        val_metrics = validate_kd(teacher, student, val_loader, kd_criterion, ce_criterion, device)

        # Combine metrics
        metrics = {**train_metrics, **val_metrics}

        # Log to W&B
        wandb.log(metrics)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_metrics['train/loss']:.4f} "
              f"(KD: {train_metrics['train/kd_loss']:.4f}, Hard: {train_metrics['train/hard_loss']:.4f})")
        print(f"  Train Acc: {train_metrics['train/accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['val/loss']:.4f} "
              f"(KD: {val_metrics['val/kd_loss']:.4f}, Hard: {val_metrics['val/hard_loss']:.4f})")
        print(f"  Val Acc: {val_metrics['val/accuracy']:.2f}%")
        print(f"  Val Precision: {val_metrics['val/precision']:.4f}, "
              f"Val Recall: {val_metrics['val/recall']:.4f}, Val F1: {val_metrics['val/f1']:.4f}")

        # Update learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({'learning_rate': current_lr})
            print(f"  Learning rate: {current_lr:.6f}")

        # Save best model
        if val_metrics['val/accuracy'] > best_val_acc:
            best_val_acc = val_metrics['val/accuracy']
            best_checkpoint_path = os.path.join(
                checkpoint_dir, f'best_model_t{temperature}_a{alpha}.pth'
            )
            save_checkpoint(
                student, optimizer, scheduler, epoch, best_val_acc,
                best_checkpoint_path, temperature, alpha, teacher_checkpoint_path, metrics
            )
            print(f"  New best validation accuracy: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save latest checkpoint
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        save_checkpoint(
            student, optimizer, scheduler, epoch, best_val_acc,
            latest_checkpoint_path, temperature, alpha, teacher_checkpoint_path, metrics
        )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            break

    print("\n" + "="*60)
    print("Knowledge distillation training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {checkpoint_dir}")
    print(f"Best model: best_model_t{temperature}_a{alpha}.pth")
    print("="*60)

    wandb.finish()


if __name__ == "__main__":
    main()
