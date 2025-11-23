"""Tests for knowledge distillation training script."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from train_student_kd import (
    parse_args,
    load_config,
    setup_device,
    load_teacher_model,
    save_checkpoint,
    load_checkpoint,
    train_one_epoch_kd,
    validate_kd,
)
from models.teacher import TeacherModel
from models.student import StudentModel
from models.distillation import DistillationLoss


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_teacher_checkpoint_required(self):
        """Test that --teacher-checkpoint is required."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_default_args_with_teacher(self):
        """Test parsing with default arguments (teacher provided)."""
        args = parse_args(["--teacher-checkpoint", "checkpoints/teacher/best_model.pth"])
        assert args.config == "configs/config.yaml"
        assert args.teacher_checkpoint == "checkpoints/teacher/best_model.pth"
        assert args.output_dir == "checkpoints/student_kd"
        assert args.resume is None
        assert args.epochs is None
        assert args.batch_size is None
        assert args.lr is None
        assert args.temperature is None
        assert args.alpha is None
        assert args.wandb_project == "coral-bleaching"
        assert args.wandb_mode == "online"
        assert args.device is None
        assert args.no_pretrained is False

    def test_custom_teacher_checkpoint(self):
        """Test parsing with custom teacher checkpoint path."""
        args = parse_args(["--teacher-checkpoint", "/path/to/teacher.pth"])
        assert args.teacher_checkpoint == "/path/to/teacher.pth"

    def test_custom_output_dir(self):
        """Test parsing with custom output directory."""
        args = parse_args([
            "--teacher-checkpoint", "teacher.pth",
            "--output-dir", "my_kd_checkpoints"
        ])
        assert args.output_dir == "my_kd_checkpoints"

    def test_hyperparameter_overrides(self):
        """Test parsing with hyperparameter overrides."""
        args = parse_args([
            "--teacher-checkpoint", "teacher.pth",
            "--epochs", "100",
            "--batch-size", "64",
            "--lr", "0.0001",
            "--temperature", "8.0",
            "--alpha", "0.5"
        ])
        assert args.epochs == 100
        assert args.batch_size == 64
        assert args.lr == 0.0001
        assert args.temperature == 8.0
        assert args.alpha == 0.5

    def test_wandb_config(self):
        """Test parsing with W&B configuration."""
        args = parse_args([
            "--teacher-checkpoint", "teacher.pth",
            "--wandb-project", "my-kd-project",
            "--wandb-mode", "offline"
        ])
        assert args.wandb_project == "my-kd-project"
        assert args.wandb_mode == "offline"

    def test_device_override(self):
        """Test parsing with device override."""
        args = parse_args(["--teacher-checkpoint", "teacher.pth", "--device", "cpu"])
        assert args.device == "cpu"

    def test_no_pretrained_flag(self):
        """Test parsing with no-pretrained flag."""
        args = parse_args(["--teacher-checkpoint", "teacher.pth", "--no-pretrained"])
        assert args.no_pretrained is True


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_default_config(self):
        """Test loading default config file."""
        config = load_config("configs/config.yaml")
        assert isinstance(config, dict)
        assert 'dataset' in config
        assert 'model' in config
        assert 'training' in config

    def test_config_has_distillation_fields(self):
        """Test that config has distillation fields."""
        config = load_config("configs/config.yaml")

        # Distillation fields
        assert 'model' in config
        assert 'distillation' in config['model']
        assert 'temperature' in config['model']['distillation']
        assert 'alpha' in config['model']['distillation']


class TestDeviceSetup:
    """Test device setup."""

    def test_setup_device_cpu_from_args(self):
        """Test device setup with CPU from args."""
        config = {'training': {'device': 'cuda'}}
        args = parse_args(["--teacher-checkpoint", "teacher.pth", "--device", "cpu"])

        device = setup_device(config, args)

        assert device.type == 'cpu'

    def test_setup_device_cpu_from_config(self):
        """Test device setup with CPU from config."""
        config = {'training': {'device': 'cpu'}}
        args = parse_args(["--teacher-checkpoint", "teacher.pth"])

        device = setup_device(config, args)

        assert device.type == 'cpu'


class TestTeacherLoading:
    """Test teacher model loading."""

    @pytest.fixture
    def teacher_checkpoint(self):
        """Create a temporary teacher checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'teacher.pth')

            # Create and save a teacher model
            teacher = TeacherModel(num_classes=2, pretrained=False)
            checkpoint = {
                'epoch': 10,
                'model_state_dict': teacher.state_dict(),
                'best_val_acc': 85.0
            }
            torch.save(checkpoint, checkpoint_path)

            yield checkpoint_path

    def test_load_teacher_model(self, teacher_checkpoint):
        """Test loading teacher model from checkpoint."""
        device = torch.device('cpu')
        teacher = load_teacher_model(teacher_checkpoint, device)

        # Verify model is loaded
        assert isinstance(teacher, TeacherModel)
        assert next(teacher.parameters()).device.type == 'cpu'

    def test_teacher_is_frozen(self, teacher_checkpoint):
        """Test that loaded teacher has all parameters frozen."""
        device = torch.device('cpu')
        teacher = load_teacher_model(teacher_checkpoint, device)

        # Verify all parameters have requires_grad=False
        for param in teacher.parameters():
            assert param.requires_grad is False

    def test_teacher_in_eval_mode(self, teacher_checkpoint):
        """Test that loaded teacher is in eval mode."""
        device = torch.device('cpu')
        teacher = load_teacher_model(teacher_checkpoint, device)

        # Verify model is in eval mode
        assert not teacher.training

    def test_load_teacher_preserves_weights(self, teacher_checkpoint):
        """Test that loading teacher preserves original weights."""
        device = torch.device('cpu')

        # Load original checkpoint
        original_checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
        original_weights = original_checkpoint['model_state_dict']

        # Load teacher via function
        teacher = load_teacher_model(teacher_checkpoint, device)

        # Compare first layer weights
        teacher_weights = teacher.state_dict()
        first_key = list(original_weights.keys())[0]
        assert torch.allclose(original_weights[first_key], teacher_weights[first_key])


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading with KD metadata."""

    @pytest.fixture
    def student_model(self):
        """Create a student model for testing."""
        return StudentModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def optimizer(self, student_model):
        """Create an optimizer for testing."""
        return torch.optim.Adam(student_model.parameters(), lr=0.001)

    @pytest.fixture
    def scheduler(self, optimizer):
        """Create a scheduler for testing."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    def test_save_checkpoint_with_kd_metadata(self, student_model, optimizer, scheduler):
        """Test saving checkpoint with KD-specific metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_kd_checkpoint.pth')

            metrics = {'val/accuracy': 85.5, 'val/kd_loss': 0.3, 'val/hard_loss': 0.1}
            save_checkpoint(
                student_model, optimizer, scheduler,
                epoch=5, best_val_acc=85.5,
                checkpoint_path=checkpoint_path,
                temperature=4.0, alpha=0.7,
                teacher_checkpoint="checkpoints/teacher/best_model.pth",
                metrics=metrics
            )

            # Verify checkpoint file was created
            assert os.path.exists(checkpoint_path)

            # Load and verify contents
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert checkpoint['epoch'] == 5
            assert checkpoint['best_val_acc'] == 85.5
            assert checkpoint['temperature'] == 4.0
            assert checkpoint['alpha'] == 0.7
            assert checkpoint['teacher_checkpoint'] == "checkpoints/teacher/best_model.pth"
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'scheduler_state_dict' in checkpoint
            assert checkpoint['metrics'] == metrics

    def test_load_checkpoint_with_kd_metadata(self, student_model, optimizer, scheduler):
        """Test loading checkpoint with KD metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_kd_checkpoint.pth')

            # Save a checkpoint first
            save_checkpoint(
                student_model, optimizer, scheduler,
                epoch=5, best_val_acc=90.0,
                checkpoint_path=checkpoint_path,
                temperature=8.0, alpha=0.5,
                teacher_checkpoint="teacher.pth"
            )

            # Create new model and optimizer to load into
            new_model = StudentModel(num_classes=2, pretrained=False)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

            # Load checkpoint
            epoch, best_val_acc, temperature, alpha = load_checkpoint(
                checkpoint_path, new_model, new_optimizer, new_scheduler
            )

            assert epoch == 5
            assert best_val_acc == 90.0
            assert temperature == 8.0
            assert alpha == 0.5


class TestTrainingFunctions:
    """Test KD training and validation functions."""

    @pytest.fixture
    def teacher_model(self):
        """Create a frozen teacher model."""
        teacher = TeacherModel(num_classes=2, pretrained=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    @pytest.fixture
    def student_model(self):
        """Create a student model."""
        return StudentModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def kd_criterion(self):
        """Create a distillation loss."""
        return DistillationLoss(temperature=4.0, alpha=0.7)

    @pytest.fixture
    def ce_criterion(self):
        """Create a CE loss."""
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def optimizer(self, student_model):
        """Create an optimizer."""
        return torch.optim.Adam(student_model.parameters(), lr=0.001)

    @pytest.fixture
    def dummy_dataloader(self):
        """Create a dummy dataloader with 2 batches."""
        from torch.utils.data import TensorDataset, DataLoader

        # Create dummy data: 8 samples, 3 channels, 32x32 images
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 2, (8,))

        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=4, shuffle=False)

    def test_train_one_epoch_kd_returns_metrics(
        self, teacher_model, student_model, dummy_dataloader, kd_criterion, optimizer
    ):
        """Test that train_one_epoch_kd returns expected metrics."""
        device = torch.device('cpu')

        metrics = train_one_epoch_kd(
            teacher_model, student_model, dummy_dataloader,
            kd_criterion, optimizer, device, epoch=1
        )

        # Verify all expected metrics are present
        assert 'train/loss' in metrics
        assert 'train/kd_loss' in metrics
        assert 'train/hard_loss' in metrics
        assert 'train/accuracy' in metrics
        assert 'epoch' in metrics

        # Verify metrics are reasonable
        assert metrics['train/loss'] > 0
        assert metrics['train/kd_loss'] > 0
        assert metrics['train/hard_loss'] > 0
        assert 0 <= metrics['train/accuracy'] <= 100
        assert metrics['epoch'] == 1

    def test_train_one_epoch_kd_teacher_unchanged(
        self, teacher_model, student_model, dummy_dataloader, kd_criterion, optimizer
    ):
        """Test that teacher parameters don't change during training."""
        device = torch.device('cpu')

        # Save teacher weights before training
        teacher_weights_before = {
            name: param.clone()
            for name, param in teacher_model.named_parameters()
        }

        # Train one epoch
        train_one_epoch_kd(
            teacher_model, student_model, dummy_dataloader,
            kd_criterion, optimizer, device, epoch=1
        )

        # Verify teacher weights are unchanged
        for name, param in teacher_model.named_parameters():
            assert torch.allclose(param, teacher_weights_before[name])

    def test_validate_kd_returns_all_losses(
        self, teacher_model, student_model, dummy_dataloader, kd_criterion, ce_criterion
    ):
        """Test that validate_kd returns both CE and KD losses."""
        device = torch.device('cpu')

        metrics = validate_kd(
            teacher_model, student_model, dummy_dataloader,
            kd_criterion, ce_criterion, device
        )

        # Verify all expected metrics are present
        assert 'val/loss' in metrics  # CE loss
        assert 'val/kd_loss' in metrics
        assert 'val/hard_loss' in metrics
        assert 'val/accuracy' in metrics
        assert 'val/precision' in metrics
        assert 'val/recall' in metrics
        assert 'val/f1' in metrics

        # Verify metrics are reasonable
        assert metrics['val/loss'] > 0
        assert metrics['val/kd_loss'] > 0
        assert metrics['val/hard_loss'] > 0
        assert 0 <= metrics['val/accuracy'] <= 100

    def test_validate_kd_no_gradient(
        self, teacher_model, student_model, dummy_dataloader, kd_criterion, ce_criterion
    ):
        """Test that validate_kd doesn't compute gradients."""
        device = torch.device('cpu')

        # Save student weights before validation
        student_weights_before = {
            name: param.clone()
            for name, param in student_model.named_parameters()
        }

        # Run validation
        validate_kd(
            teacher_model, student_model, dummy_dataloader,
            kd_criterion, ce_criterion, device
        )

        # Verify student weights are unchanged
        for name, param in student_model.named_parameters():
            assert torch.allclose(param, student_weights_before[name])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
