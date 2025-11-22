"""Tests for student baseline training script."""

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from train_student_baseline import (
    parse_args,
    load_config,
    setup_device,
    save_checkpoint,
    load_checkpoint,
    train_one_epoch,
    validate,
)
from models.student import StudentModel
from utils.data_loader import build_dataloaders
from torch.utils.data import DataLoader


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_default_args(self):
        """Test parsing with default arguments."""
        args = parse_args([])
        assert args.config == "configs/config.yaml"
        assert args.output_dir == "checkpoints/student_baseline"
        assert args.resume is None
        assert args.epochs is None
        assert args.batch_size is None
        assert args.lr is None
        assert args.wandb_project == "coral-bleaching"
        assert args.wandb_mode == "online"
        assert args.device is None
        assert args.no_pretrained is False

    def test_custom_config(self):
        """Test parsing with custom config path."""
        args = parse_args(["--config", "custom_config.yaml"])
        assert args.config == "custom_config.yaml"

    def test_custom_output_dir(self):
        """Test parsing with custom output directory."""
        args = parse_args(["--output-dir", "my_checkpoints"])
        assert args.output_dir == "my_checkpoints"

    def test_resume_checkpoint(self):
        """Test parsing with resume checkpoint."""
        args = parse_args(["--resume", "checkpoints/model.pth"])
        assert args.resume == "checkpoints/model.pth"

    def test_hyperparameter_overrides(self):
        """Test parsing with hyperparameter overrides."""
        args = parse_args([
            "--epochs", "100",
            "--batch-size", "64",
            "--lr", "0.0001"
        ])
        assert args.epochs == 100
        assert args.batch_size == 64
        assert args.lr == 0.0001

    def test_wandb_config(self):
        """Test parsing with W&B configuration."""
        args = parse_args([
            "--wandb-project", "my-project",
            "--wandb-mode", "offline"
        ])
        assert args.wandb_project == "my-project"
        assert args.wandb_mode == "offline"

    def test_device_override(self):
        """Test parsing with device override."""
        args = parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_no_pretrained_flag(self):
        """Test parsing with no-pretrained flag."""
        args = parse_args(["--no-pretrained"])
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

    def test_config_has_required_fields(self):
        """Test that config has all required fields."""
        config = load_config("configs/config.yaml")

        # Dataset fields
        assert 'dataset' in config
        assert 'raw_dir' in config['dataset']
        assert 'splits_dir' in config['dataset']

        # Model fields
        assert 'model' in config
        assert 'student' in config['model']

        # Training fields
        assert 'training' in config
        assert 'epochs' in config['training']
        assert 'batch_size' in config['training']
        assert 'learning_rate' in config['training']

    def test_load_custom_config(self):
        """Test loading custom config file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'dataset': {'raw_dir': 'data/raw'},
                'model': {'student': {'name': 'mobilenet_v3_small'}},
                'training': {'epochs': 10, 'batch_size': 16, 'learning_rate': 0.001}
            }
            yaml.dump(config_data, f)
            temp_config_path = f.name

        try:
            config = load_config(temp_config_path)
            assert config['training']['epochs'] == 10
            assert config['training']['batch_size'] == 16
        finally:
            os.unlink(temp_config_path)


class TestDeviceSetup:
    """Test device setup."""

    def test_setup_device_cpu_from_args(self):
        """Test device setup with CPU from args."""
        config = {'training': {'device': 'cuda'}}
        args = parse_args(["--device", "cpu"])

        device = setup_device(config, args)

        assert device.type == 'cpu'

    def test_setup_device_cpu_from_config(self):
        """Test device setup with CPU from config."""
        config = {'training': {'device': 'cpu'}}
        args = parse_args([])

        device = setup_device(config, args)

        assert device.type == 'cpu'

    def test_setup_device_cuda_fallback_to_cpu(self):
        """Test CUDA device falls back to CPU if unavailable."""
        config = {'training': {'device': 'cuda'}}
        args = parse_args([])

        device = setup_device(config, args)

        # Should be cuda if available, cpu otherwise
        assert device.type in ['cpu', 'cuda']

    def test_setup_device_args_override_config(self):
        """Test that args override config for device."""
        config = {'training': {'device': 'cuda'}}
        args = parse_args(["--device", "cpu"])

        device = setup_device(config, args)

        assert device.type == 'cpu'


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading."""

    @pytest.fixture
    def model(self):
        """Create a simple student model for testing."""
        return StudentModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def optimizer(self, model):
        """Create an optimizer for testing."""
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @pytest.fixture
    def scheduler(self, optimizer):
        """Create a scheduler for testing."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    def test_save_checkpoint(self, model, optimizer, scheduler):
        """Test saving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

            metrics = {'val/accuracy': 85.5, 'val/loss': 0.4}
            save_checkpoint(model, optimizer, scheduler, epoch=5, best_val_acc=85.5,
                          checkpoint_path=checkpoint_path, metrics=metrics)

            # Verify checkpoint file was created
            assert os.path.exists(checkpoint_path)

            # Load and verify contents
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            assert 'epoch' in checkpoint
            assert checkpoint['epoch'] == 5
            assert 'best_val_acc' in checkpoint
            assert checkpoint['best_val_acc'] == 85.5
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert 'scheduler_state_dict' in checkpoint
            assert 'metrics' in checkpoint
            assert checkpoint['metrics'] == metrics

    def test_load_checkpoint(self, model, optimizer, scheduler):
        """Test loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

            # Save a checkpoint first
            save_checkpoint(model, optimizer, scheduler, epoch=5, best_val_acc=90.0,
                          checkpoint_path=checkpoint_path)

            # Create new model and optimizer to load into
            new_model = StudentModel(num_classes=2, pretrained=False)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

            # Load checkpoint
            epoch, best_val_acc = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler)

            assert epoch == 5
            assert best_val_acc == 90.0

            # Verify model weights were loaded (compare a specific layer's weights)
            original_weights = model.backbone.features[0][0].weight.data
            loaded_weights = new_model.backbone.features[0][0].weight.data
            assert torch.allclose(original_weights, loaded_weights)

    def test_save_load_round_trip(self, model, optimizer, scheduler):
        """Test save and load round trip preserves model state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

            # Get initial model state
            initial_state = model.state_dict()

            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch=10, best_val_acc=88.5,
                          checkpoint_path=checkpoint_path)

            # Create new model and load checkpoint
            new_model = StudentModel(num_classes=2, pretrained=False)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

            load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler)

            # Verify all parameters match
            for (name1, param1), (name2, param2) in zip(initial_state.items(), new_model.state_dict().items()):
                assert name1 == name2
                assert torch.allclose(param1, param2)

    def test_load_checkpoint_without_optimizer_scheduler(self, model):
        """Test loading checkpoint without optimizer and scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pth')

            # Save with optimizer and scheduler
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            save_checkpoint(model, optimizer, scheduler, epoch=3, best_val_acc=75.0,
                          checkpoint_path=checkpoint_path)

            # Load without optimizer and scheduler
            new_model = StudentModel(num_classes=2, pretrained=False)
            epoch, best_val_acc = load_checkpoint(checkpoint_path, new_model)

            assert epoch == 3
            assert best_val_acc == 75.0


class TestTrainingOneEpoch:
    """Test training for one epoch with real components."""

    @pytest.fixture
    def model(self):
        """Create a student model for testing."""
        return StudentModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def train_loader(self):
        """Create a simple dataloader with synthetic data."""
        # Create synthetic dataset
        images = torch.randn(16, 3, 224, 224)  # 16 samples
        labels = torch.randint(0, 2, (16,))    # Binary labels
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    def test_train_one_epoch_runs(self, model, train_loader):
        """Test that training one epoch completes successfully."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')

        metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1)

        # Check that metrics are returned
        assert 'train/loss' in metrics
        assert 'train/accuracy' in metrics
        assert 'epoch' in metrics

        # Check that metrics are valid
        assert metrics['train/loss'] > 0
        assert 0 <= metrics['train/accuracy'] <= 100
        assert metrics['epoch'] == 1

    def test_train_one_epoch_updates_weights(self, model, train_loader):
        """Test that training actually updates model weights."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')

        # Get initial weights (from classifier)
        initial_weights = model.backbone.classifier[-1].weight.data.clone()

        # Train for one epoch
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1)

        # Get updated weights
        updated_weights = model.backbone.classifier[-1].weight.data

        # Weights should have changed
        assert not torch.allclose(initial_weights, updated_weights)


class TestValidation:
    """Test validation function with real components."""

    @pytest.fixture
    def model(self):
        """Create a student model for testing."""
        return StudentModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def val_loader(self):
        """Create a simple validation dataloader with synthetic data."""
        images = torch.randn(12, 3, 224, 224)  # 12 samples
        labels = torch.randint(0, 2, (12,))    # Binary labels
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    def test_validate_runs(self, model, val_loader):
        """Test that validation completes successfully."""
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')

        model.eval()
        metrics = validate(model, val_loader, criterion, device)

        # Check that metrics are returned
        assert 'val/loss' in metrics
        assert 'val/accuracy' in metrics
        assert 'val/precision' in metrics
        assert 'val/recall' in metrics
        assert 'val/f1' in metrics

        # Check that metrics are valid
        assert metrics['val/loss'] > 0
        assert 0 <= metrics['val/accuracy'] <= 100
        assert 0 <= metrics['val/precision'] <= 1
        assert 0 <= metrics['val/recall'] <= 1
        assert 0 <= metrics['val/f1'] <= 1


class TestTrainingIntegration:
    """Integration tests for full training workflow."""

    def test_train_and_validate_step(self):
        """Test a complete training and validation step."""
        # Create model, criterion, optimizer
        model = StudentModel(num_classes=2, pretrained=False)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device('cpu')

        # Create synthetic dataloaders
        train_images = torch.randn(16, 3, 224, 224)
        train_labels = torch.randint(0, 2, (16,))
        train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

        val_images = torch.randn(8, 3, 224, 224)
        val_labels = torch.randint(0, 2, (8,))
        val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Check that both return valid metrics
        assert train_metrics['train/loss'] > 0
        assert val_metrics['val/loss'] > 0
        assert 0 <= train_metrics['train/accuracy'] <= 100
        assert 0 <= val_metrics['val/accuracy'] <= 100

    def test_checkpoint_save_and_resume(self):
        """Test saving a checkpoint and resuming training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'resume_test.pth')

            # Create and train model for 1 step
            model = StudentModel(num_classes=2, pretrained=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch=5, best_val_acc=80.0,
                          checkpoint_path=checkpoint_path)

            # Create new model and resume
            new_model = StudentModel(num_classes=2, pretrained=False)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
            new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

            start_epoch, best_val_acc = load_checkpoint(checkpoint_path, new_model, new_optimizer, new_scheduler)

            # Verify resume state
            assert start_epoch == 5
            assert best_val_acc == 80.0

            # Verify model can continue training
            criterion = nn.CrossEntropyLoss()
            device = torch.device('cpu')

            train_images = torch.randn(8, 3, 224, 224)
            train_labels = torch.randint(0, 2, (8,))
            train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

            # This should not raise any errors
            metrics = train_one_epoch(new_model, train_loader, criterion, new_optimizer, device, epoch=6)
            assert metrics is not None


class TestBuildDataloadersIntegration:
    """Integration tests for build_dataloaders usage in training script."""

    @pytest.fixture
    def temp_config_path(self):
        """Create a temporary config file for testing."""
        config = {
            'dataset': {
                'splits_dir': 'data/splits',
                'train_split': 'train',
                'val_split': 'val',
                'test_split': 'test',
                'image_size': [224, 224],
            },
            'augmentations': {
                'random_flip': True,
                'color_jitter': {
                    'enabled': True,
                    'brightness': 0.1,
                    'contrast': 0.1,
                    'saturation': 0.1,
                    'hue': 0.05,
                },
                'normalization': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225],
                },
            },
            'training': {
                'batch_size': 16,
            },
            'dataloader': {
                'num_workers': 0,
                'pin_memory': False,
            },
            'model': {
                'student': {
                    'num_classes': 2,
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_build_dataloaders_returns_dict_of_loaders(self, temp_config_path):
        """Test that build_dataloaders returns a dict, not a single loader."""
        config = load_config(temp_config_path)

        dataloaders = build_dataloaders(config, splits=['train', 'val'])

        # Should return a dictionary
        assert isinstance(dataloaders, dict), "build_dataloaders should return a dict"
        assert 'train' in dataloaders, "Dict should contain 'train' key"
        assert 'val' in dataloaders, "Dict should contain 'val' key"

        # Each value should be a DataLoader
        assert isinstance(dataloaders['train'], DataLoader), "'train' value should be a DataLoader"
        assert isinstance(dataloaders['val'], DataLoader), "'val' value should be a DataLoader"

    def test_build_dataloaders_with_splits_parameter(self, temp_config_path):
        """Test build_dataloaders with correct 'splits' parameter name."""
        config = load_config(temp_config_path)

        # This should not raise TypeError with correct parameter name
        dataloaders = build_dataloaders(config, splits=['train'])

        assert 'train' in dataloaders
        assert isinstance(dataloaders['train'], DataLoader)

    def test_dataloaders_have_correct_batch_size(self, temp_config_path):
        """Test that dataloaders use batch_size from config."""
        config = load_config(temp_config_path)
        expected_batch_size = config['training']['batch_size']

        dataloaders = build_dataloaders(config, splits=['train'])
        train_loader = dataloaders['train']

        # Get actual batch size from first batch
        for images, labels in train_loader:
            actual_batch_size = images.shape[0]
            # Should match config or be smaller if dataset is small
            assert actual_batch_size <= expected_batch_size, \
                f"Batch size {actual_batch_size} exceeds config batch size {expected_batch_size}"
            assert actual_batch_size > 0, "Batch size should be positive"
            break

    def test_main_with_correct_dataloader_usage(self, temp_config_path):
        """Integration test showing correct usage pattern for main()."""
        config = load_config(temp_config_path)
        batch_size = 8

        # Simulate what main() does:
        # 1. Apply CLI batch_size override to config
        config['training']['batch_size'] = batch_size

        # 2. Call build_dataloaders with correct parameter name
        dataloaders = build_dataloaders(config, splits=['train', 'val'])

        # 3. Unpack returned dict
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        # Verify loaders are usable
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

        # Verify batch sizes are correct
        for images, labels in train_loader:
            assert images.shape[0] <= batch_size, "Train batch size should match config"
            assert labels.shape[0] == images.shape[0], "Labels should match images"
            break

        for images, labels in val_loader:
            assert images.shape[0] <= batch_size, "Val batch size should match config"
            assert labels.shape[0] == images.shape[0], "Labels should match images"
            break
