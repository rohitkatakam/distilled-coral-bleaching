"""Tests for evaluation script."""

import os
import sys
import tempfile
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from evaluate import (
    load_checkpoint,
    load_model,
    count_parameters,
    run_inference,
    compute_metrics,
    save_results
)

from models.teacher import TeacherModel
from models.student import StudentModel


class TestLoadCheckpoint:
    """Test checkpoint loading."""

    def test_load_checkpoint_success(self):
        """Test loading a valid checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test.pth')

            # Create a dummy checkpoint
            checkpoint_data = {
                'epoch': 10,
                'model_state_dict': {},
                'best_val_acc': 0.85
            }
            torch.save(checkpoint_data, checkpoint_path)

            # Load checkpoint
            checkpoint = load_checkpoint(checkpoint_path)

            assert checkpoint['epoch'] == 10
            assert checkpoint['best_val_acc'] == 0.85

    def test_load_checkpoint_file_not_found(self):
        """Test loading from non-existent path raises error."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint('nonexistent/path/checkpoint.pth')


class TestLoadModel:
    """Test model loading from checkpoint."""

    @pytest.fixture
    def temp_config(self):
        """Create temporary config for testing."""
        return {
            'model': {
                'num_classes': 2,
                'dropout': None
            }
        }

    def test_load_teacher_model(self, temp_config):
        """Test loading teacher model from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'teacher.pth')

            # Create teacher model and save checkpoint
            model = TeacherModel(num_classes=2, pretrained=False)
            checkpoint = {
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'best_val_acc': 0.80
            }
            torch.save(checkpoint, checkpoint_path)

            # Load model
            loaded_model, loaded_checkpoint = load_model(
                checkpoint_path, temp_config, 'teacher', torch.device('cpu')
            )

            assert isinstance(loaded_model, TeacherModel)
            assert loaded_checkpoint['epoch'] == 5
            assert loaded_checkpoint['best_val_acc'] == 0.80

    def test_load_student_model(self, temp_config):
        """Test loading student model from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'student.pth')

            # Create student model and save checkpoint
            model = StudentModel(num_classes=2, pretrained=False)
            checkpoint = {
                'epoch': 8,
                'model_state_dict': model.state_dict(),
                'best_val_acc': 0.75
            }
            torch.save(checkpoint, checkpoint_path)

            # Load model
            loaded_model, loaded_checkpoint = load_model(
                checkpoint_path, temp_config, 'student', torch.device('cpu')
            )

            assert isinstance(loaded_model, StudentModel)
            assert loaded_checkpoint['epoch'] == 8
            assert loaded_checkpoint['best_val_acc'] == 0.75

    def test_load_model_invalid_type(self, temp_config):
        """Test loading with invalid model type raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'model.pth')

            model = TeacherModel(num_classes=2, pretrained=False)
            checkpoint = {
                'epoch': 1,
                'model_state_dict': model.state_dict(),
                'best_val_acc': 0.5
            }
            torch.save(checkpoint, checkpoint_path)

            with pytest.raises(ValueError, match="Unknown model type"):
                load_model(checkpoint_path, temp_config, 'invalid_type', torch.device('cpu'))


class TestCountParameters:
    """Test parameter counting."""

    def test_count_teacher_parameters(self):
        """Test counting parameters in teacher model."""
        model = TeacherModel(num_classes=2, pretrained=False)
        total, trainable = count_parameters(model)

        # ResNet50 should have ~23-25M parameters
        assert 23_000_000 < total < 26_000_000
        assert total == trainable  # All parameters trainable by default

    def test_count_student_parameters(self):
        """Test counting parameters in student model."""
        model = StudentModel(num_classes=2, pretrained=False)
        total, trainable = count_parameters(model)

        # MobileNetV3-Small should have ~1.5-2.5M parameters
        assert 1_500_000 < total < 2_500_000
        assert total == trainable

    def test_count_frozen_model_parameters(self):
        """Test counting parameters when some are frozen."""
        model = TeacherModel(num_classes=2, pretrained=False)
        model.freeze_backbone()

        total, trainable = count_parameters(model)

        # Total should be unchanged
        assert 23_000_000 < total < 26_000_000
        # Only FC layer should be trainable (~4000 params)
        assert 4000 < trainable < 5000


class TestRunInference:
    """Test inference on dataloader."""

    @pytest.fixture
    def model(self):
        """Create a simple model for testing."""
        return TeacherModel(num_classes=2, pretrained=False)

    @pytest.fixture
    def dataloader(self):
        """Create a simple dataloader with synthetic data."""
        images = torch.randn(12, 3, 224, 224)
        labels = torch.randint(0, 2, (12,))
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=4)

    def test_run_inference(self, model, dataloader):
        """Test running inference on dataloader."""
        model.eval()
        preds, labels, probs = run_inference(model, dataloader, torch.device('cpu'))

        # Check shapes
        assert len(preds) == 12
        assert len(labels) == 12
        assert len(probs) == 12
        assert probs.shape == (12, 2)  # 12 samples, 2 classes

        # Check value ranges
        assert all((preds >= 0) & (preds < 2))
        assert all((labels >= 0) & (labels < 2))
        assert all((probs >= 0).all(axis=1) & (probs <= 1).all(axis=1))

        # Check probabilities sum to 1
        prob_sums = probs.sum(axis=1)
        assert all(abs(prob_sums - 1.0) < 1e-5)


class TestComputeMetrics:
    """Test metrics computation."""

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        preds = [0, 1, 0, 1, 0, 1]
        labels = [0, 1, 0, 1, 0, 1]
        probs = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]

        import numpy as np
        preds = np.array(preds)
        labels = np.array(labels)
        probs = np.array(probs)

        metrics = compute_metrics(preds, labels, probs)

        # Perfect predictions should have accuracy = 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0

        # Check confusion matrix
        cm = metrics['confusion_matrix']
        assert cm[0][0] == 3  # True negatives
        assert cm[0][1] == 0  # False positives
        assert cm[1][0] == 0  # False negatives
        assert cm[1][1] == 3  # True positives

    def test_compute_metrics_has_required_fields(self):
        """Test that metrics dict has all required fields."""
        import numpy as np
        preds = np.array([0, 1, 0, 1])
        labels = np.array([0, 0, 1, 1])
        probs = np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]])

        metrics = compute_metrics(preds, labels, probs)

        # Check required fields
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'confusion_matrix' in metrics
        assert 'per_class_metrics' in metrics

        # Check per-class metrics
        assert 'healthy' in metrics['per_class_metrics']
        assert 'bleached' in metrics['per_class_metrics']

        for class_name in ['healthy', 'bleached']:
            class_metrics = metrics['per_class_metrics'][class_name]
            assert 'precision' in class_metrics
            assert 'recall' in class_metrics
            assert 'f1' in class_metrics
            assert 'support' in class_metrics


class TestSaveResults:
    """Test saving results to JSON."""

    def test_save_results(self):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'results', 'test_results.json')

            results = {
                'model': {'type': 'teacher'},
                'metrics': {'accuracy': 0.85},
                'timestamp': '2024-01-01T00:00:00'
            }

            save_results(results, output_path)

            # Verify file was created
            assert os.path.exists(output_path)

            # Verify contents
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)

            assert loaded_results['model']['type'] == 'teacher'
            assert loaded_results['metrics']['accuracy'] == 0.85

    def test_save_results_creates_directory(self):
        """Test that save_results creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nested path that doesn't exist
            output_path = os.path.join(tmpdir, 'a', 'b', 'c', 'results.json')

            results = {'test': 'data'}
            save_results(results, output_path)

            # Verify file was created (and parent dirs were created)
            assert os.path.exists(output_path)
