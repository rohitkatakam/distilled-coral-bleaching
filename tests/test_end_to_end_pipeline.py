"""
END-TO-END INTEGRATION TEST: Complete pipeline verification.

This test verifies that all utilities work together in a realistic workflow:
DataLoader → Transforms → Model → Metrics → Wandb Logging

Tests use real data, real transforms, real model inference, and real (offline) wandb logging.
"""

import pytest
import torch
import torch.nn as nn
import wandb
import yaml
import tempfile
import shutil
from pathlib import Path

from utils.data_loader import build_dataloaders
from utils.metrics import compute_classification_metrics, log_metrics_to_wandb
from utils.preprocessing import get_transforms


@pytest.fixture
def config():
    """Load real config for end-to-end testing."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


class DummyBinaryClassifier(nn.Module):
    """Simple classifier for testing."""

    def __init__(self, input_size=224*224*3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        """Forward pass."""
        x = self.flatten(x)
        return self.fc(x)


class TestEndToEndPipeline:
    """END-TO-END integration test for complete training/evaluation pipeline."""

    def test_complete_inference_pipeline(self, config):
        """
        INTEGRATION: Test complete pipeline from data loading to metrics logging.

        Pipeline flow:
        1. Load real data with DataLoader
        2. Apply real transforms
        3. Run inference with dummy model
        4. Compute metrics
        5. Log to wandb (offline)
        """
        # Reduce batch size and data for faster testing
        config_copy = config.copy()
        config_copy['training']['batch_size'] = 8
        config_copy['dataloader']['num_workers'] = 0  # Single-threaded for testing

        # Create temp directory for offline wandb
        temp_dir = tempfile.mkdtemp()

        try:
            # Initialize wandb in offline mode
            wandb.init(
                mode='offline',
                project='test-end-to-end',
                dir=temp_dir,
                reinit=True
            )

            # STEP 1: Build dataloaders with real data
            dataloaders = build_dataloaders(config_copy, splits=['val'])
            val_loader = dataloaders['val']

            # STEP 2: Create dummy model
            model = DummyBinaryClassifier()
            model.eval()

            # STEP 3: Run inference on one batch
            images, labels = next(iter(val_loader))

            assert images.shape[0] <= 8  # Batch size
            assert images.shape[1:] == (3, 224, 224)  # C, H, W
            assert labels.shape[0] <= 8

            # STEP 4: Get predictions from model
            with torch.no_grad():
                logits = model(images)
                predictions = logits.argmax(dim=1)

            # STEP 5: Compute metrics
            metrics = compute_classification_metrics(
                labels.numpy(),
                predictions.numpy(),
                class_names=['bleached', 'healthy']
            )

            # Verify metrics are computed
            assert 'accuracy' in metrics
            assert 'precision_macro' in metrics
            assert 'recall_macro' in metrics
            assert 'f1_macro' in metrics
            assert 'confusion_matrix' in metrics

            # STEP 6: Log to wandb
            log_metrics_to_wandb(metrics, step=1, prefix='val/')

            # Verify wandb logging worked
            assert wandb.run is not None

            # Clean up
            wandb.finish()

        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_multiple_batch_inference(self, config):
        """
        INTEGRATION: Test inference over multiple batches.

        Simulates mini-epoch evaluation.
        """
        # Reduce data for faster testing
        config_copy = config.copy()
        config_copy['training']['batch_size'] = 8
        config_copy['dataloader']['num_workers'] = 0

        # Build dataloader
        dataloaders = build_dataloaders(config_copy, splits=['val'])
        val_loader = dataloaders['val']

        # Create model
        model = DummyBinaryClassifier()
        model.eval()

        # Collect predictions and labels over multiple batches
        all_predictions = []
        all_labels = []

        num_batches_to_test = 3
        for i, (images, labels) in enumerate(val_loader):
            if i >= num_batches_to_test:
                break

            # Run inference
            with torch.no_grad():
                logits = model(images)
                predictions = logits.argmax(dim=1)

            all_predictions.append(predictions)
            all_labels.append(labels)

        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        # Compute metrics on aggregated results
        metrics = compute_classification_metrics(
            all_labels.numpy(),
            all_predictions.numpy()
        )

        # Verify we processed multiple batches
        assert len(all_labels) > 8  # More than one batch
        assert metrics['accuracy'] >= 0.0  # Sanity check

    def test_train_dataloader_integration(self, config):
        """
        INTEGRATION: Test that train dataloader with augmentation works end-to-end.
        """
        config_copy = config.copy()
        config_copy['training']['batch_size'] = 8
        config_copy['dataloader']['num_workers'] = 0

        # Build train dataloader (has augmentation)
        dataloaders = build_dataloaders(config_copy, splits=['train'])
        train_loader = dataloaders['train']

        # Create model
        model = DummyBinaryClassifier()
        model.eval()

        # Get two batches
        batch1_images, batch1_labels = next(iter(train_loader))
        batch2_images, batch2_labels = next(iter(train_loader))

        # Verify shapes
        assert batch1_images.shape[1:] == (3, 224, 224)
        assert batch2_images.shape[1:] == (3, 224, 224)

        # Run inference on both
        with torch.no_grad():
            pred1 = model(batch1_images).argmax(dim=1)
            pred2 = model(batch2_images).argmax(dim=1)

        # Both should produce valid predictions
        assert all(p in [0, 1] for p in pred1)
        assert all(p in [0, 1] for p in pred2)

    def test_dataloader_exhaustion(self, config):
        """
        INTEGRATION: Test that dataloader can be exhausted (full epoch).
        """
        config_copy = config.copy()
        config_copy['training']['batch_size'] = 16
        config_copy['dataloader']['num_workers'] = 0

        # Use test split (smallest, 139 samples)
        dataloaders = build_dataloaders(config_copy, splits=['test'])
        test_loader = dataloaders['test']

        # Create model
        model = DummyBinaryClassifier()
        model.eval()

        total_samples = 0
        batch_count = 0

        # Iterate through entire dataset
        for images, labels in test_loader:
            # Run inference
            with torch.no_grad():
                logits = model(images)
                predictions = logits.argmax(dim=1)

            total_samples += len(labels)
            batch_count += 1

            # Verify each batch
            assert images.shape[1:] == (3, 224, 224)
            assert all(p in [0, 1] for p in predictions)

        # Verify we processed all test samples
        assert total_samples == 139  # Known test split size
        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
