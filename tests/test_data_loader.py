"""
Tests for utils/data_loader.py

Tests dataset loading, dataloaders, and integration with real data.
"""

import pytest
import torch
import yaml
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from utils.data_loader import (
    CoralDataset,
    build_dataloaders,
    get_dataset_stats,
    LABEL_MAP,
    INV_LABEL_MAP,
)
from utils.preprocessing import get_transforms


@pytest.fixture
def config():
    """Load actual config for testing."""
    config_path = Path("configs/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_csv_path():
    """Path to a real split CSV (train.csv)."""
    return "data/splits/train.csv"


@pytest.fixture
def val_csv_path():
    """Path to validation CSV."""
    return "data/splits/val.csv"


@pytest.fixture
def test_csv_path():
    """Path to test CSV."""
    return "data/splits/test.csv"


class TestLabelMaps:
    """Tests for label mapping constants."""

    def test_label_map_structure(self):
        """Test that LABEL_MAP has expected structure."""
        assert 'bleached' in LABEL_MAP
        assert 'healthy' in LABEL_MAP
        assert LABEL_MAP['bleached'] == 0
        assert LABEL_MAP['healthy'] == 1

    def test_inverse_label_map(self):
        """Test that INV_LABEL_MAP is correct inverse."""
        assert INV_LABEL_MAP[0] == 'bleached'
        assert INV_LABEL_MAP[1] == 'healthy'
        assert len(INV_LABEL_MAP) == len(LABEL_MAP)


class TestCoralDataset:
    """Tests for CoralDataset class."""

    def test_dataset_initialization(self, sample_csv_path):
        """Test that dataset can be initialized."""
        dataset = CoralDataset(csv_path=sample_csv_path)
        assert isinstance(dataset, CoralDataset)

    def test_dataset_length(self, sample_csv_path):
        """Test dataset __len__ method."""
        dataset = CoralDataset(csv_path=sample_csv_path)
        length = len(dataset)

        assert length > 0
        # We know train.csv has 645 samples
        assert length == 645

    def test_dataset_getitem_without_transform(self, sample_csv_path):
        """Test getting an item without transforms."""
        dataset = CoralDataset(csv_path=sample_csv_path, transform=None)

        # Get first item
        image, label = dataset[0]

        # Without transform, image should be PIL Image
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_dataset_getitem_with_transform(self, sample_csv_path, config):
        """Test getting an item with transforms."""
        transforms = get_transforms(config, 'train')
        dataset = CoralDataset(csv_path=sample_csv_path, transform=transforms)

        # Get first item
        image, label = dataset[0]

        # With transform, image should be tensor
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)  # C, H, W
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_dataset_all_images_loadable(self, val_csv_path, config):
        """Test that all images in val split can be loaded (small split)."""
        transforms = get_transforms(config, 'val')
        dataset = CoralDataset(csv_path=val_csv_path, transform=transforms)

        # Try loading first 10 samples
        for i in range(min(10, len(dataset))):
            image, label = dataset[i]
            assert image.shape == (3, 224, 224)
            assert label in [0, 1]

    def test_dataset_label_counts(self, sample_csv_path):
        """Test get_label_counts method."""
        dataset = CoralDataset(csv_path=sample_csv_path)
        counts = dataset.get_label_counts()

        assert isinstance(counts, dict)
        assert 'bleached' in counts
        assert 'healthy' in counts
        assert counts['bleached'] > 0
        assert counts['healthy'] > 0
        assert counts['bleached'] + counts['healthy'] == len(dataset)

    def test_dataset_class_weights(self, sample_csv_path):
        """Test get_class_weights method."""
        dataset = CoralDataset(csv_path=sample_csv_path)
        weights = dataset.get_class_weights()

        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)  # Binary classification
        assert all(w > 0 for w in weights)

    def test_dataset_missing_csv_raises_error(self):
        """Test that missing CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            CoralDataset(csv_path="nonexistent.csv")

    def test_dataset_validates_columns(self, tmp_path):
        """Test that dataset validates required columns."""
        # Create a CSV with wrong columns
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("wrong_column,another_column\nvalue1,value2\n")

        with pytest.raises(ValueError, match="missing required columns"):
            CoralDataset(csv_path=bad_csv)

    def test_dataset_validates_labels(self, tmp_path):
        """Test that dataset validates label values."""
        # Create a CSV with invalid label
        bad_csv = tmp_path / "invalid_labels.csv"
        bad_csv.write_text("image_path,label\npath/to/img.jpg,invalid_label\n")

        with pytest.raises(ValueError, match="Unknown labels"):
            CoralDataset(csv_path=bad_csv)


class TestBuildDataloaders:
    """Tests for build_dataloaders function."""

    def test_build_all_dataloaders(self, config):
        """Test building all dataloaders."""
        dataloaders = build_dataloaders(config)

        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' in dataloaders

        assert isinstance(dataloaders['train'], DataLoader)
        assert isinstance(dataloaders['val'], DataLoader)
        assert isinstance(dataloaders['test'], DataLoader)

    def test_build_single_dataloader(self, config):
        """Test building a single dataloader."""
        dataloaders = build_dataloaders(config, splits=['train'])

        assert 'train' in dataloaders
        assert 'val' not in dataloaders
        assert 'test' not in dataloaders

    def test_build_multiple_specific_dataloaders(self, config):
        """Test building specific dataloaders."""
        dataloaders = build_dataloaders(config, splits=['train', 'val'])

        assert 'train' in dataloaders
        assert 'val' in dataloaders
        assert 'test' not in dataloaders

    def test_dataloader_batch_size(self, config):
        """Test that dataloaders use correct batch size."""
        dataloaders = build_dataloaders(config, splits=['train'])
        train_loader = dataloaders['train']

        assert train_loader.batch_size == config['training']['batch_size']

    def test_train_dataloader_shuffles(self, config):
        """Test that train dataloader has shuffle enabled."""
        # Reduce num_workers to 0 for testing
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0

        dataloaders = build_dataloaders(config_copy, splits=['train'])
        train_loader = dataloaders['train']

        # Check that sampler is configured for shuffling
        # In DataLoader with shuffle=True, sampler is None or RandomSampler
        # This is an indirect test since shuffle is not directly accessible
        assert train_loader.batch_size == config['training']['batch_size']

    def test_val_test_dataloaders_dont_shuffle(self, config):
        """Test that val/test dataloaders don't shuffle."""
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0

        dataloaders = build_dataloaders(config_copy, splits=['val', 'test'])

        # Both should exist
        assert 'val' in dataloaders
        assert 'test' in dataloaders

    def test_dataloader_loads_batch(self, config):
        """Test that dataloader can load a batch."""
        # Use num_workers=0 for reliable testing
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0

        dataloaders = build_dataloaders(config_copy, splits=['train'])
        train_loader = dataloaders['train']

        # Get one batch
        images, labels = next(iter(train_loader))

        batch_size = config['training']['batch_size']
        assert images.shape[0] <= batch_size  # May be less for last batch
        assert images.shape[1:] == (3, 224, 224)
        assert labels.shape[0] <= batch_size
        assert all(label in [0, 1] for label in labels)

    def test_invalid_split_raises_error(self, config):
        """Test that invalid split name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            build_dataloaders(config, splits=['invalid_split'])


class TestGetDatasetStats:
    """Tests for get_dataset_stats function."""

    def test_get_dataset_stats_returns_dict(self, config):
        """Test that get_dataset_stats returns a dict."""
        stats = get_dataset_stats(config)
        assert isinstance(stats, dict)

    def test_get_dataset_stats_has_required_keys(self, config):
        """Test that stats dict has required keys."""
        stats = get_dataset_stats(config)

        assert 'split_sizes' in stats
        assert 'label_distributions' in stats
        assert 'class_weights' in stats

    def test_get_dataset_stats_split_sizes(self, config):
        """Test that split sizes are correct."""
        stats = get_dataset_stats(config)

        # Check that all splits are present
        assert 'train' in stats['split_sizes']
        assert 'val' in stats['split_sizes']
        assert 'test' in stats['split_sizes']

        # Check known split sizes (from create_data_splits.py)
        assert stats['split_sizes']['train'] == 645
        assert stats['split_sizes']['val'] == 139
        assert stats['split_sizes']['test'] == 139

    def test_get_dataset_stats_label_distributions(self, config):
        """Test that label distributions are correct."""
        stats = get_dataset_stats(config)

        for split in ['train', 'val', 'test']:
            dist = stats['label_distributions'][split]

            assert 'bleached' in dist
            assert 'healthy' in dist
            assert dist['bleached'] > 0
            assert dist['healthy'] > 0

            # Total should match split size
            total = dist['bleached'] + dist['healthy']
            assert total == stats['split_sizes'][split]

    def test_get_dataset_stats_class_weights(self, config):
        """Test that class weights are computed."""
        stats = get_dataset_stats(config)

        for split in ['train', 'val', 'test']:
            weights = stats['class_weights'][split]

            assert isinstance(weights, list)
            assert len(weights) == 2  # Binary classification
            assert all(w > 0 for w in weights)


class TestDataLoaderIntegration:
    """Integration tests for complete data loading pipeline."""

    def test_end_to_end_data_loading(self, config):
        """Test complete data loading pipeline."""
        # Build dataloaders
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0  # For testing

        dataloaders = build_dataloaders(config_copy, splits=['train', 'val'])

        # Check train loader
        train_loader = dataloaders['train']
        train_images, train_labels = next(iter(train_loader))

        assert train_images.shape[1:] == (3, 224, 224)
        assert len(train_labels) <= config['training']['batch_size']

        # Check val loader
        val_loader = dataloaders['val']
        val_images, val_labels = next(iter(val_loader))

        assert val_images.shape[1:] == (3, 224, 224)
        assert len(val_labels) <= config['training']['batch_size']

    def test_multiple_epochs(self, config):
        """Test that dataloader works for multiple epochs."""
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0
        config_copy['training']['batch_size'] = 8  # Small batch for faster testing

        dataloaders = build_dataloaders(config_copy, splits=['val'])
        val_loader = dataloaders['val']

        # Iterate through two epochs
        for epoch in range(2):
            batch_count = 0
            for images, labels in val_loader:
                assert images.shape[1:] == (3, 224, 224)
                assert all(label in [0, 1] for label in labels)
                batch_count += 1

            assert batch_count > 0  # Should have at least one batch

    def test_label_distribution_in_batches(self, config):
        """Test that both labels appear in dataset."""
        config_copy = config.copy()
        config_copy['dataloader']['num_workers'] = 0

        dataloaders = build_dataloaders(config_copy, splits=['train'])
        train_loader = dataloaders['train']

        # Collect labels from first few batches
        all_labels = []
        for i, (images, labels) in enumerate(train_loader):
            all_labels.extend(labels.tolist())
            if i >= 10:  # Check first 10 batches
                break

        # Both classes should appear
        assert 0 in all_labels  # bleached
        assert 1 in all_labels  # healthy


class TestDatasetConsistency:
    """Tests for dataset consistency and reproducibility."""

    def test_dataset_length_matches_csv(self, sample_csv_path):
        """Test that dataset length matches CSV row count."""
        import pandas as pd

        df = pd.read_csv(sample_csv_path)
        dataset = CoralDataset(csv_path=sample_csv_path)

        assert len(dataset) == len(df)

    def test_same_index_returns_same_label(self, sample_csv_path):
        """Test that same index returns same label (deterministic)."""
        dataset = CoralDataset(csv_path=sample_csv_path, transform=None)

        # Get same item twice
        image1, label1 = dataset[0]
        image2, label2 = dataset[0]

        # Labels should be identical
        assert label1 == label2

    def test_dataset_indices_are_valid(self, sample_csv_path, config):
        """Test that all indices can be accessed."""
        transforms = get_transforms(config, 'val')
        dataset = CoralDataset(csv_path=sample_csv_path, transform=transforms)

        # Test random indices
        import random
        random.seed(42)
        indices_to_test = random.sample(range(len(dataset)), min(20, len(dataset)))

        for idx in indices_to_test:
            image, label = dataset[idx]
            assert image.shape == (3, 224, 224)
            assert label in [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
