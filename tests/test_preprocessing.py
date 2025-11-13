"""
Tests for utils/preprocessing.py

Tests image transformation pipelines for train/val/test modes.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose

from utils.preprocessing import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
    get_transforms,
    get_normalization_stats,
    get_image_size,
    apply_preprocessing,
)


@pytest.fixture
def sample_config():
    """Sample config for testing."""
    return {
        'dataset': {
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
            }
        }
    }


@pytest.fixture
def minimal_config():
    """Minimal config without augmentation."""
    return {
        'dataset': {
            'image_size': [224, 224],
        },
        'augmentations': {
            'random_flip': False,
            'color_jitter': {
                'enabled': False,
            },
            'normalization': {
                'mean': [0.5, 0.5, 0.5],
                'std': [0.5, 0.5, 0.5],
            }
        }
    }


@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


class TestGetTrainTransforms:
    """Tests for get_train_transforms() function."""

    def test_returns_compose_object(self, sample_config):
        """Test that get_train_transforms returns a Compose object."""
        transforms = get_train_transforms(sample_config)
        assert isinstance(transforms, Compose)

    def test_transform_output_shape(self, sample_config, sample_image):
        """Test that transform output has correct shape."""
        transforms = get_train_transforms(sample_config)
        output = transforms(sample_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)  # C, H, W

    def test_transform_output_dtype(self, sample_config, sample_image):
        """Test that transform output has correct dtype."""
        transforms = get_train_transforms(sample_config)
        output = transforms(sample_image)

        assert output.dtype == torch.float32

    def test_normalization_applied(self, sample_config, sample_image):
        """Test that normalization is applied (values not in [0, 1])."""
        transforms = get_train_transforms(sample_config)
        output = transforms(sample_image)

        # After normalization, values should not be strictly in [0, 1]
        # ImageNet normalization typically produces negative values
        assert output.min() < 0 or output.max() > 1

    def test_with_minimal_config(self, minimal_config, sample_image):
        """Test transforms with minimal augmentation."""
        transforms = get_train_transforms(minimal_config)
        output = transforms(sample_image)

        assert output.shape == (3, 224, 224)
        assert isinstance(output, torch.Tensor)

    def test_different_image_sizes(self, sample_config, sample_image):
        """Test with different target image sizes."""
        for size in [[128, 128], [256, 256], [299, 299]]:
            config = sample_config.copy()
            config['dataset']['image_size'] = size
            transforms = get_train_transforms(config)
            output = transforms(sample_image)

            assert output.shape == (3, size[0], size[1])


class TestGetValTransforms:
    """Tests for get_val_transforms() function."""

    def test_returns_compose_object(self, sample_config):
        """Test that get_val_transforms returns a Compose object."""
        transforms = get_val_transforms(sample_config)
        assert isinstance(transforms, Compose)

    def test_transform_output_shape(self, sample_config, sample_image):
        """Test that transform output has correct shape."""
        transforms = get_val_transforms(sample_config)
        output = transforms(sample_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_deterministic_output(self, sample_config, sample_image):
        """Test that val transforms are deterministic (no randomness)."""
        transforms = get_val_transforms(sample_config)

        # Apply transforms twice to the same image
        output1 = transforms(sample_image)
        output2 = transforms(sample_image)

        # Outputs should be identical (no random augmentation)
        assert torch.allclose(output1, output2)

    def test_normalization_applied(self, sample_config, sample_image):
        """Test that normalization is applied."""
        transforms = get_val_transforms(sample_config)
        output = transforms(sample_image)

        # After normalization, values should not be strictly in [0, 1]
        assert output.min() < 0 or output.max() > 1


class TestGetTestTransforms:
    """Tests for get_test_transforms() function."""

    def test_returns_compose_object(self, sample_config):
        """Test that get_test_transforms returns a Compose object."""
        transforms = get_test_transforms(sample_config)
        assert isinstance(transforms, Compose)

    def test_transform_output_shape(self, sample_config, sample_image):
        """Test that transform output has correct shape."""
        transforms = get_test_transforms(sample_config)
        output = transforms(sample_image)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (3, 224, 224)

    def test_identical_to_val_transforms(self, sample_config, sample_image):
        """Test that test transforms are identical to val transforms."""
        val_transforms = get_val_transforms(sample_config)
        test_transforms = get_test_transforms(sample_config)

        val_output = val_transforms(sample_image)
        test_output = test_transforms(sample_image)

        # Outputs should be identical
        assert torch.allclose(val_output, test_output)


class TestGetTransforms:
    """Tests for get_transforms() convenience function."""

    def test_train_split(self, sample_config, sample_image):
        """Test get_transforms with 'train' split."""
        transforms = get_transforms(sample_config, 'train')
        output = transforms(sample_image)

        assert output.shape == (3, 224, 224)

    def test_val_split(self, sample_config, sample_image):
        """Test get_transforms with 'val' split."""
        transforms = get_transforms(sample_config, 'val')
        output = transforms(sample_image)

        assert output.shape == (3, 224, 224)

    def test_test_split(self, sample_config, sample_image):
        """Test get_transforms with 'test' split."""
        transforms = get_transforms(sample_config, 'test')
        output = transforms(sample_image)

        assert output.shape == (3, 224, 224)

    def test_invalid_split_raises_error(self, sample_config):
        """Test that invalid split raises ValueError."""
        with pytest.raises(ValueError, match="Invalid split"):
            get_transforms(sample_config, 'invalid_split')

    def test_default_split_is_train(self, sample_config):
        """Test that default split is 'train'."""
        transforms = get_transforms(sample_config)
        train_transforms = get_train_transforms(sample_config)

        # Should have same number of transforms
        assert len(transforms.transforms) == len(train_transforms.transforms)


class TestGetNormalizationStats:
    """Tests for get_normalization_stats() function."""

    def test_returns_tuple(self, sample_config):
        """Test that function returns a tuple."""
        result = get_normalization_stats(sample_config)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_correct_values(self, sample_config):
        """Test that function returns correct mean and std."""
        mean, std = get_normalization_stats(sample_config)

        expected_mean = [0.485, 0.456, 0.406]
        expected_std = [0.229, 0.224, 0.225]

        assert mean == expected_mean
        assert std == expected_std

    def test_returns_lists(self, sample_config):
        """Test that mean and std are lists."""
        mean, std = get_normalization_stats(sample_config)

        assert isinstance(mean, list)
        assert isinstance(std, list)
        assert len(mean) == 3  # RGB channels
        assert len(std) == 3


class TestGetImageSize:
    """Tests for get_image_size() function."""

    def test_returns_tuple(self, sample_config):
        """Test that function returns a tuple."""
        result = get_image_size(sample_config)
        assert isinstance(result, tuple)

    def test_returns_correct_size(self, sample_config):
        """Test that function returns correct image size."""
        size = get_image_size(sample_config)
        assert size == (224, 224)

    def test_different_sizes(self):
        """Test with different image sizes."""
        for h, w in [(128, 128), (256, 256), (224, 224)]:
            config = {'dataset': {'image_size': [h, w]}}
            size = get_image_size(config)
            assert size == (h, w)


class TestApplyPreprocessing:
    """Tests for apply_preprocessing() backwards compatibility function."""

    def test_backwards_compatibility(self, sample_config, sample_image):
        """Test that apply_preprocessing works like get_transforms."""
        old_way = apply_preprocessing(sample_config, 'train')
        new_way = get_transforms(sample_config, 'train')

        old_output = old_way(sample_image)
        new_output = new_way(sample_image)

        # Outputs should have same shape (might differ due to randomness)
        assert old_output.shape == new_output.shape

    def test_all_splits_work(self, sample_config, sample_image):
        """Test that apply_preprocessing works for all splits."""
        for split in ['train', 'val', 'test']:
            transforms = apply_preprocessing(sample_config, split)
            output = transforms(sample_image)
            assert output.shape == (3, 224, 224)


class TestTransformPipeline:
    """Integration tests for transform pipeline."""

    def test_train_vs_val_difference(self, sample_config):
        """Test that train and val transforms are different."""
        train_transforms = get_train_transforms(sample_config)
        val_transforms = get_val_transforms(sample_config)

        # Train should have more transforms (augmentation)
        assert len(train_transforms.transforms) > len(val_transforms.transforms)

    def test_val_vs_test_identical(self, sample_config):
        """Test that val and test transforms are identical."""
        val_transforms = get_val_transforms(sample_config)
        test_transforms = get_test_transforms(sample_config)

        # Should have same number of transforms
        assert len(val_transforms.transforms) == len(test_transforms.transforms)

    def test_batch_processing(self, sample_config):
        """Test that transforms can be applied to multiple images."""
        transforms = get_val_transforms(sample_config)

        # Create multiple images
        images = [Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        ) for _ in range(5)]

        # Apply transforms to all
        outputs = [transforms(img) for img in images]

        # All outputs should have correct shape
        assert all(out.shape == (3, 224, 224) for out in outputs)

        # Can be stacked into a batch
        batch = torch.stack(outputs)
        assert batch.shape == (5, 3, 224, 224)


class TestRealCoralImages:
    """INTEGRATION TESTS: Real coral images from dataset."""

    def test_transforms_on_real_coral_images(self, sample_config):
        """INTEGRATION: Test transforms on actual coral images from dataset."""
        from utils.data_loader import CoralDataset
        from pathlib import Path

        # Load dataset without transforms to get raw PIL images
        dataset = CoralDataset('data/splits/train.csv', transform=None)

        # Test on first 5 real coral images
        for i in range(min(5, len(dataset))):
            real_image, label = dataset[i]

            # Test all transform modes on real coral images
            for split in ['train', 'val', 'test']:
                transforms = get_transforms(sample_config, split)
                output = transforms(real_image)

                # Verify output is correct
                assert isinstance(output, torch.Tensor)
                assert output.shape == (3, 224, 224)
                assert output.dtype == torch.float32

                # Verify values are reasonable (not NaN or Inf)
                assert not torch.isnan(output).any()
                assert not torch.isinf(output).any()

    def test_augmentation_preserves_coral_features(self, sample_config):
        """INTEGRATION: Test that augmentations preserve coral image characteristics."""
        from utils.data_loader import CoralDataset

        # Load one real coral image
        dataset = CoralDataset('data/splits/train.csv', transform=None)
        real_image, label = dataset[0]

        # Get train transforms (with augmentation)
        train_transforms = get_train_transforms(sample_config)

        # Apply transforms multiple times
        outputs = [train_transforms(real_image) for _ in range(10)]

        # All outputs should be valid tensors
        for output in outputs:
            assert output.shape == (3, 224, 224)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

            # Check that values are in reasonable range after normalization
            # ImageNet normalization typically gives values in roughly [-3, 3]
            assert output.min() >= -5.0
            assert output.max() <= 5.0

        # Check that augmentation introduces variation
        # (outputs should not all be identical due to random transforms)
        stacked = torch.stack(outputs)
        std_per_pixel = stacked.std(dim=0)

        # At least some pixels should have variation across augmentations
        assert (std_per_pixel > 0.01).any()

    def test_val_transforms_are_deterministic_on_real_images(self, sample_config):
        """INTEGRATION: Test that val transforms are deterministic on real images."""
        from utils.data_loader import CoralDataset

        # Load one real coral image
        dataset = CoralDataset('data/splits/val.csv', transform=None)
        real_image, label = dataset[0]

        # Get val transforms (no augmentation)
        val_transforms = get_val_transforms(sample_config)

        # Apply transforms multiple times
        output1 = val_transforms(real_image)
        output2 = val_transforms(real_image)
        output3 = val_transforms(real_image)

        # All outputs should be identical (no randomness)
        assert torch.allclose(output1, output2)
        assert torch.allclose(output2, output3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
