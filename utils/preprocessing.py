"""
Preprocessing transformations for coral bleaching images.

This module provides image transformation pipelines for training, validation, and testing.
Supports augmentation from config and ImageNet normalization.
"""

from typing import Dict, Any, List, Tuple
import torchvision.transforms as T
from torchvision.transforms import Compose


def get_train_transforms(config: Dict[str, Any]) -> Compose:
    """
    Get training transforms with augmentation.

    Applies the following transforms in order:
    1. Resize to target image size
    2. Random horizontal flip (if enabled)
    3. Color jitter (if enabled)
    4. Convert to tensor
    5. Normalize with ImageNet stats

    Args:
        config: Configuration dictionary containing:
            - dataset.image_size: Target image size [height, width]
            - augmentations.random_flip: Whether to apply random horizontal flip
            - augmentations.color_jitter: Color jitter parameters
            - augmentations.normalization: Normalization mean and std

    Returns:
        Compose: Composed torchvision transforms for training.

    Examples:
        >>> config = {'dataset': {'image_size': [224, 224]}, ...}
        >>> transforms = get_train_transforms(config)
        >>> # Apply to PIL image
        >>> tensor = transforms(pil_image)
    """
    # Extract config values
    image_size = tuple(config['dataset']['image_size'])
    aug_config = config['augmentations']

    transforms = []

    # 1. Resize
    transforms.append(T.Resize(image_size))

    # 2. Random horizontal flip
    if aug_config.get('random_flip', False):
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # 3. Color jitter
    if aug_config.get('color_jitter', {}).get('enabled', False):
        jitter_params = aug_config['color_jitter']
        transforms.append(T.ColorJitter(
            brightness=jitter_params.get('brightness', 0.0),
            contrast=jitter_params.get('contrast', 0.0),
            saturation=jitter_params.get('saturation', 0.0),
            hue=jitter_params.get('hue', 0.0),
        ))

    # 4. Convert to tensor
    transforms.append(T.ToTensor())

    # 5. Normalize
    norm_config = aug_config['normalization']
    transforms.append(T.Normalize(
        mean=norm_config['mean'],
        std=norm_config['std']
    ))

    return Compose(transforms)


def get_val_transforms(config: Dict[str, Any]) -> Compose:
    """
    Get validation transforms without augmentation.

    Applies the following transforms in order:
    1. Resize to target image size
    2. Convert to tensor
    3. Normalize with ImageNet stats

    Args:
        config: Configuration dictionary containing:
            - dataset.image_size: Target image size [height, width]
            - augmentations.normalization: Normalization mean and std

    Returns:
        Compose: Composed torchvision transforms for validation.

    Examples:
        >>> config = {'dataset': {'image_size': [224, 224]}, ...}
        >>> transforms = get_val_transforms(config)
        >>> tensor = transforms(pil_image)
    """
    # Extract config values
    image_size = tuple(config['dataset']['image_size'])
    norm_config = config['augmentations']['normalization']

    transforms = [
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(
            mean=norm_config['mean'],
            std=norm_config['std']
        )
    ]

    return Compose(transforms)


def get_test_transforms(config: Dict[str, Any]) -> Compose:
    """
    Get test transforms (same as validation).

    Applies the following transforms in order:
    1. Resize to target image size
    2. Convert to tensor
    3. Normalize with ImageNet stats

    Args:
        config: Configuration dictionary containing:
            - dataset.image_size: Target image size [height, width]
            - augmentations.normalization: Normalization mean and std

    Returns:
        Compose: Composed torchvision transforms for testing.

    Examples:
        >>> config = {'dataset': {'image_size': [224, 224]}, ...}
        >>> transforms = get_test_transforms(config)
        >>> tensor = transforms(pil_image)
    """
    # Test transforms are identical to validation transforms
    return get_val_transforms(config)


def get_transforms(config: Dict[str, Any], split: str = 'train') -> Compose:
    """
    Get transforms for a given split.

    Convenience function that routes to the appropriate transform function
    based on the split name.

    Args:
        config: Configuration dictionary.
        split: Split name ('train', 'val', or 'test').

    Returns:
        Compose: Composed torchvision transforms for the specified split.

    Raises:
        ValueError: If split is not 'train', 'val', or 'test'.

    Examples:
        >>> config = load_config('configs/config.yaml')
        >>> train_transforms = get_transforms(config, 'train')
        >>> val_transforms = get_transforms(config, 'val')
    """
    if split == 'train':
        return get_train_transforms(config)
    elif split == 'val':
        return get_val_transforms(config)
    elif split == 'test':
        return get_test_transforms(config)
    else:
        raise ValueError(
            f"Invalid split '{split}'. Must be 'train', 'val', or 'test'."
        )


def get_normalization_stats(config: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    """
    Get normalization mean and standard deviation from config.

    Args:
        config: Configuration dictionary containing augmentations.normalization.

    Returns:
        Tuple[List[float], List[float]]: (mean, std) for normalization.

    Examples:
        >>> config = {'augmentations': {'normalization': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}}}
        >>> mean, std = get_normalization_stats(config)
        >>> print(mean)
        [0.5, 0.5, 0.5]
    """
    norm_config = config['augmentations']['normalization']
    return norm_config['mean'], norm_config['std']


def get_image_size(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    Get target image size from config.

    Args:
        config: Configuration dictionary containing dataset.image_size.

    Returns:
        Tuple[int, int]: (height, width) for image resizing.

    Examples:
        >>> config = {'dataset': {'image_size': [224, 224]}}
        >>> height, width = get_image_size(config)
        >>> print(height, width)
        224 224
    """
    image_size = config['dataset']['image_size']
    return tuple(image_size)


# Backwards compatibility - keep the old function name but update implementation
def apply_preprocessing(config: Dict[str, Any], split: str = 'train') -> Compose:
    """
    Apply preprocessing pipeline for a given split.

    This is an alias for get_transforms() for backwards compatibility.

    Args:
        config: Configuration dictionary.
        split: Split name ('train', 'val', or 'test').

    Returns:
        Compose: Composed torchvision transforms for the specified split.
    """
    return get_transforms(config, split)


if __name__ == "__main__":
    # Quick diagnostic when run directly
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("=== Train Transforms ===")
        train_transforms = get_train_transforms(config)
        print(train_transforms)

        print("\n=== Val/Test Transforms ===")
        val_transforms = get_val_transforms(config)
        print(val_transforms)

        print("\n=== Image Size ===")
        print(get_image_size(config))

        print("\n=== Normalization Stats ===")
        mean, std = get_normalization_stats(config)
        print(f"Mean: {mean}")
        print(f"Std: {std}")
    else:
        print(f"Config file not found: {config_path}")
