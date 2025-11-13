"""
Data loading utilities for coral bleaching datasets.

This module provides dataset classes and dataloader factories for loading
coral bleaching images from split CSV files.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from utils.env_utils import resolve_data_path, get_project_root
from utils.preprocessing import get_transforms


# Label mapping for binary classification
LABEL_MAP = {
    'bleached': 0,
    'healthy': 1,
}

# Inverse mapping for displaying labels
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class CoralDataset(Dataset):
    """
    PyTorch Dataset for coral bleaching images.

    Reads image paths and labels from a CSV file and loads images on-the-fly.
    Applies transforms for preprocessing and augmentation.

    Attributes:
        data_frame: DataFrame containing image paths and labels
        transform: Transform pipeline to apply to images
        label_map: Dictionary mapping label strings to integers
        base_dir: Base directory for resolving relative image paths
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        label_map: Dict[str, int] = LABEL_MAP,
        base_dir: Optional[Path] = None,
    ):
        """
        Initialize CoralDataset.

        Args:
            csv_path: Path to CSV file containing image_path and label columns.
            transform: Optional transform to apply to images.
            label_map: Dictionary mapping label strings to class indices.
            base_dir: Base directory for resolving relative paths.
                     If None, uses project root.

        Raises:
            FileNotFoundError: If CSV file doesn't exist.
            ValueError: If CSV is missing required columns.
        """
        # Resolve CSV path
        csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = get_project_root() / csv_path

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV
        self.data_frame = pd.read_csv(csv_path)

        # Validate required columns
        required_columns = ['image_path', 'label']
        missing_columns = set(required_columns) - set(self.data_frame.columns)
        if missing_columns:
            raise ValueError(
                f"CSV missing required columns: {missing_columns}. "
                f"Found columns: {list(self.data_frame.columns)}"
            )

        self.transform = transform
        self.label_map = label_map
        self.base_dir = base_dir if base_dir is not None else get_project_root()

        # Validate that all labels are in label_map
        unique_labels = set(self.data_frame['label'].unique())
        unknown_labels = unique_labels - set(label_map.keys())
        if unknown_labels:
            raise ValueError(
                f"Unknown labels in CSV: {unknown_labels}. "
                f"Expected labels: {set(label_map.keys())}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, int]: (image_tensor, label_index)
                image_tensor: Transformed image tensor of shape (C, H, W)
                label_index: Integer label (0 for bleached, 1 for healthy)

        Raises:
            FileNotFoundError: If image file doesn't exist.
        """
        # Get image path and label from dataframe
        row = self.data_frame.iloc[idx]
        image_path = row['image_path']
        label_str = row['label']

        # Resolve image path
        full_image_path = resolve_data_path(image_path, base_dir=self.base_dir)

        if not full_image_path.exists():
            raise FileNotFoundError(
                f"Image file not found: {full_image_path}\n"
                f"Original path from CSV: {image_path}"
            )

        # Load image
        image = Image.open(full_image_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Convert label to index
        label_idx = self.label_map[label_str]

        return image, label_idx

    def get_label_counts(self) -> Dict[str, int]:
        """
        Get the count of samples for each label.

        Returns:
            Dict[str, int]: Dictionary mapping label names to counts.
        """
        return self.data_frame['label'].value_counts().to_dict()

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.

        Weights are computed as inverse frequency: w_i = N / (n_classes * n_i)
        where N is total samples and n_i is samples for class i.

        Returns:
            torch.Tensor: Class weights of shape (n_classes,).
        """
        label_counts = self.get_label_counts()
        total_samples = len(self)
        n_classes = len(self.label_map)

        weights = []
        for label_str in sorted(self.label_map.keys(), key=lambda x: self.label_map[x]):
            count = label_counts.get(label_str, 0)
            if count == 0:
                weight = 0.0
            else:
                weight = total_samples / (n_classes * count)
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(
    config: Dict[str, Any],
    splits: Optional[list] = None,
) -> Dict[str, DataLoader]:
    """
    Build dataloaders for train, validation, and test splits.

    Args:
        config: Configuration dictionary containing:
            - dataset.splits_dir: Directory containing split CSV files
            - dataset.train_split: Name of train split (default: 'train')
            - dataset.val_split: Name of val split (default: 'val')
            - dataset.test_split: Name of test split (default: 'test')
            - training.batch_size: Batch size for dataloaders
            - dataloader.num_workers: Number of worker processes
            - dataloader.pin_memory: Whether to pin memory
        splits: Optional list of split names to build. If None, builds all.
                Valid values: ['train', 'val', 'test']

    Returns:
        Dict[str, DataLoader]: Dictionary mapping split names to DataLoader objects.

    Examples:
        >>> config = load_config('configs/config.yaml')
        >>> dataloaders = build_dataloaders(config)
        >>> train_loader = dataloaders['train']
        >>> for images, labels in train_loader:
        ...     # Training loop
        ...     pass
    """
    # Default to all splits if not specified
    if splits is None:
        splits = ['train', 'val', 'test']

    # Extract config values
    splits_dir = config['dataset']['splits_dir']
    batch_size = config['training']['batch_size']
    num_workers = config['dataloader'].get('num_workers', 4)
    pin_memory = config['dataloader'].get('pin_memory', True)

    # Map split names to CSV filenames
    split_name_map = {
        'train': config['dataset'].get('train_split', 'train'),
        'val': config['dataset'].get('val_split', 'val'),
        'test': config['dataset'].get('test_split', 'test'),
    }

    dataloaders = {}

    for split in splits:
        if split not in split_name_map:
            raise ValueError(
                f"Invalid split '{split}'. Valid splits: {list(split_name_map.keys())}"
            )

        # Get CSV filename for this split
        csv_filename = f"{split_name_map[split]}.csv"
        csv_path = Path(splits_dir) / csv_filename

        # Get transforms for this split
        transform = get_transforms(config, split=split)

        # Create dataset
        dataset = CoralDataset(
            csv_path=csv_path,
            transform=transform,
            label_map=LABEL_MAP,
        )

        # Shuffle only for training
        shuffle = (split == 'train')

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Only if workers are used
        )

        dataloaders[split] = dataloader

    return dataloaders


def get_dataset_stats(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get statistics about the dataset splits.

    Args:
        config: Configuration dictionary.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - split_sizes: Number of samples per split
            - label_distributions: Label counts per split
            - class_weights: Class weights for each split

    Examples:
        >>> config = load_config('configs/config.yaml')
        >>> stats = get_dataset_stats(config)
        >>> print(f"Train size: {stats['split_sizes']['train']}")
    """
    splits_dir = config['dataset']['splits_dir']
    split_names = ['train', 'val', 'test']

    stats = {
        'split_sizes': {},
        'label_distributions': {},
        'class_weights': {},
    }

    for split in split_names:
        csv_path = Path(splits_dir) / f"{split}.csv"

        if not csv_path.exists():
            continue

        # Create dataset (no transforms needed for stats)
        dataset = CoralDataset(csv_path=csv_path, transform=None)

        # Get statistics
        stats['split_sizes'][split] = len(dataset)
        stats['label_distributions'][split] = dataset.get_label_counts()
        stats['class_weights'][split] = dataset.get_class_weights().tolist()

    return stats


if __name__ == "__main__":
    # Quick diagnostic when run directly
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path("configs/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("=== Dataset Statistics ===")
        stats = get_dataset_stats(config)

        for split in ['train', 'val', 'test']:
            if split in stats['split_sizes']:
                print(f"\n{split.upper()} Split:")
                print(f"  Total samples: {stats['split_sizes'][split]}")
                print(f"  Label distribution: {stats['label_distributions'][split]}")
                print(f"  Class weights: {stats['class_weights'][split]}")

        print("\n=== Building DataLoaders ===")
        dataloaders = build_dataloaders(config, splits=['train'])
        train_loader = dataloaders['train']
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Batch size: {train_loader.batch_size}")

        # Try loading one batch
        print("\n=== Loading Sample Batch ===")
        images, labels = next(iter(train_loader))
        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels[:5]}")
    else:
        print(f"Config file not found: {config_path}")
