#!/usr/bin/env python3
"""
Data Splitting Script for Coral Bleaching Classification

Creates stratified train/val/test splits from the raw dataset.
Generates CSV manifest files with image paths and labels.

Usage:
    python scripts/create_data_splits.py
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# Configuration
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Class labels
CLASSES = ["bleached", "healthy"]


def collect_image_paths() -> List[Tuple[str, str]]:
    """
    Collect all image paths from raw data directory.

    Returns:
        List of tuples (relative_path, label)
    """
    image_data = []

    for class_name in CLASSES:
        class_dir = RAW_DATA_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        # Get all image files (jpg, jpeg, png)
        image_files = list(class_dir.glob("*.jpg")) + \
                      list(class_dir.glob("*.jpeg")) + \
                      list(class_dir.glob("*.png"))

        # Convert to relative paths from project root
        for img_path in image_files:
            relative_path = img_path.relative_to(PROJECT_ROOT)
            image_data.append((str(relative_path), class_name))

        print(f"Found {len(image_files)} images in class '{class_name}'")

    return image_data


def create_splits(image_data: List[Tuple[str, str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.

    Args:
        image_data: List of (image_path, label) tuples

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Convert to DataFrame
    df = pd.DataFrame(image_data, columns=["image_path", "label"])

    # Extract features and labels
    X = df["image_path"].values
    y = df["label"].values

    # First split: separate out test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=y
    )

    # Second split: separate train and val from remaining data
    # Val should be 15% of total, which is 15/85 ≈ 0.176 of the remaining data
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    # Create DataFrames
    train_df = pd.DataFrame({"image_path": X_train, "label": y_train})
    val_df = pd.DataFrame({"image_path": X_val, "label": y_val})
    test_df = pd.DataFrame({"image_path": X_test, "label": y_test})

    return train_df, val_df, test_df


def print_split_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print detailed statistics about the splits."""

    total_images = len(train_df) + len(val_df) + len(test_df)

    print("\n" + "="*60)
    print("SPLIT STATISTICS")
    print("="*60)
    print(f"\nTotal images: {total_images}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"\nSplit ratios: Train={TRAIN_RATIO:.0%} | Val={VAL_RATIO:.0%} | Test={TEST_RATIO:.0%}")

    print("\n" + "-"*60)
    print("TRAIN SET")
    print("-"*60)
    print(f"Total: {len(train_df)} ({len(train_df)/total_images:.1%})")
    print(f"  - Bleached: {len(train_df[train_df['label']=='bleached'])}")
    print(f"  - Healthy:  {len(train_df[train_df['label']=='healthy'])}")

    print("\n" + "-"*60)
    print("VALIDATION SET")
    print("-"*60)
    print(f"Total: {len(val_df)} ({len(val_df)/total_images:.1%})")
    print(f"  - Bleached: {len(val_df[val_df['label']=='bleached'])}")
    print(f"  - Healthy:  {len(val_df[val_df['label']=='healthy'])}")

    print("\n" + "-"*60)
    print("TEST SET")
    print("-"*60)
    print(f"Total: {len(test_df)} ({len(test_df)/total_images:.1%})")
    print(f"  - Bleached: {len(test_df[test_df['label']=='bleached'])}")
    print(f"  - Healthy:  {len(test_df[test_df['label']=='healthy'])}")

    print("\n" + "="*60)

    # Verify no duplicates across splits
    train_paths = set(train_df["image_path"])
    val_paths = set(val_df["image_path"])
    test_paths = set(test_df["image_path"])

    assert len(train_paths & val_paths) == 0, "Train/Val overlap detected!"
    assert len(train_paths & test_paths) == 0, "Train/Test overlap detected!"
    assert len(val_paths & test_paths) == 0, "Val/Test overlap detected!"

    print("✓ No overlapping images between splits")
    print("="*60 + "\n")


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save split DataFrames to CSV files."""

    # Create splits directory if it doesn't exist
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ Saved train split to: {train_path}")
    print(f"✓ Saved val split to:   {val_path}")
    print(f"✓ Saved test split to:  {test_path}")


def main():
    """Main execution function."""

    print("="*60)
    print("CORAL BLEACHING DATASET - SPLIT CREATION")
    print("="*60 + "\n")

    # Collect image paths
    print("Collecting image paths from raw data directory...")
    image_data = collect_image_paths()
    print(f"\nTotal images collected: {len(image_data)}\n")

    # Create splits
    print("Creating stratified splits...")
    train_df, val_df, test_df = create_splits(image_data)

    # Print statistics
    print_split_statistics(train_df, val_df, test_df)

    # Save splits
    print("Saving split manifests...")
    save_splits(train_df, val_df, test_df)

    print("\n✓ Data splitting complete!")
    print(f"\nNext steps:")
    print(f"  1. Review split statistics above")
    print(f"  2. Update data/README.md with split information")
    print(f"  3. Commit splits to Git for reproducibility")


if __name__ == "__main__":
    main()
