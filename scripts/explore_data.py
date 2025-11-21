#!/usr/bin/env python3
"""
Dataset Exploration and Visualization

Analyzes the coral bleaching dataset splits and generates exploratory visualizations.
This script provides comprehensive dataset statistics and sample visualizations for the paper.

Usage:
    python scripts/explore_data.py

Outputs:
    - scripts/results/data_exploration/class_distribution.png
    - scripts/results/data_exploration/sample_images.png
    - scripts/results/data_exploration/dataset_stats.txt
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.visualization import plot_class_distribution, plot_sample_grid
from utils.env_utils import get_project_root, resolve_data_path
from utils.preprocessing import get_test_transforms
import yaml

# %% Configuration
PROJECT_ROOT = get_project_root()
DATA_SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "scripts" / "results" / "data_exploration"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Class names
CLASS_NAMES = ['healthy', 'bleached']

# %% Helper Functions

def load_split_statistics():
    """
    Load all split CSVs and compute statistics.

    Returns:
        dict: Statistics for each split with keys 'train', 'val', 'test'
    """
    stats = {}

    for split_name in ['train', 'val', 'test']:
        csv_path = DATA_SPLITS_DIR / f"{split_name}.csv"

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            continue

        # Load CSV
        df = pd.read_csv(csv_path)

        # Count classes
        label_counts = df['label'].value_counts().to_dict()

        stats[split_name] = {
            'total': len(df),
            'healthy': label_counts.get('healthy', 0),
            'bleached': label_counts.get('bleached', 0),
            'dataframe': df
        }

        print(f"{split_name.capitalize()} split: {stats[split_name]['total']} images "
              f"({stats[split_name]['healthy']} healthy, {stats[split_name]['bleached']} bleached)")

    return stats


def compute_image_statistics(split_stats, num_samples=50):
    """
    Compute image statistics (resolution, file size, format) from sample images.

    Args:
        split_stats: Dictionary of split statistics
        num_samples: Number of random images to sample for statistics

    Returns:
        dict: Image statistics (resolutions, file sizes, formats)
    """
    print(f"\nComputing image statistics from {num_samples} random samples...")

    # Collect sample image paths from all splits
    all_paths = []
    for split_name, stats in split_stats.items():
        df = stats['dataframe']
        all_paths.extend(df['image_path'].tolist())

    # Sample random images
    sample_paths = np.random.choice(all_paths, min(num_samples, len(all_paths)), replace=False)

    resolutions = []
    file_sizes = []
    formats = []

    for img_path in sample_paths:
        full_path = resolve_data_path(img_path)

        try:
            # Get file size
            file_size_kb = full_path.stat().st_size / 1024  # Convert to KB
            file_sizes.append(file_size_kb)

            # Get image resolution and format
            with Image.open(full_path) as img:
                resolutions.append(img.size)  # (width, height)
                formats.append(img.format)
        except Exception as e:
            print(f"Warning: Could not load {full_path}: {e}")
            continue

    # Compute statistics
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    format_counts = Counter(formats)

    image_stats = {
        'num_samples': len(resolutions),
        'mean_width': np.mean(widths),
        'mean_height': np.mean(heights),
        'std_width': np.std(widths),
        'std_height': np.std(heights),
        'min_resolution': (min(widths), min(heights)),
        'max_resolution': (max(widths), max(heights)),
        'mean_file_size_kb': np.mean(file_sizes),
        'std_file_size_kb': np.std(file_sizes),
        'formats': dict(format_counts)
    }

    return image_stats


def plot_split_comparison(split_stats, save_path):
    """
    Create stacked bar chart comparing class distribution across splits.

    Args:
        split_stats: Dictionary of split statistics
        save_path: Path to save the plot
    """
    splits = ['train', 'val', 'test']
    healthy_counts = [split_stats[s]['healthy'] for s in splits]
    bleached_counts = [split_stats[s]['bleached'] for s in splits]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(splits))
    width = 0.6

    # Create stacked bars
    p1 = ax.bar(x, healthy_counts, width, label='Healthy', color='#2ecc71')
    p2 = ax.bar(x, bleached_counts, width, bottom=healthy_counts, label='Bleached', color='#e74c3c')

    # Customize plot
    ax.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution Across Dataset Splits', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (h, b) in enumerate(zip(healthy_counts, bleached_counts)):
        # Healthy label
        ax.text(i, h/2, str(h), ha='center', va='center', fontweight='bold', color='white')
        # Bleached label
        ax.text(i, h + b/2, str(b), ha='center', va='center', fontweight='bold', color='white')
        # Total label
        ax.text(i, h + b + 10, f'Total: {h+b}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved class distribution plot to {save_path}")
    plt.close()


def visualize_sample_images(split_stats, save_path, num_per_class=8):
    """
    Create a grid of sample images from both classes.

    Args:
        split_stats: Dictionary of split statistics
        save_path: Path to save the plot
        num_per_class: Number of samples per class to display
    """
    print(f"\nGenerating sample image grid ({num_per_class} per class)...")

    # Load config for image transforms
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Get test transforms (no augmentation)
    transform = get_test_transforms(config)

    # Collect sample paths from train split (largest)
    train_df = split_stats['train']['dataframe']

    # Sample images from each class
    healthy_df = train_df[train_df['label'] == 'healthy'].sample(n=num_per_class, random_state=42)
    bleached_df = train_df[train_df['label'] == 'bleached'].sample(n=num_per_class, random_state=42)

    # Load images
    images = []
    labels = []

    for _, row in pd.concat([healthy_df, bleached_df]).iterrows():
        img_path = resolve_data_path(row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            # Apply transform
            img_tensor = transform(img)
            images.append(img_tensor)
            # Convert label string to index (healthy=0, bleached=1)
            label_idx = CLASS_NAMES.index(row['label'])
            labels.append(label_idx)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            continue

    # Create grid using existing utility
    fig = plot_sample_grid(
        images=images,
        labels=labels,
        predictions=None,  # No predictions for data exploration
        class_names=CLASS_NAMES,
        num_samples=len(images),
        save_path=save_path
    )

    print(f"Saved sample images to {save_path}")


def save_statistics_summary(split_stats, image_stats, save_path):
    """
    Save dataset statistics to a text file.

    Args:
        split_stats: Dictionary of split statistics
        image_stats: Dictionary of image statistics
        save_path: Path to save the summary
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CORAL BLEACHING DATASET STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        total_images = sum(stats['total'] for stats in split_stats.values())
        total_healthy = sum(stats['healthy'] for stats in split_stats.values())
        total_bleached = sum(stats['bleached'] for stats in split_stats.values())

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"  - Healthy: {total_healthy} ({total_healthy/total_images*100:.1f}%)\n")
        f.write(f"  - Bleached: {total_bleached} ({total_bleached/total_images*100:.1f}%)\n\n")

        # Split statistics
        f.write("SPLIT STATISTICS:\n")
        f.write("-" * 80 + "\n")
        for split_name in ['train', 'val', 'test']:
            stats = split_stats[split_name]
            f.write(f"\n{split_name.upper()} Split:\n")
            f.write(f"  Total: {stats['total']} ({stats['total']/total_images*100:.1f}% of dataset)\n")
            f.write(f"  Healthy: {stats['healthy']} ({stats['healthy']/stats['total']*100:.1f}%)\n")
            f.write(f"  Bleached: {stats['bleached']} ({stats['bleached']/stats['total']*100:.1f}%)\n")

        # Image statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("IMAGE STATISTICS (from random sample):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Sample Size: {image_stats['num_samples']} images\n\n")
        f.write(f"Resolution:\n")
        f.write(f"  Mean: {image_stats['mean_width']:.0f} x {image_stats['mean_height']:.0f} pixels\n")
        f.write(f"  Std:  {image_stats['std_width']:.0f} x {image_stats['std_height']:.0f} pixels\n")
        f.write(f"  Min:  {image_stats['min_resolution'][0]} x {image_stats['min_resolution'][1]} pixels\n")
        f.write(f"  Max:  {image_stats['max_resolution'][0]} x {image_stats['max_resolution'][1]} pixels\n\n")
        f.write(f"File Size:\n")
        f.write(f"  Mean: {image_stats['mean_file_size_kb']:.1f} KB\n")
        f.write(f"  Std:  {image_stats['std_file_size_kb']:.1f} KB\n\n")
        f.write(f"Image Formats:\n")
        for fmt, count in image_stats['formats'].items():
            f.write(f"  {fmt}: {count} images\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Saved statistics summary to {save_path}")


# %% Main Execution

def main():
    """Main execution function."""
    print("=" * 80)
    print("CORAL BLEACHING DATASET EXPLORATION")
    print("=" * 80)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # 1. Load split statistics
    print("\n" + "=" * 80)
    print("LOADING DATASET SPLITS")
    print("=" * 80)
    split_stats = load_split_statistics()

    # 2. Compute image statistics
    print("\n" + "=" * 80)
    print("COMPUTING IMAGE STATISTICS")
    print("=" * 80)
    image_stats = compute_image_statistics(split_stats, num_samples=50)

    # 3. Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot class distribution
    print("\nGenerating class distribution plot...")
    plot_split_comparison(
        split_stats,
        save_path=OUTPUT_DIR / "class_distribution.png"
    )

    # Visualize sample images
    visualize_sample_images(
        split_stats,
        save_path=OUTPUT_DIR / "sample_images.png",
        num_per_class=8
    )

    # 4. Save statistics summary
    print("\n" + "=" * 80)
    print("SAVING STATISTICS SUMMARY")
    print("=" * 80)
    save_statistics_summary(
        split_stats,
        image_stats,
        save_path=OUTPUT_DIR / "dataset_stats.txt"
    )

    print("\n" + "=" * 80)
    print("DATASET EXPLORATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - class_distribution.png")
    print(f"  - sample_images.png")
    print(f"  - dataset_stats.txt")


if __name__ == "__main__":
    main()
