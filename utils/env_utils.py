"""
Environment utilities for handling local vs Colab environments.

This module provides helpers to detect the runtime environment and resolve
file paths correctly in both local development and Google Colab environments.
"""

import os
import sys
from pathlib import Path
from typing import Union, Optional


def is_colab() -> bool:
    """
    Detect if code is running in Google Colab environment.

    Returns:
        bool: True if running in Colab, False otherwise.
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    """
    Get the project root directory path.

    In local environment: Returns the repo root directory.
    In Colab: Returns /content/drive/MyDrive/coral-bleaching/ if mounted,
              otherwise returns /content/<repo-name>/ (cloned repo).

    Returns:
        Path: Absolute path to project root.
    """
    if is_colab():
        # Check if Google Drive is mounted
        drive_path = Path("/content/drive/MyDrive/coral-bleaching")
        if drive_path.exists():
            return drive_path

        # Otherwise, assume repo is cloned to /content/
        # Look for a directory containing configs/ and data/
        content_dir = Path("/content")
        for item in content_dir.iterdir():
            if item.is_dir() and (item / "configs").exists():
                return item

        # Fallback: return current working directory
        return Path.cwd()
    else:
        # Local environment: find repo root by looking for .git directory
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "configs").exists():
                return parent

        # Fallback: return parent of utils directory
        return Path(__file__).resolve().parent.parent


def resolve_data_path(
    relative_path: Union[str, Path],
    base_dir: Optional[Path] = None
) -> Path:
    """
    Resolve a relative data path to an absolute path.

    This function handles paths from split CSVs (e.g., 'data/raw/bleached/img.jpg')
    and resolves them correctly in both local and Colab environments.

    Args:
        relative_path: Relative path from split CSV (e.g., 'data/raw/bleached/img.jpg').
        base_dir: Optional base directory. If None, uses project root.

    Returns:
        Path: Absolute path to the data file.

    Examples:
        >>> resolve_data_path('data/raw/bleached/image.jpg')
        PosixPath('/Users/user/projects/coral-bleaching/data/raw/bleached/image.jpg')
    """
    if base_dir is None:
        base_dir = get_project_root()

    relative_path = Path(relative_path)

    # If already absolute, return as-is
    if relative_path.is_absolute():
        return relative_path

    # Resolve relative to base directory
    return (base_dir / relative_path).resolve()


def resolve_checkpoint_path(
    checkpoint_name: Union[str, Path],
    checkpoint_dir: str = "checkpoints"
) -> Path:
    """
    Resolve a checkpoint path for saving or loading model checkpoints.

    In local environment: Uses local checkpoints/ directory.
    In Colab: Uses Google Drive checkpoints/ directory if mounted,
              otherwise uses local /content/checkpoints/.

    Args:
        checkpoint_name: Name or relative path of checkpoint
                        (e.g., 'teacher/best_model.pth' or 'model.pth').
        checkpoint_dir: Name of checkpoint directory (default: 'checkpoints').

    Returns:
        Path: Absolute path to checkpoint location.

    Examples:
        >>> resolve_checkpoint_path('teacher/best_model.pth')
        PosixPath('/Users/user/projects/coral-bleaching/checkpoints/teacher/best_model.pth')
    """
    checkpoint_name = Path(checkpoint_name)

    # If already absolute, return as-is
    if checkpoint_name.is_absolute():
        return checkpoint_name

    # Get project root
    root = get_project_root()

    # Construct checkpoint path
    checkpoint_path = root / checkpoint_dir / checkpoint_name

    # Ensure parent directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return checkpoint_path


def get_data_root() -> Path:
    """
    Get the root directory for data files.

    Returns:
        Path: Absolute path to data root directory.
    """
    return get_project_root() / "data"


def get_checkpoint_root() -> Path:
    """
    Get the root directory for checkpoints.

    Returns:
        Path: Absolute path to checkpoint root directory.
    """
    checkpoint_root = get_project_root() / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return checkpoint_root


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print(f"Running in Colab: {is_colab()}")
    print(f"Project root: {get_project_root()}")
    print(f"Data root: {get_data_root()}")
    print(f"Checkpoint root: {get_checkpoint_root()}")

    # Test path resolution
    test_path = "data/raw/bleached/test.jpg"
    print(f"\nTest relative path: {test_path}")
    print(f"Resolved to: {resolve_data_path(test_path)}")

    test_checkpoint = "teacher/best_model.pth"
    print(f"\nTest checkpoint: {test_checkpoint}")
    print(f"Resolved to: {resolve_checkpoint_path(test_checkpoint)}")
