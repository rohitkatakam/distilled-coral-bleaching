"""
Tests for utils/env_utils.py

Tests environment detection and path resolution functionality.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from utils.env_utils import (
    is_colab,
    get_project_root,
    resolve_data_path,
    resolve_checkpoint_path,
    get_data_root,
    get_checkpoint_root,
)


class TestIsColab:
    """Tests for is_colab() function."""

    def test_is_colab_returns_false_locally(self):
        """Test that is_colab() returns False in local environment."""
        # In normal pytest environment, google.colab won't be importable
        assert is_colab() is False

    @patch.dict(sys.modules, {"google.colab": MagicMock()})
    def test_is_colab_returns_true_when_module_exists(self):
        """Test that is_colab() returns True when google.colab is available."""
        assert is_colab() is True


class TestGetProjectRoot:
    """Tests for get_project_root() function."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root() returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_get_project_root_is_absolute(self):
        """Test that get_project_root() returns an absolute path."""
        root = get_project_root()
        assert root.is_absolute()

    def test_get_project_root_contains_configs(self):
        """Test that project root contains configs directory."""
        root = get_project_root()
        assert (root / "configs").exists()

    def test_get_project_root_contains_utils(self):
        """Test that project root contains utils directory."""
        root = get_project_root()
        assert (root / "utils").exists()


class TestResolveDataPath:
    """Tests for resolve_data_path() function."""

    def test_resolve_data_path_with_relative_path(self):
        """Test resolving a relative data path."""
        relative = "data/raw/bleached/test.jpg"
        resolved = resolve_data_path(relative)

        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert "data/raw/bleached/test.jpg" in str(resolved)

    def test_resolve_data_path_with_path_object(self):
        """Test resolving a Path object."""
        relative = Path("data/raw/healthy/test.jpg")
        resolved = resolve_data_path(relative)

        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert "data/raw/healthy/test.jpg" in str(resolved)

    def test_resolve_data_path_with_custom_base_dir(self):
        """Test resolving with a custom base directory."""
        custom_base = Path("/tmp/custom")
        relative = "data/splits/train.csv"
        resolved = resolve_data_path(relative, base_dir=custom_base)

        expected = custom_base / relative
        assert resolved == expected.resolve()

    def test_resolve_data_path_with_absolute_path(self):
        """Test that absolute paths are returned as-is."""
        absolute = Path("/tmp/absolute/path/image.jpg")
        resolved = resolve_data_path(absolute)

        assert resolved == absolute

    def test_resolve_data_path_uses_project_root_by_default(self):
        """Test that resolve_data_path uses project root by default."""
        relative = "data/raw/test.jpg"
        resolved = resolve_data_path(relative)
        expected_parent = get_project_root()

        assert str(expected_parent) in str(resolved)


class TestResolveCheckpointPath:
    """Tests for resolve_checkpoint_path() function."""

    def test_resolve_checkpoint_path_simple_name(self):
        """Test resolving a simple checkpoint name."""
        checkpoint = "model.pth"
        resolved = resolve_checkpoint_path(checkpoint)

        assert isinstance(resolved, Path)
        assert resolved.is_absolute()
        assert "checkpoints" in str(resolved)
        assert "model.pth" in str(resolved)

    def test_resolve_checkpoint_path_nested_name(self):
        """Test resolving a nested checkpoint path."""
        checkpoint = "teacher/best_model.pth"
        resolved = resolve_checkpoint_path(checkpoint)

        assert isinstance(resolved, Path)
        assert "checkpoints/teacher/best_model.pth" in str(resolved)

    def test_resolve_checkpoint_path_custom_dir(self):
        """Test resolving with a custom checkpoint directory name."""
        checkpoint = "model.pth"
        resolved = resolve_checkpoint_path(checkpoint, checkpoint_dir="saved_models")

        assert "saved_models" in str(resolved)
        assert "model.pth" in str(resolved)

    def test_resolve_checkpoint_path_with_absolute_path(self):
        """Test that absolute paths are returned as-is."""
        absolute = Path("/tmp/absolute/checkpoint.pth")
        resolved = resolve_checkpoint_path(absolute)

        assert resolved == absolute

    def test_resolve_checkpoint_path_creates_parent_dir(self, tmp_path):
        """Test that parent directories are created if they don't exist."""
        # Use a temporary directory for this test
        with patch("utils.env_utils.get_project_root", return_value=tmp_path):
            checkpoint = "nested/path/model.pth"
            resolved = resolve_checkpoint_path(checkpoint)

            # Check that parent directory was created
            assert resolved.parent.exists()
            assert resolved.parent.is_dir()


class TestGetDataRoot:
    """Tests for get_data_root() function."""

    def test_get_data_root_returns_path(self):
        """Test that get_data_root() returns a Path object."""
        data_root = get_data_root()
        assert isinstance(data_root, Path)

    def test_get_data_root_is_absolute(self):
        """Test that get_data_root() returns an absolute path."""
        data_root = get_data_root()
        assert data_root.is_absolute()

    def test_get_data_root_ends_with_data(self):
        """Test that data root path ends with 'data'."""
        data_root = get_data_root()
        assert data_root.name == "data"

    def test_get_data_root_exists(self):
        """Test that data root directory exists."""
        data_root = get_data_root()
        assert data_root.exists()


class TestGetCheckpointRoot:
    """Tests for get_checkpoint_root() function."""

    def test_get_checkpoint_root_returns_path(self):
        """Test that get_checkpoint_root() returns a Path object."""
        checkpoint_root = get_checkpoint_root()
        assert isinstance(checkpoint_root, Path)

    def test_get_checkpoint_root_is_absolute(self):
        """Test that get_checkpoint_root() returns an absolute path."""
        checkpoint_root = get_checkpoint_root()
        assert checkpoint_root.is_absolute()

    def test_get_checkpoint_root_ends_with_checkpoints(self):
        """Test that checkpoint root path ends with 'checkpoints'."""
        checkpoint_root = get_checkpoint_root()
        assert checkpoint_root.name == "checkpoints"

    def test_get_checkpoint_root_creates_directory(self):
        """Test that get_checkpoint_root() creates directory if it doesn't exist."""
        checkpoint_root = get_checkpoint_root()
        # Directory should exist after calling the function
        assert checkpoint_root.exists()
        assert checkpoint_root.is_dir()


class TestIntegration:
    """Integration tests for env_utils."""

    def test_path_resolution_chain(self):
        """Test that paths can be resolved in a realistic workflow."""
        # Get project root
        root = get_project_root()
        assert root.exists()

        # Resolve a data path
        data_path = resolve_data_path("data/splits/train.csv")
        assert data_path.is_absolute()
        assert data_path.exists()  # We know train.csv exists

        # Resolve a checkpoint path (may not exist yet)
        checkpoint_path = resolve_checkpoint_path("test_model.pth")
        assert checkpoint_path.is_absolute()
        assert "checkpoints" in str(checkpoint_path)

    def test_environment_consistency(self):
        """Test that environment functions return consistent results."""
        # Call functions multiple times - should return same results
        root1 = get_project_root()
        root2 = get_project_root()
        assert root1 == root2

        data_root1 = get_data_root()
        data_root2 = get_data_root()
        assert data_root1 == data_root2

        checkpoint_root1 = get_checkpoint_root()
        checkpoint_root2 = get_checkpoint_root()
        assert checkpoint_root1 == checkpoint_root2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
