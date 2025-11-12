"""Data loading utilities for coral bleaching datasets.

TODO:
    * Implement dataset wrappers for raw and processed image assets.
    * Add dataloader factories that respect configuration files and reproducible splits.
    * Support augmentations and sampling strategies for imbalanced classes.
"""


def build_dataloaders(*args, **kwargs):  # pylint: disable=unused-argument
    """Placeholder to build train/val/test dataloaders."""
    raise NotImplementedError("build_dataloaders needs implementation.")
