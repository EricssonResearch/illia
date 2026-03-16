"""PyTorch normalization layer wrappers."""

# 3pps
import torch.nn as nn


class BatchNorm1d(nn.BatchNorm1d):
    """Wrapper for PyTorch BatchNorm1d."""


class BatchNorm2d(nn.BatchNorm2d):
    """Wrapper for PyTorch BatchNorm2d."""


class LayerNorm(nn.LayerNorm):
    """Wrapper for PyTorch LayerNorm."""
