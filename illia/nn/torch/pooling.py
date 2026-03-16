"""PyTorch pooling layer wrappers."""

# 3pps
import torch.nn as nn


class MaxPool1d(nn.MaxPool1d):
    """Wrapper for PyTorch MaxPool1d."""


class MaxPool2d(nn.MaxPool2d):
    """Wrapper for PyTorch MaxPool2d."""


class AvgPool1d(nn.AvgPool1d):
    """Wrapper for PyTorch AvgPool1d."""


class AvgPool2d(nn.AvgPool2d):
    """Wrapper for PyTorch AvgPool2d."""


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """Wrapper for PyTorch AdaptiveAvgPool2d."""


class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    """Wrapper for PyTorch AdaptiveMaxPool2d."""
