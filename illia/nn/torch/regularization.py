"""PyTorch regularization layer wrappers."""

# 3pps
import torch.nn as nn


class Dropout(nn.Dropout):
    """Wrapper for PyTorch Dropout."""


class Dropout2d(nn.Dropout2d):
    """Wrapper for PyTorch Dropout2d."""
