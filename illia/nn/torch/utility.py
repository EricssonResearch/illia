"""PyTorch utility layer wrappers."""

# 3pps
import torch.nn as nn


class Flatten(nn.Flatten):
    """Wrapper for PyTorch Flatten."""


class Identity(nn.Identity):
    """Wrapper for PyTorch Identity."""
