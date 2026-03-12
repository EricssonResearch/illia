"""PyTorch activation layer wrappers."""

# 3pps
from torch import nn


class ReLU(nn.ReLU):
    """Wrapper for PyTorch ReLU."""


class Sigmoid(nn.Sigmoid):
    """Wrapper for PyTorch Sigmoid."""


class Tanh(nn.Tanh):
    """Wrapper for PyTorch Tanh."""


class LeakyReLU(nn.LeakyReLU):
    """Wrapper for PyTorch LeakyReLU."""


class GELU(nn.GELU):
    """Wrapper for PyTorch GELU."""
