"""
This module consolidates and exposes layers-related classes
implemented in PyTorch. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.torch.activation import GELU, LeakyReLU, ReLU, Sigmoid, Tanh
from illia.nn.torch.base import BayesianModule
from illia.nn.torch.conv1d import Conv1d
from illia.nn.torch.conv2d import Conv2d
from illia.nn.torch.embedding import Embedding
from illia.nn.torch.linear import Linear
from illia.nn.torch.lstm import LSTM
from illia.nn.torch.normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from illia.nn.torch.pooling import (
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)
from illia.nn.torch.regularization import Dropout, Dropout2d
from illia.nn.torch.utility import Flatten, Identity


__all__: list[str] = [
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Dropout",
    "Dropout2d",
    "Embedding",
    "Flatten",
    "GELU",
    "Identity",
    "LSTM",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MaxPool1d",
    "MaxPool2d",
    "ReLU",
    "Sigmoid",
    "Tanh",
]
