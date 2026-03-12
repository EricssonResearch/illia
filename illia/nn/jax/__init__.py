"""
This module consolidates and exposes layers-related classes
implemented in JAX. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.jax.activation import GELU, ReLU, Sigmoid, Tanh
from illia.nn.jax.base import BayesianModule
from illia.nn.jax.conv1d import Conv1d
from illia.nn.jax.conv2d import Conv2d
from illia.nn.jax.embedding import Embedding
from illia.nn.jax.linear import Linear
from illia.nn.jax.lstm import LSTM
from illia.nn.jax.normalization import BatchNorm1d, BatchNorm2d, LayerNorm
from illia.nn.jax.pooling import AvgPool1d, AvgPool2d, MaxPool1d, MaxPool2d
from illia.nn.jax.regularization import Dropout


__all__: list[str] = [
    "AvgPool1d",
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Dropout",
    "Embedding",
    "GELU",
    "LSTM",
    "LayerNorm",
    "Linear",
    "MaxPool1d",
    "MaxPool2d",
    "ReLU",
    "Sigmoid",
    "Tanh",
]
