"""
This module consolidates and exposes layers-related classes
implemented in Tensorflow. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.tf.activation import GELU, LeakyReLU, ReLU, Sigmoid, Tanh

# Own modules - Bayesian layers
from illia.nn.tf.base import BayesianModule
from illia.nn.tf.conv1d import Conv1d
from illia.nn.tf.conv2d import Conv2d
from illia.nn.tf.embedding import Embedding
from illia.nn.tf.linear import Linear
from illia.nn.tf.lstm import LSTM
from illia.nn.tf.normalization import BatchNorm1d, BatchNorm2d, LayerNorm

# Own modules - Non-parametric layers
from illia.nn.tf.pooling import (
    AdaptiveAvgPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)
from illia.nn.tf.regularization import Dropout, Dropout2d
from illia.nn.tf.utility import Flatten


__all__: list[str] = [
    # Bayesian layers
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Embedding",
    "Linear",
    "LSTM",
    # Pooling layers
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
    # Activation layers
    "ReLU",
    "Sigmoid",
    "Tanh",
    "LeakyReLU",
    "GELU",
    # Normalization layers
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",
    # Regularization layers
    "Dropout",
    "Dropout2d",
    # Utility layers
    "Flatten",
]
