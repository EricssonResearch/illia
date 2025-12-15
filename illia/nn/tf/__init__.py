"""
This module consolidates and exposes layers-related classes
implemented in Tensorflow. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.tf.base import BayesianModule
from illia.nn.tf.conv1d import Conv1d
from illia.nn.tf.conv2d import Conv2d
from illia.nn.tf.embedding import Embedding
from illia.nn.tf.linear import Linear
from illia.nn.tf.lstm import LSTM


__all__: list[str] = [
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Embedding",
    "Linear",
    "LSTM",
]
