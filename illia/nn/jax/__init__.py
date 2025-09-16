"""
This module consolidates and exposes layers-related classes
implemented in JAX. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.jax.base import BayesianModule
from illia.nn.jax.conv1d import Conv1d
from illia.nn.jax.conv2d import Conv2d
from illia.nn.jax.embedding import Embedding
from illia.nn.jax.linear import Linear
from illia.nn.jax.lstm import LSTM


# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Embedding",
    "Linear",
    "LSTM",
]
