"""
This module contains the import statements for NN layers used with JAX.
"""

# Own modules
from illia.nn.jax.base import BayesianModule
from illia.nn.jax.conv1d import Conv1D
from illia.nn.jax.conv2d import Conv2D
from illia.nn.jax.linear import Linear


# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Conv1D",
    "Conv2D",
    "Linear",
]
