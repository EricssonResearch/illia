"""
This module contains the import statements for NN layers used with JAX.
"""

# Own modules
from illia.nn.jax.base import BayesianModule
from illia.nn.jax.linear import Linear


# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Linear",
]
