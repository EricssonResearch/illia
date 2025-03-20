"""
This module defines the imports for illia.jax.nn.
"""

# Own modules
from illia.jax.nn.base import BayesianModule
from illia.jax.nn.linear import Linear

# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Linear",
]
