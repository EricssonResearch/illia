"""
This module defines the imports for illia.jax.distributions.
"""

# Own modules
from illia.jax.distributions.base import Distribution
from illia.jax.distributions.gaussian import GaussianDistribution

# Define all names to be imported
__all__: list[str] = [
    "Distribution",
    "GaussianDistribution",
]
