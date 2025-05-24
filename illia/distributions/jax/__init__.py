"""
This module defines the imports for illia.jax.distributions.
"""

# Own modules
from illia.distributions.jax.gaussian import GaussianDistribution

# Define all names to be imported
__all__: list[str] = [
    "GaussianDistribution",
]
