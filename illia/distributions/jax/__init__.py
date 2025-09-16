"""
This module consolidates and exposes distribution-related classes
implemented in JAX. It imports core base classes and specific
distribution implementations for easier access in other modules.
"""

# Own modules
from illia.distributions.jax.base import DistributionModule
from illia.distributions.jax.gaussian import GaussianDistribution


# Define all names to be imported
__all__: list[str] = [
    "DistributionModule",
    "GaussianDistribution",
]
