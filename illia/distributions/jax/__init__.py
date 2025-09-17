"""
This module provides a unified interface for distribution-related
classes implemented in JAX. It imports the fundamental base class along
with specific distribution implementations, facilitating their
accessibility and use in other modules.
"""

# Own modules
from illia.distributions.jax.base import DistributionModule
from illia.distributions.jax.gaussian import GaussianDistribution


# Define all names to be imported
__all__: list[str] = [
    "DistributionModule",
    "GaussianDistribution",
]
