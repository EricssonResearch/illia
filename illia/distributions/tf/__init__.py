"""
This module consolidates and exposes distribution-related classes
implemented in Tensorflow. It imports core base classes and specific
distribution implementations for easier access in other modules.
"""

# Own modules
from illia.distributions.tf.base import DistributionModule
from illia.distributions.tf.gaussian import GaussianDistribution


# Define all names to be imported
__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
