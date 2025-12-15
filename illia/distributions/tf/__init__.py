"""
This module provides a unified interface for distribution-related
classes implemented in Tensorflow. It imports the fundamental base class
along with specific distribution implementations, facilitating their
accessibility and use in other modules.
"""

# Own modules
from illia.distributions.tf.base import DistributionModule
from illia.distributions.tf.gaussian import GaussianDistribution


__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
