"""
This module consolidates and exposes distribution-related classes
implemented in PyTorch. It imports core base classes and specific
distribution implementations for easier access in other modules.
"""

# Own modules
from illia.distributions.torch.base import DistributionModule
from illia.distributions.torch.gaussian import GaussianDistribution


# Define all names to be imported
__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
