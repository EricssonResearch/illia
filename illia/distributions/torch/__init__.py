"""
This module defines the imports for distribution functions in PyTorch.
"""

# Own modules
from illia.distributions.torch.base import DistributionModule
from illia.distributions.torch.gaussian import GaussianDistribution

# Define all names to be imported
__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
