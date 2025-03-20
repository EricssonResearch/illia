"""
This module defines the imports for illia.torch.distributions.
"""

# Own modules
from illia.torch.distributions.base import Distribution
from illia.torch.distributions.gaussian import GaussianDistribution

# Define all names to be imported
__all__: list[str] = ["Distribution", "GaussianDistribution"]
