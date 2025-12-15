"""
This module provides a unified interface for distribution-related
classes implemented in PyTorch. It imports the fundamental base class
along with specific distribution implementations, facilitating their
accessibility and use in other modules.
"""

# Own modules
from illia.distributions.torch.base import DistributionModule
from illia.distributions.torch.gaussian import GaussianDistribution


__all__: list[str] = ["DistributionModule", "GaussianDistribution"]
