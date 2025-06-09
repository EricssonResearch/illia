"""
This module contains the base class for the Distributions.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class DistributionModule(torch.nn.Module, ABC):
    """
    This class serves as the base class for Distributions modules.
    Any module designed to function as a distribution layer should
    inherit from this class.
    """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        This method samples a tensor from the distribution.

        Returns:
            Sampled tensor. Dimensions: [*] (same ones as the mu and
                std parameters).
        """

    @abstractmethod
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This method computes the log prob of the distribution.

        Args:
            x: Output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution.

        Returns:
            Log prob calculated as a tensor. Dimensions: [].
        """

    @abstractmethod
    def num_params(self) -> int:
        """
        This method computes the number of parameters of the
        distribution.

        Returns:
            Number of parameters.
        """
