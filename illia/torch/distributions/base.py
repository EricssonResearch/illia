"""
This module contains the base class for the Distributions.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import torch


class Distribution(ABC, torch.nn.Module):
    """
    This class is the base class for distributions.
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
                distribution. Defaults to None.

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
