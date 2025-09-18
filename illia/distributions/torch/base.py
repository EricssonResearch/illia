# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import torch


class DistributionModule(torch.nn.Module, ABC):
    """
    Abstract base for probabilistic distribution modules in PyTorch.
    Defines the required interface for sampling, computing
    log-probabilities, and counting learnable parameters.

    Notes:
        This class is abstract and cannot be instantiated directly.
        All abstract methods must be implemented by subclasses.
    """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Draw a sample from the distribution.

        Returns:
            torch.Tensor: A sample drawn from the distribution.
        """

    @abstractmethod
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the log-probability of a provided sample. If no
        sample is passed, one is drawn internally.

        Args:
            x: Optional sample to evaluate. If None, a new sample is
                drawn from the distribution.

        Returns:
            torch.Tensor: Scalar log-probability value.

        Notes:
            Works with both user-supplied and internally drawn
            samples.
        """

    @abstractmethod
    def num_params(self) -> int:
        """
        Return the number of learnable parameters in the
        distribution.

        Returns:
            int: Total count of learnable parameters.
        """
