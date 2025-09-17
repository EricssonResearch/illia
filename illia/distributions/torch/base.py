# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import torch


class DistributionModule(torch.nn.Module, ABC):
    """
    Abstract base for PyTorch-based probabilistic distribution
    modules. Defines the interface for sampling, computing
    log-probabilities, and retrieving parameter counts. Subclasses must
    implement all abstract methods to provide specific distribution
    logic.

    Notes:
        This class is abstract and should not be instantiated directly.
        All abstract methods must be implemented by subclasses.
    """

    @abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Generate a sample from the underlying distribution.

        Returns:
            Array containing a sample matching the distribution shape.
        """

    @abstractmethod
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the log-probability of a given sample. If no sample is
        provided, a new one is drawn internally from the distribution.

        Args:
            x: Optional sample tensor to evaluate.

        Returns:
            Scalar array containing the log-probability.

        Notes:
            Supports both user-supplied and internally generated
                samples.
        """

    @abstractmethod
    def num_params(self) -> int:
        """
        Return the total number of learnable parameters in the
        distribution.

        Returns:
            Integer count of all learnable parameters.
        """
