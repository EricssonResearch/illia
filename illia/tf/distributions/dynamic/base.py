# Libraries
from abc import abstractmethod
from typing import Any

from keras import Model


class DynamicDistribution(Model):
    """
    A base class for creating a Dynamic distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.

    Methods:
        sample(): Generate a sample from the distribution.
        log_prob(x): Compute the log probability of a given observation.

    Properties:
        num_params: Retrieve the number of parameters in the
            distribution.
    """

    @abstractmethod
    def sample(self) -> Any:
        """
        Generate a sample from the distribution.
        """

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        """
        Compute the log probability of a given observation.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Retrieve the number of parameters in the distribution.
        """
