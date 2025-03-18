# Libraries
from abc import abstractmethod
from typing import Any

from keras import Model


class StaticDistribution(Model):
    """
    A base class for creating a Static distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.

    Methods:
        log_prob(x): Compute the log probability of a given observation.
    """

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        """
        Compute the log probability of a given observation.
        """
