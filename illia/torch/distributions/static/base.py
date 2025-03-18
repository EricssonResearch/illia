# Libraries
from abc import abstractmethod
from typing import Any

from torch.nn import Module


class StaticDistribution(Module):
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
