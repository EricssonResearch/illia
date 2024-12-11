# Libraries
from abc import abstractmethod
from typing import Any

from torch.nn import Module


class StaticDistribution(Module):
    """
    A base class for creating a Static distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.
    """

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        pass
