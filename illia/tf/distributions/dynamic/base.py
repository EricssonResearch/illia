# Libraries
from abc import abstractmethod
from typing import Any

from tensorflow.keras import Model  # type: ignore


class DynamicDistribution(Model):
    """
    A base class for creating a Dynamic distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.
    """

    @abstractmethod
    def sample(self) -> Any:
        pass

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
