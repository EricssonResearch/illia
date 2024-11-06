# Libraries
from abc import ABC, abstractmethod
from typing import Optional, Any


class DynamicDistribution(ABC):
    """
    A base class for creating a Dynamic distribution.
    Each function in this class is intended to be overridden by specific 
    backend implementations.
    """

    @abstractmethod
    def sample(self) -> Any:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[Any]) -> Any:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
