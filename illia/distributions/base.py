# standard libraries
from abc import ABC, abstractmethod
from typing import Optional, Any


class Distribution(ABC):
    @abstractmethod
    def sample(self, **kwargs) -> Any:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[Any] = None) -> Any:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
