# Libraries
from abc import ABC, abstractmethod
from typing import Any


class StaticDistribution(ABC):

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        pass
