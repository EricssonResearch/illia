# other libraries
from abc import ABC, abstractmethod
from typing import Dict, Any


class StaticDistribution(ABC):
    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        pass
