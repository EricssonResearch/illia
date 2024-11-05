# Libraries
from abc import ABC, abstractmethod
from typing import Tuple, Any


class BayesianModule(ABC):
    """
    A base class for creating a Bayesion Module.
    Each of the functions is subsequently override by the specific backend.
    """

    @abstractmethod
    def freeze(self) -> None:
        pass
    
    @abstractmethod
    def unfreeze(self) -> None:
        pass

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
