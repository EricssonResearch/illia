# Libraries
from abc import abstractmethod
from typing import Any

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from torch.nn import Module as BackendModule
elif backend() == "tf":
    from tensorflow.keras import Model as BackendModule


class DynamicDistribution(BackendModule):
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
