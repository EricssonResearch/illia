# Libraries
from abc import abstractmethod
from typing import Any

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from torch.nn import Module as BackendModule
elif backend() == "tf":
    from tensorflow.keras import Model as BackendModule


class StaticDistribution(BackendModule):
    """
    A base class for creating a Static distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.
    """

    @abstractmethod
    def log_prob(self, x: Any) -> Any:
        pass
