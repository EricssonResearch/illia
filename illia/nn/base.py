# Libraries
from abc import abstractmethod
from typing import Tuple, Any

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from torch.nn import Module as BackendModule
elif backend() == "tf":
    from tensorflow.keras import Model as BackendModule


class BayesianModule(BackendModule):
    """
    A base class for creating a Bayesion Module.
    Each of the functions is subsequently override by the specific backend.
    """

    frozen: bool

    def __init__(self):

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen = False

    def freeze(self) -> None:

        # Set frozen indicator to true for current layer
        self.frozen = True

        if backend() == "torch":
            # Set frozen indicator to true for children
            for module in self.modules():
                if self != module and isinstance(module, BayesianModule):
                    module.freeze()
                else:
                    continue
        elif backend() == "tf":
            # Set forzen indicator to true for children
            for layer in self.submodules:
                if isinstance(layer, BayesianModule):
                    layer.freeze()
                else:
                    continue

    def unfreeze(self) -> None:

        # Set frozen indicator to false for current layer
        self.frozen = False

        if backend() == "torch":
            # Set frozen indicators to false for children
            for module in self.modules():
                if module != self and isinstance(module, BayesianModule):
                    module.unfreeze()
        elif backend() == "tf":
            # Set frozen indicators to false for children
            for layer in self.submodules:
                if isinstance(layer, BayesianModule):
                    layer.unfreeze()

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
