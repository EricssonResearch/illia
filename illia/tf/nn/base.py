# Libraries
from abc import abstractmethod
from typing import Tuple, Any

from tensorflow.keras import Model  # type: ignore


class BayesianModule(Model):
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

        # Set forzen indicator to true for children
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.freeze()
            else:
                continue

    def unfreeze(self) -> None:

        # Set frozen indicator to false for current layer
        self.frozen = False

        # Set frozen indicators to false for children
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.unfreeze()
            else:
                continue

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
