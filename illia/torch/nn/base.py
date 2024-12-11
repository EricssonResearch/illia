# Libraries
from abc import abstractmethod
from typing import Tuple, Any

from torch.nn import Module


class BayesianModule(Module):
    """
    A base class for creating a Bayesian Module with the torch backend.
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

        # Set frozen indicator to true for children
        for module in self.modules():
            if self != module and isinstance(module, BayesianModule):
                module.freeze()
            else:
                continue

    def unfreeze(self) -> None:
        # Set frozen indicator to false for current layer
        self.frozen = False

        # Set frozen indicators to false for children
        for module in self.modules():
            if self != module and isinstance(module, BayesianModule):
                module.unfreeze()
            else:
                continue

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
