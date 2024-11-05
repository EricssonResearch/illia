# Libraries
from abc import abstractmethod
from typing import Tuple

from torch import Tensor
from torch.nn import Module

from illia.nn.base import BayesianModule


class BayesianModule(BayesianModule, Module):
    frozen: bool

    def __init__(self) -> None:
        
        # Call super class constructor
        super(BayesianModule, self).__init__()

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
            if module != self and isinstance(module, BayesianModule):
                module.unfreeze()

    @abstractmethod
    def kl_cost(self) -> Tuple[Tensor, int]:
        pass
