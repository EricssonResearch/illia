# deep learning libraries
import torch

# other libraries
from abc import ABC, abstractmethod
from typing import Tuple


class BayesianModule(ABC, torch.nn.Module):
    frozen: bool

    def __init__(self) -> None:
        # call super class constructor
        super().__init__()

        # set freeze false by default
        self.frozen = False

    def freeze(self) -> None:
        # set frozen indicator to true for current layer
        self.frozen = True

        # set forzen indicator to true for children
        for module in self.modules():
            if self != module and isinstance(module, BayesianModule):
                module.freeze()
            else:
                continue

    def unfreeze(self) -> None:
        # set frozen indicator to false for current layer
        self.frozen = False

        # set forzen indicators to false for children
        for module in self.modules():
            if module != self and isinstance(module, BayesianModule):
                module.unfreeze()

    @abstractmethod
    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        pass
