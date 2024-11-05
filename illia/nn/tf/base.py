# Libraries
from abc import abstractmethod
from typing import Tuple

from tensorflow import Tensor
from tensorflow.keras.layers import Layer

from illia.nn.base import BayesianModule


class BayesianModule(BayesianModule, Layer):

    def __init__(self, *args, **kwargs):

        # Call super class constructor
        super(BayesianModule, self).__init__(*args, **kwargs)
        
        # Set freeze false by default
        self.frozen = False

    def freeze(self):

        # Set frozen indicator to true for current layer
        self.frozen = True

        # Set forzen indicator to true for children
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.freeze()

    def unfreeze(self):

        # Set frozen indicator to false for current layer
        self.frozen = False

        # Set frozen indicators to false for children
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.unfreeze()

    @abstractmethod
    def kl_cost(self) -> Tuple[Tensor, int]:
        pass
