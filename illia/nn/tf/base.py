# Libraries
from abc import ABC, abstractmethod
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers


class BayesianModule(ABC, layers.Layer):

    def __init__(self, *args, **kwargs):
        super(BayesianModule, self).__init__(*args, **kwargs)
        self.frozen = False

    def freeze(self):
        self.frozen = True
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.freeze()

    def unfreeze(self):
        self.frozen = False
        for layer in self.submodules:
            if isinstance(layer, BayesianModule):
                layer.unfreeze()

    @abstractmethod
    def kl_cost(self) -> Tuple[tf.Tensor, int]:
        pass
