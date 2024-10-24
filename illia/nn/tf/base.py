# deep learning libraries
import tensorflow as tf
# other libraries
from abc import ABC, abstractmethod
from typing import Tuple

class BayesianModule(ABC, tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
