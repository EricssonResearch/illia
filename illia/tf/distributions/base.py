# standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pp
import tensorflow as tf


class Distribution(ABC, tf.keras.Model):
    @abstractmethod
    def sample(self) -> tf.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
