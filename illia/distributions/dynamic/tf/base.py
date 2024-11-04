# Libraries
from abc import abstractmethod
from typing import Optional

import tensorflow as tf

import illia.distributions.dynamic as dynamic


class DynamicDistribution(dynamic.base.DynamicDistribution, tf.keras.Model):

    @abstractmethod
    def sample(self) -> tf.Variable:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor]) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
