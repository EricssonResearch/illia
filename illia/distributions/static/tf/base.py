# Libraries
from abc import abstractmethod
from typing import Dict

import tensorflow as tf

import illia.distributions.static as static


class StaticDistribution(static.StaticDistribution):

    @abstractmethod
    def __init__(self, parameters: Dict[str, float]) -> None:
        pass

    @abstractmethod
    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        pass
