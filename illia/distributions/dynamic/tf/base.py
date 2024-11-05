# Libraries
from abc import abstractmethod
from typing import Optional

from tensorflow import Variable, Tensor
from tensorflow.keras import Model

from illia.distributions.dynamic.base import DynamicDistribution


class DynamicDistribution(DynamicDistribution, Model):

    @abstractmethod
    def sample(self) -> Variable:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[Tensor]) -> Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
