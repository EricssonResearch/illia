# Libraries
from abc import abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module

from illia.distributions.dynamic.base import DynamicDistribution


class DynamicDistribution(DynamicDistribution, Module):

    @abstractmethod
    def sample(self) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[Tensor]) -> Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
