# Libraries
from abc import abstractmethod
from typing import Optional

from torch import Tensor
from torch.nn import Module

import illia.distributions.dynamic as dynamic


class DynamicDistribution(dynamic.DynamicDistribution, Module):

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
