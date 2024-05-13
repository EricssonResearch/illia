# deep learning libraries
import torch

# other libraries
from abc import abstractmethod
from typing import Optional

# own modules
import illia.distributions.dynamic as dynamic


class DynamicDistribution(dynamic.base.DynamicDistribution, torch.nn.Module):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
