# deep learning libraries
import torch

# other libraries
from abc import ABC, abstractmethod
from typing import Dict

# own modules
import illia.distributions.static as static


class StaticDistribution(static.StaticDistribution):
    @abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pass
