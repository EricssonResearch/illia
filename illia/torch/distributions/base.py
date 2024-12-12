# deep learning libraries
import torch

# other libraries
from abc import ABC, abstractmethod
from typing import Optional


class Distribution(ABC, torch.nn.Module):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
