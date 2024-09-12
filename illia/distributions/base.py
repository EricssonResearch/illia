# deep learning libraries
import torch

# other libraries
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any


class Distribution(ABC, torch.nn.Module):
    @abstractmethod
    def sample(self) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob(self, x: Optional[Any] = None) -> Any:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
