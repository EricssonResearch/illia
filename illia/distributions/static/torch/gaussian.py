# deep larning libraries
import torch

# other libraries
from typing import Dict

# own modules
from illia.distributions.static import StaticDistribution

# static variables
PI: torch.Tensor = torch.acos(torch.zeros(1)) * 2


class GaussianDistribution(StaticDistribution):
    # overriding method
    def __init__(self, mu: float, std: float) -> None:
        self.mu = torch.Tensor(mu)
        self.std = torch.Tensor(std)

    # overriding method
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        self.mu = self.mu.to(x.device)
        self.std = self.std.to(x.device)

        log_probs = (
            -torch.log(torch.sqrt(2 * PI)).to(x.device)
            - torch.log(self.std)
            - (((x - self.mu) ** 2) / (2 * self.std**2))
            - 0.5
        )

        return log_probs.sum()
