# Libraries
from typing import Tuple, Optional

import torch

from illia.distributions.dynamic.base import DynamicDistribution

# static nn.Parameters
PI: torch.Tensor = torch.acos(torch.zeros(1)) * 2


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self, shape: Tuple[int, ...], mu_init: float = 0.0, rho_init: float = -7.0
    ) -> None:
        super().__init__()

        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(rho_init, 0.1)
        )

    def sample(self) -> torch.Tensor:
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if x is None:
            x = self.sample()
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        log_posteriors: torch.Tensor = (
            -torch.log(torch.sqrt(2 * PI))
            - torch.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        return log_posteriors.sum()

    @property
    def num_params(self) -> int:
        return len(self.mu.view(-1))
