# deep larning libraries
import torch

# other libraries
from typing import Tuple, Optional

# own modules
from ..base import (
    Distribution,
)


class GaussianDistribution(Distribution):
    # overriding method
    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        # call super-class constructor
        super().__init__()

        # define priors
        self.mu_prior: torch.Tensor = torch.tensor([mu_prior],dtype=torch.float32)
        self.std_prior: torch.Tensor = torch.tensor([std_prior],dtype=torch.float32)

        # define initial mu and rho
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape,dtype=torch.float32).normal_(mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape,dtype=torch.float32).normal_(rho_init, 0.1)
        )

    # overriding method
    @torch.jit.export
    def sample(self) -> torch.Tensor:
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    # overriding method
    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sample if x is None
        if x is None:
            x = self.sample()

        # define pi variable
        pi: torch.Tensor = torch.acos(torch.zeros(1)) * 2

        # compute log priors
        log_prior = (
            -torch.log(torch.sqrt(2 * pi)).to(x.device)
            - torch.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # compute sigma
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho)).to(x.device)

        # compute log posteriors
        log_posteriors = (
            -torch.log(torch.sqrt(2 * pi)).to(x.device)
            - torch.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # compute final log probs
        log_probs = log_posteriors.sum() - log_prior.sum()

        return log_probs

    @property
    @torch.jit.export
    @torch.no_grad()
    def num_params(self) -> int:
        return len(self.mu.view(-1))
