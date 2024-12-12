# standard libraries
from typing import Optional

# 3pp
import torch

# own modules
from illia.torch.distributions.base import (
    Distribution,
)


class GaussianDistribution(Distribution):
    # overriding method
    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        This class is the constructor for GaussianDistribution.

        Args:
            shape: _description_
            mu_prior: _description_. Defaults to 0.0.
            std_prior: _description_. Defaults to 0.1.
            mu_init: _description_. Defaults to 0.0.
            rho_init: _description_. Defaults to -7.0.
        """

        # call super-class constructor
        super().__init__()

        # define priors
        self.mu_prior: torch.Tensor = torch.tensor([mu_prior])
        self.std_prior: torch.Tensor = torch.tensor([std_prior])

        # define initial mu and rho
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(rho_init, 0.1)
        )

    # overriding method
    @torch.jit.export
    def sample(self) -> torch.Tensor:
        """
        This method samples a tensor from the distribution.

        Returns:
            sampled tensor. Dimensions: [*] (same ones as the mu and
                std parameters).
        """

        # sampling with reparametrization trick
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    # overriding method
    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function computes the log probability.

        Args:
            x: output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution. Defaults to None.

        Returns:
            log probs. Dimensions: [].
        """

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
