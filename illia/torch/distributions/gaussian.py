"""
This module contains the code for the gaussian distribution.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch

# Own modules
from illia.torch.distributions.base import Distribution


class GaussianDistribution(Distribution):
    """
    This is the class to implement a learnable Gaussian distribution.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Constructor for GaussianDistribution.

        Args:
            shape: Shape of the distribution.
            mu_prior: Mean for the prior distribution.
            std_prior: Standard deviation for the prior distribution.
            mu_init: Initial mean for mu.
            rho_init: Initial mean for rho.
        """

        # Call super-class constructor
        super().__init__()

        # Define priors
        self.register_buffer("mu_prior", torch.tensor([mu_prior]))
        self.register_buffer("std_prior", torch.tensor([std_prior]))

        # Define initial mu and rho
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(rho_init, 0.1)
        )

    @torch.jit.export
    def sample(self) -> torch.Tensor:
        """
        This method samples a tensor from the distribution.

        Returns:
            Sampled tensor. Dimensions: [*] (same ones as the mu and
                std parameters).
        """

        # Sampling with reparametrization trick
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This method computes the log prob of the distribution.

        Args:
            x: Output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution.

        Returns:
            Log prob calculated as a tensor. Dimensions: [].
        """

        # Sample if x is None
        if x is None:
            x = self.sample()

        # Define pi variable
        pi: torch.Tensor = torch.acos(torch.zeros(1)) * 2

        # Compute log priors
        log_prior = (
            -torch.log(torch.sqrt(2 * pi)).to(x.device)
            - torch.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # Compute sigma
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho)).to(x.device)

        # Compute log posteriors
        log_posteriors = (
            -torch.log(torch.sqrt(2 * pi)).to(x.device)
            - torch.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # Compute final log probs
        log_probs = log_posteriors.sum() - log_prior.sum()

        return log_probs

    @torch.jit.export
    @torch.no_grad()
    def num_params(self) -> int:
        """
        This method computes the number of parameters of the
        distribution.

        Returns:
            Number of parameters.
        """

        return len(self.mu.view(-1))
