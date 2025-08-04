"""
This module contains the code for the gaussian distribution.

Defines a learnable diagonal Gaussian distribution for use in
Bayesian models using PyTorch, including sampling and log-prob
estimation.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch

# Own modules
from illia.distributions.torch.base import DistributionModule


class GaussianDistribution(DistributionModule):
    """
    Implements a learnable Gaussian distribution using PyTorch.

    Parameters are the mean and a softplus-transformed standard
    deviation. Provides methods for sampling and computing the
    KL divergence via `log_prob`.
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
        Initializes the Gaussian distribution with priors and initial
        values.

        Args:
            shape: Shape of the learnable parameters.
            mu_prior: Mean of the Gaussian prior.
            std_prior: Standard deviation of the prior.
            mu_init: Initial value for the mean parameter.
            rho_init: Initial value for the rho parameter.
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
        Draws a sample from the distribution using reparameterization.

        Returns:
            A sample tensor with the same shape as `mu` and `rho`.
        """

        # Sampling with reparametrization trick
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the KL divergence between posterior and prior.

        If no sample is given, one is drawn from the distribution.

        Args:
            x: Optional sample tensor. If None, generates a new sample.

        Returns:
            A scalar tensor representing the KL divergence.
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
        Returns the number of learnable parameters in the distribution.

        Returns:
            Total number of parameters as an integer.
        """

        return len(self.mu.view(-1))
