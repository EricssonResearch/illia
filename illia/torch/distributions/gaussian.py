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
    This class implements a gaussian distribution.

    Attributes:
        mu_prior: Initial value for the mu.
        std_prior: Initial value for the std.
        mu: Trainable parameter for the posterior mu.
        std: Trainable parameter for the posterior std.
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
        This class is the constructor for GaussianDistribution.

        Args:
            shape: shape of the distribution.
            mu_prior: mu for the prior distribution. Defaults to 0.0.
            std_prior: std for the prior distribution. Defaults to 0.1.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here. Defaults to 0.0.
            rho_init: init value for rho. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here. Defaults to -7.0.
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

        return None

    # Overriding method
    @torch.jit.export
    def sample(self) -> torch.Tensor:
        """
        This method samples a tensor from the distribution.

        Returns:
            Sampled tensor. Dimensions: [*] (same ones as the mu and
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
            x: Output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution. Defaults to None.

        Returns:
            Log probs. Dimensions: [].
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
