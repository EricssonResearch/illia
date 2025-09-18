# Standard libraries
from typing import Any, Optional

# 3pps
import torch

# Own modules
from illia.distributions.torch.base import DistributionModule


class GaussianDistribution(DistributionModule):
    """
    Learnable Gaussian distribution with diagonal covariance.
    Represents a Gaussian with trainable mean and standard
    deviation. The standard deviation is derived from `rho`
    using a softplus transformation to ensure positivity.

    Notes:
        Assumes diagonal covariance. KL divergence can be
        estimated via log-probability differences from
        `log_prob`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a learnable Gaussian distribution module.

        Args:
            shape: Shape of the learnable parameters.
            mu_prior: Mean of the Gaussian prior.
            std_prior: Standard deviation of the prior.
            mu_init: Initial value for the learnable mean.
            rho_init: Initial value for the learnable rho.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None
        """

        # Call super-class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init

        # Define priors
        self.register_buffer("mu_prior", torch.tensor([mu_prior]))
        self.register_buffer("std_prior", torch.tensor([std_prior]))

        # Define initial mu and rho
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(self.shape).normal_(self.mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(self.shape).normal_(self.rho_init, 0.1)
        )

    @torch.jit.export
    def sample(self) -> torch.Tensor:
        """
        Draw a sample from the Gaussian distribution.

        Returns:
            torch.Tensor: A sample drawn from the distribution.
        """

        # Sampling with reparametrization trick
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the log-probability of a given sample. If no
        sample is provided, one is drawn internally.

        Args:
            x: Optional input sample to evaluate. If None,
                a new sample is drawn from the distribution.

        Returns:
            torch.Tensor: Scalar log-probability value.

        Notes:
            Supports both user-supplied and internally drawn
            samples.
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
        Return the number of learnable parameters in the
        distribution.

        Returns:
            int: Total count of learnable parameters.
        """

        return len(self.mu.view(-1))
