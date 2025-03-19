from typing import Optional

import torch

from .base import Distribution


class GaussianDistribution(Distribution):
    """
    This is the class to implement a learnable gausssian distribution.
    """

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
        Initializes the GaussianDistribution with given priors and
        initial parameters.

        Args:
            shape: The shape of the parameters.
            mu_prior: The mean prior value.
            std_prior: The standard deviation prior value.
            mu_init: The initial mean value.
            rho_init: The initial rho value, which affects the initial
                standard deviation.
        """

        # Call super-class constructor
        super().__init__()

        # Define priors
        self.mu_prior: torch.Tensor = torch.tensor([mu_prior])
        self.std_prior: torch.Tensor = torch.tensor([std_prior])

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
        Samples from the distribution using the current parameters.

        Args:
            seed: A random seed for generating the sample.

        Returns:
            A sampled tensor.
        """

        # sampling with reparametrization trick
        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    @torch.jit.export
    def log_prob(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sampled array. If None, a sample is
                generated.

        Returns:
            The log probability of the sample as a tensor.
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

    @property
    @torch.jit.export
    @torch.no_grad()
    def num_params(self) -> int:
        """
        Returns the number of parameters in the module.

        Returns:
            The number of parameters as an integer.
        """

        return len(self.mu.view(-1))
