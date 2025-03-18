# Libraries
import math
from typing import Tuple

import torch

from . import DynamicDistribution


class GaussianDistribution(DynamicDistribution):
    """
    Gaussian distribution with trainable parameters for mean (mu) and
    standard deviation (derived from rho).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initializes a Gaussian distribution with trainable parameters
        mu and rho. Parameters are initialized with normal
        distributions.

        Args:
            shape: Shape of the distribution parameters.
            mu_init: Initial mean value for mu.
            rho_init: Initial mean value for rho.
        """

        # Call super class constructor
        super().__init__()

        # Parameter for standard deviation
        self.std = 0.1

        # Set attributes
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(self.shape).normal_(mean=self.mu_init, std=self.std)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(self.shape).normal_(mean=self.rho_init, std=self.std)
        )

    def sample(self) -> torch.Tensor:
        """
        Generates a sample from the Gaussian distribution using current
        parameters mu and rho.

        Returns:
            Tensor representing a sample from the Gaussian distribution.
        """

        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability density of the given input data
        based on current parameters.

        Args:
            x: Input data for which to calculate the log PDF.

        Returns:
            Log probability density of the input data.
        """

        if x is None:
            x = self.sample()

        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        log_posteriors: torch.Tensor = (
            -torch.log(torch.sqrt(2 * torch.tensor(math.pi)))
            - torch.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        return log_posteriors.sum()

    @property
    def num_params(self) -> int:
        """
        Returns the total number of parameters in the Gaussian
        distribution.

        Returns:
            Total number of parameters.
        """

        return len(self.mu.view(-1))
