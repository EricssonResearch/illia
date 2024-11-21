# Libraries
import math
from typing import Tuple, Optional

import torch

from illia.distributions.dynamic.base import DynamicDistribution


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initialize a Gaussian Distribution object with trainable parameters mu and rho.
        The parameters are initialized with a normal distribution with mean mu_init and rho_init,
        and a standard deviation of 0.1.

        Args:
            shape (Tuple[int, ...]): The shape of the distribution parameters.
            mu_init (float): The initial mean value for mu.
            rho_init (float): The initial mean value for rho.
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
        Generate a sample from the Gaussian distribution using the current parameters
        mu and rho. The sample is obtained by adding a random noise (epsilon) to the mean (mu),
        where the noise is scaled by the standard deviation (sigma).

        Args:
            self (GaussianDistribution): The instance of the Gaussian Distribution object.

        Returns:
            torch.Tensor: A tensor representing a sample from the Gaussian distribution.
        """

        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters mu and rho, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x (Optional[torch.Tensor]): Input data for which the log PDF needs to be calculated.
                                If None, a sample is generated using the current parameters.

        Returns:
            torch.Tensor: The log probability density function (PDF) of the input data or sample.
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
        Calculate the total number of parameters in the Gaussian Distribution, which is the product
        of the dimensions of the mean (mu) parameter.

        Args:
            self (GaussianDistribution): The instance of the Gaussian Distribution object.

        Returns:
            int: The total number of parameters in the Gaussian Distribution.
        """

        return len(self.mu.view(-1))
