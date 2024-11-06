# Libraries
from typing import Tuple, Optional

import torch

from illia.distributions.dynamic.base import DynamicDistribution

# Static nn.Parameters
PI: torch.Tensor = torch.acos(torch.zeros(1)) * 2


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self, shape: Tuple[int, ...], mu_init: float = 0.0, rho_init: float = -7.0
    ) -> None:
        """
        Initializes a set of parameters for a distribution.

        Args:
            shape (Tuple[int, ...]): The shape of the parameter tensors.
            mu_init (float, optional): The initial value for the parameter mu. Defaults to 0.0.
            rho_init (float, optional): The initial value for the parameter rho. Defaults to -7.0.
        """

        # Call super class constructor
        super(GaussianDistribution, self).__init__()

        # Set attributes
        self.mu: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(mu_init, 0.1)
        )
        self.rho: torch.Tensor = torch.nn.Parameter(
            torch.randn(shape).normal_(rho_init, 0.1)
        )

    def sample(self) -> torch.Tensor:
        """
        Samples a tensor from the distribution using the reparameterization trick.

        Returns:
            torch.Tensor: A sample tensor from the distribution.
        """

        eps: torch.Tensor = torch.randn_like(self.rho)
        sigma: torch.Tensor = torch.log1p(torch.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Computes the log probability of a given tensor under the distribution.

        Args:
            x (Optional[torch.Tensor]): The tensor for which to compute the log probability.
                                        If None, a sample is drawn from the distribution.

        Returns:
            torch.Tensor: The log probability of the tensor under the distribution.
        """

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
        """
        Returns the number of parameters in the distribution.

        This is calculated as the total number of elements in the tensor representing mu.

        Returns:
            int: The total number of parameters.
        """
        
        return len(self.mu.view(-1))