# Libraries
import math

import torch

from illia.distributions.static.base import StaticDistribution


class GaussianDistribution(StaticDistribution):

    def __init__(self, mu: float, std: float) -> None:
        """
        Initialize a Gaussian distribution with given mean (mu) and standard deviation (std).

        Args:
                mu (float): The mean of the Gaussian distribution.
                std (float): The standard deviation of the Gaussian distribution.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.mu = torch.tensor(mu)
        self.std = torch.tensor(std)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters mu and rho, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x (Optional[torch.Tensor]): Input data for which the log PDF needs to be calculated.

        Returns:
            output (torch.Tensor): The log probability density function (PDF) of the input data or sample.
        """

        self.mu = self.mu.to(x.device)
        self.std = self.std.to(x.device)

        log_probs = (
            -torch.log(torch.sqrt(2 * torch.tensor(math.pi))).to(x.device)
            - torch.log(self.std)
            - (((x - self.mu) ** 2) / (2 * self.std**2))
            - 0.5
        )

        return log_probs.sum()
