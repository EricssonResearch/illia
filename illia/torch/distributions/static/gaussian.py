# Libraries
import math

import torch

from . import StaticDistribution


class GaussianDistribution(StaticDistribution):
    """
    Represents a static Gaussian distribution with specified mean (mu)
    and standard deviation (std).
    """

    def __init__(self, mu: float, std: float) -> None:
        """
        Initializes a Gaussian distribution with given mean and standard
        deviation.

        Args:
            mu: Mean of the Gaussian distribution.
            std: Standard deviation of the Gaussian distribution.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.mu = torch.tensor(mu)
        self.std = torch.tensor(std)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the log probability density function (PDF) of the
        given input data based on the Gaussian distribution's
        parameters.

        Args:
            x: Input data for which the log PDF is calculated.

        Returns:
            Log probability density of the input data.
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
