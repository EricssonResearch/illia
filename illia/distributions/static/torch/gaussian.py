# Libraries
import math

import torch

from illia.distributions.static.base import StaticDistribution


class GaussianDistribution(StaticDistribution):

    def __init__(self, mu: float, std: float) -> None:
        """
        This method is the constructor of the class.

        Args:
            mu: mu parameter.
            std: standard deviation parameter.
        """

        # Create Gaussian Distribution that inherits from GaussianDistribution
        super().__init__()

        # Set attributes
        self.mu = torch.tensor(mu)
        self.std = torch.tensor(std)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        self.mu = self.mu.to(x.device)
        self.std = self.std.to(x.device)

        log_probs = (
            -torch.log(torch.sqrt(2 * torch.tensor(math.pi))).to(x.device)
            - torch.log(self.std)
            - (((x - self.mu) ** 2) / (2 * self.std**2))
            - 0.5
        )

        return log_probs.sum()
