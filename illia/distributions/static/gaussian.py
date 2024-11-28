# Libraries
from typing import Any

from illia.distributions.static import StaticDistribution

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from illia.distributions.static.torch import gaussian
elif backend() == "tf":
    from illia.distributions.static.tf import gaussian


class GaussianDistribution(StaticDistribution):

    def __init__(
        self,
        mu: float,
        std: float,
    ) -> None:
        r"""
        Initialize a Gaussian distribution with given mean ($\mu$) and standard deviation ($\sigma$).

        Args:
            mu: The mean of the Gaussian distribution.
            std: The standard deviation of the Gaussian distribution.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super().__init__()

        # Define distribution based on the imported library
        self.distribution = gaussian.GaussianDistribution(mu=mu, std=std)

    # Overriding method
    def log_prob(self, x: Any) -> Any:
        r"""
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters $\mu$ and $\rho$, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x: Input data for which the log PDF needs to be calculated.

        Returns:
            The log probability density function (PDF) of the input data or sample.
        """

        return self.distribution.log_prob(x)
