# Libraries
from typing import Any, Optional

from illia.distributions.static import StaticDistribution


class GaussianDistribution(StaticDistribution):

    def __init__(
        self,
        mu: float,
        std: float,
        backend: Optional[str] = "torch",
    ) -> None:
        r"""
        Initialize a Gaussian distribution with given mean ($\mu$) and standard deviation ($\sigma$).

        Args:
                mu (float): The mean of the Gaussian distribution.
                std (float): The standard deviation of the Gaussian distribution.
                backend (Optional[str]): The backend library to use for the distribution.

        Raises:
                ValueError: If an invalid backend value is provided.
        """

        # Set attributes
        self.backend = backend

        # Choose backend
        if backend == "torch":
            # Import torch part
            from illia.distributions.static.torch import gaussian  # type: ignore
        elif backend == "tf":
            # Import tensorflow part
            from illia.distributions.static.tf import gaussian  # type: ignore
        else:
            raise ValueError("Invalid backend value")

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
            x (Optional[Tensor]): Input data for which the log PDF needs to be calculated.

        Returns:
            output (Tensor): The log probability density function (PDF) of the input data or sample.
        """

        return self.distribution.log_prob(x)
