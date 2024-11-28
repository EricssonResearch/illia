# Libraries
from typing import Tuple, Any

from illia.distributions.dynamic import DynamicDistribution

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from illia.distributions.dynamic.torch import gaussian
elif backend() == "tf":
    from illia.distributions.dynamic.tf import gaussian


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        r"""
        Initialize a Gaussian Distribution object with a specified backend and with trainable parameters $\mu$ and $\rho$.
        The parameters are initialized with a normal distribution with mean mu_init and rho_init,
        and a standard deviation of 0.1.

        Args:
            shape: The shape of the distribution parameters.
            mu_init: The initial mean value for $\mu$.
            rho_init: The initial mean value for $\rho$.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super().__init__()

        # Define distribution based on the imported library
        self.distribution = gaussian.GaussianDistribution(
            shape=shape, mu_init=mu_init, rho_init=rho_init
        )

    # Overriding method
    def sample(self) -> Any:
        r"""
        Generate a sample from the Gaussian distribution using the current parameters
        $\mu$ and $\rho$. The sample is obtained by adding a random noise ($\epsilon$) to the mean ($\mu$),
        where the noise is scaled by the standard deviation ($\sigma$).

        Returns:
            A tensor representing a sample from the Gaussian distribution.
        """

        return self.distribution.sample()

    # Overriding method
    def log_prob(self, x: Any) -> Any:
        r"""
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters $\mu$ and $\rho$, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x: Input data for which the log PDF needs to be calculated.
                If None, a sample is generated using the current parameters.

        Returns:
            The log probability density function (PDF) of the input data or sample.
        """

        return self.distribution.log_prob(x)

    @property
    def num_params(self) -> int:
        r"""
        Calculate the total number of parameters in the Gaussian Distribution, which is the product
        of the dimensions of the mean ($\mu$) parameter.

        Returns:
            The total number of parameters in the Gaussian Distribution.
        """

        return self.distribution.num_params
