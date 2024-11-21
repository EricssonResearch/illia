# Libraries
from typing import Tuple, Optional, Any

from illia.distributions.dynamic import DynamicDistribution


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        backend: Optional[str] = "torch",
    ) -> None:
        r"""
        Initialize a Gaussian Distribution object with a specified backend and with trainable parameters $\mu$ and $\rho$.
        The parameters are initialized with a normal distribution with mean mu_init and rho_init,
        and a standard deviation of 0.1.

        Args:
            shape (Tuple[int, ...]): The shape of the distribution parameters.
            mu_init (float): The initial mean value for $\mu$.
            rho_init (float): The initial mean value for $\rho$.
            backend (str): The backend library to use for the distribution.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Set attributes
        self.backend = backend

        # Choose backend
        if backend == "torch":
            # Import torch part
            from illia.distributions.dynamic.torch import gaussian  # type: ignore
        elif backend == "tf":
            # Import tensorflow part
            from illia.distributions.dynamic.tf import gaussian  # type: ignore
        else:
            raise ValueError("Invalid backend value")

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

        Args:
            self (GaussianDistribution): The instance of the Gaussian Distribution object.

        Returns:
            A tensor representing a sample from the Gaussian distribution.
        """

        return self.distribution.sample()

    # Overriding method
    def log_prob(self, x: Optional[Any]) -> Any:
        r"""
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters $\mu$ and $\rho$, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x (Optional[Tensor]): Input data for which the log PDF needs to be calculated.
                                    If None, a sample is generated using the current parameters.

        Returns:
            output (Tensor): The log probability density function (PDF) of the input data or sample.
        """

        return self.distribution.log_prob(x)

    @property
    def num_params(self) -> int:
        r"""
        Calculate the total number of parameters in the Gaussian Distribution, which is the product
        of the dimensions of the mean ($\mu$) parameter.

        Args:
            self (GaussianDistribution): The instance of the Gaussian Distribution object.

        Returns:
            output (int): The total number of parameters in the Gaussian Distribution.
        """

        return self.distribution.num_params
