# Libraries
from typing import Tuple, Optional, Any

from illia.distributions.dynamic import DynamicDistribution


class GaussianDistribution(DynamicDistribution):
    """
    A base class for creating a Dynamic Gaussian distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Initializes a Gaussian distribution parameter set.

        Args:
            shape (Tuple[int, ...]): The shape of the parameter tensors.
            mu_init (float, optional): The initial value for mu. Defaults to 0.0.
            rho_init (float, optional): The initial value for rho. Defaults to -7.0.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

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
        return self.distribution.sample()

    # Overriding method
    def log_prob(self, x: Optional[Any]) -> Any:
        return self.distribution.log_prob(x)

    @property
    def num_params(self) -> int:
        return self.distribution.num_params
