# Libraries
from typing import Any, Optional

from illia.distributions.static import StaticDistribution


class GaussianDistribution(StaticDistribution):
    """
    A base class for creating a Static Gaussian distribution.
    Each function in this class is intended to be overridden by specific
    backend implementations.
    """

    def __init__(
        self,
        mu: float,
        std: float,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Initializes a Gaussian distribution parameter set.

        Args:
            mu (float, optional): The value for mu.
            std (float, optional): The value for std.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

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
        return self.distribution.log_prob(x)
