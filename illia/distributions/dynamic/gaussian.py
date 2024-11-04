# Libraries
from typing import Tuple, Optional, Any

import illia.distributions.dynamic as dynamic


class GaussianDistribution(dynamic.DynamicDistribution):
    # Overriding method
    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        This function is a gaussian distribution.

        Args:
            shape: shape of the parameters tensors.
            mu_init: init value for the mu. Defaults to 0.0.
            rho_init: init value for the rho. Defaults to -7.0.
            backend: backend to use.

        Raises:
            ValueError: Invalid backend value.
        """

        # Call super class constructor
        super(GaussianDistribution, self).__init__()

        # Choose backend
        self.distribution: dynamic.DynamicDistribution
        if backend == "torch":
            # Import dynamically torch part
            import illia.distributions.dynamic.torch as torch_gaussian

            # Define distribution
            self.distribution = torch_gaussian.GaussianDistribution(
                shape, mu_init, rho_init
            )
        elif backend == "tf":
            # Import dynamically illia library
            import illia.distributions.dynamic.tf as tf_gaussian

            # Define distribution
            self.distribution = tf_gaussian.GaussianDistribution(
                shape, mu_init, rho_init
            )
        else:
            raise ValueError("Invalid backend value")

    # Overriding method
    def sample(self) -> Any:
        return self.distribution.sample()

    # Overriding method
    def log_prob(self, x: Optional[Any]) -> Any:
        return self.distribution.log_prob(x)

    @property
    def num_params(self) -> int:
        return self.distribution.num_params
