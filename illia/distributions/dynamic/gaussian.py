# other libraries
from typing import Tuple, Optional, Literal, Any

# own modules
import illia.distributions.dynamic as dynamic


class GaussianDistribution(dynamic.DynamicDistribution):
    # overriding method
    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        backend: int = 0,
    ) -> None:
        """
        This function is a gaussian distribution.

        Args:
            shape: shape of the parameters tensors.
            mu_init: init value for the mu. Defaults to 0.0.
            rho_init: init value for the rho. Defaults to -7.0.
            backend: backend to use. Defaults to 0, which mean is
                using torch.

        Raises:
            ValueError: Invalid backend value.
        """

        # call super class constructor
        super().__init__()

        # choose backend
        self.distribution: dynamic.DynamicDistribution
        if backend == 0:
            # import dynamically torch part
            import illia.distributions.dynamic.torch as torch_gaussian

            # define distribution
            self.distribution = torch_gaussian.GaussianDistribution(
                shape, mu_init, rho_init
            )

        elif backend == 1:
            # import dynamically illia library
            import illia.distributions.dynamic.tf as tf_gaussian

            # define distribution
            self.distribution = tf_gaussian.GaussianDistribution(
                shape, mu_init, rho_init
            )

        else:
            raise ValueError("Invalid backend value")

    # overriding method
    def sample(self) -> Any:
        return self.distribution.sample()

    # overriding method
    def log_prob(self, x: Optional[Any]) -> Any:
        return self.distribution.log_prob(x)

    @property
    def num_params(self) -> int:
        return self.distribution.num_params
