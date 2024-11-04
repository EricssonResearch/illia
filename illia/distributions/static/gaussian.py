# Libraries
from typing import Any, Optional

import illia.distributions.static as static


class GaussianDistribution(static.StaticDistribution):
    # overriding method
    def __init__(
        self,
        mu: float,
        std: float,
        backend: Optional[str] = "torch",
    ) -> None:
        # call super class constructor
        super(GaussianDistribution, self).__init__()

        self.distribution: static.StaticDistribution
        if backend == "torch":
            import illia.distributions.static.torch as torch_gaussian

            self.distribution = torch_gaussian.GaussianDistribution(mu, std)
        elif backend == "tf":
            import illia.distributions.static.tf as tf_gaussian

            self.distribution = tf_gaussian.GaussianDistribution(mu, std)
        else:
            raise ValueError("Invalid backend value")

    # overriding method
    def log_prob(self, x: Any) -> Any:
        return self.distribution.log_prob(x)
