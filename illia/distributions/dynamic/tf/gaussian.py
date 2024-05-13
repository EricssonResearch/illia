# deep larning libraries
import numpy as np
import tensorflow as tf

# other libraries
from typing import Tuple, Optional

# own modules
from illia.distributions.dynamic.tf.base import DynamicDistribution

# static variables
PI: tf.Tensor = tf.math.acos(tf.zeros(1)) * 2


class GaussianDistribution(DynamicDistribution):
    # overriding method
    def __init__(
        self, shape: Tuple[int, ...], mu_init: float = 0.0, rho_init: float = -7.0
    ) -> None:
        super().__init__()

        self.mu: tf.Variable = tf.Variable(
            np.random.normal(mu_init, 0.1, shape), dtype=tf.float32
        )
        self.rho: tf.Variable = tf.Variable(
            np.random.normal(rho_init, 0.1, shape), dtype=tf.float32
        )

    # overriding method
    def sample(self) -> tf.Tensor:
        eps: tf.Tensor = tf.random.normal(self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    # overriding method
    def log_prob(self, x: Optional[tf.Tensor]) -> tf.Tensor:
        if x is None:
            x = self.sample()
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        log_posteriors: tf.Tensor = (
            -tf.math.log(tf.math.sqrt(2 * PI))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        return tf.math.reduce_sum(log_posteriors)

    @property
    def num_params(self) -> int:
        return len(self.mu.view(-1))
