import tensorflow as tf
from typing import Tuple, Optional
from ..base import Distribution

class GaussianDistribution(Distribution):
    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        super().__init__()
        self.mu_prior = tf.constant([mu_prior], dtype=tf.float32)
        self.std_prior = tf.constant([std_prior], dtype=tf.float32)
        self.mu = tf.Variable(
            tf.random.normal(shape, mean=mu_init, stddev=0.1),
            trainable=True,
            dtype=tf.float32
        )
        self.rho = tf.Variable(
            tf.random.normal(shape, mean=rho_init, stddev=0.1),
            trainable=True,
            dtype=tf.float32
        )

    @tf.function
    def sample(self) -> tf.Tensor:
        eps = tf.random.normal(tf.shape(self.rho))
        sigma = tf.math.log1p(tf.exp(self.rho))
        return self.mu + sigma * eps

    @tf.function
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        if x is None:
            x = self.sample()
        pi = tf.constant(3.14159265358979323846, dtype=tf.float32)
        log_prior = (
            -tf.math.log(tf.sqrt(2 * pi))
            - tf.math.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )
        sigma = tf.math.log1p(tf.exp(self.rho))
        log_posteriors = (
            -tf.math.log(tf.sqrt(2 * pi))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )
        log_probs = tf.reduce_sum(log_posteriors) - tf.reduce_sum(log_prior)
        return log_probs

    @property
    def num_params(self) -> int:
        return int(tf.size(self.mu))
