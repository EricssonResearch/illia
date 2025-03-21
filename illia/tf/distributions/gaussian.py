"""
This module contains the code for the gaussian distribution.
"""

# Standard libraries
import math
from typing import Optional

# 3pps
import keras
import tensorflow as tf

# Own modules
from illia.tf.distributions.base import Distribution


class GaussianDistribution(Distribution):
    """
    This is the class to implement a learnable gausssian distribution.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initializes the GaussianDistribution with given priors and
        initial parameters.

        Args:
            shape: The shape of the parameters.
            mu_prior: The mean prior value.
            std_prior: The standard deviation prior value.
            mu_init: The initial mean value.
            rho_init: The initial rho value, which affects the initial
                standard deviation.
        """

        # Call super-class constructor
        super().__init__()

        # Set parameters
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init

        # Define priors
        self.mu_prior: tf.Tensor = tf.convert_to_tensor(mu_prior, dtype=tf.float32)
        self.std_prior: tf.Tensor = tf.convert_to_tensor(std_prior, dtype=tf.float32)

        # Define trainable parameters
        self.mu = self.add_weight(
            shape=self.shape,
            initializer=keras.initializers.RandomNormal(mean=self.mu_init, stddev=0.1),
            trainable=True,
            name=f"{self.name}_mu",
        )

        self.rho = self.add_weight(
            shape=self.shape,
            initializer=keras.initializers.RandomNormal(mean=self.rho_init, stddev=0.1),
            trainable=True,
            name=f"{self.name}_rho",
        )

    def sample(self) -> tf.Tensor:
        """
        Samples from the distribution using the current parameters.

        Returns:
            A sampled tensor.
        """

        # Sampling with reparametrization trick
        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + tf.multiply(sigma, eps)

    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sampled array. If None, a sample is
                generated.

        Returns:
            The log probability of the sample as a tensor.
        """

        # Sample if x is None
        if x is None:
            x = self.sample()

        # Define pi
        pi: tf.Tensor = tf.convert_to_tensor(math.pi)

        # Compute log priors
        log_prior = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # Compute sigma
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        # Compute log posteriors
        log_posteriors: tf.Tensor = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # Compute final log probs
        log_probs = tf.math.reduce_sum(log_posteriors) - tf.math.reduce_sum(log_prior)

        return log_probs

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters in the module.

        Returns:
            The number of parameters as an integer.
        """

        return int(tf.size(self.mu).numpy())
