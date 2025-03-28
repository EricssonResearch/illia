"""
This module contains the code for the Gaussian distribution.
"""

# Standard libraries
import math
from typing import Optional

# 3pps
import keras
import tensorflow as tf
from keras import saving

# Own modules
from illia.tf.distributions.base import Distribution


@saving.register_keras_serializable(
    package="BayesianModule", name="GaussianDistribution"
)
class GaussianDistribution(Distribution):
    """
    This is the class to implement a learnable Gaussian distribution.
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
        Constructor for GaussianDistribution.

        Args:
            shape: Shape of the distribution.
            mu_prior: Mean for the prior distribution.
            std_prior: Standard deviation for the prior distribution.
            mu_init: Initial mean for mu.
            rho_init: Initial mean for rho.
        """

        # Call super class constructor
        super().__init__()

        # Set parameters
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init
        self.mu_prior_value = mu_prior
        self.std_prior_value = std_prior

        # Define non-trainable priors variables correctamente
        self.mu_prior = self.add_weight(
            name="mu_prior",
            shape=(),
            initializer=tf.constant_initializer(self.mu_prior_value),
            trainable=False,
        )
        self.std_prior = self.add_weight(
            name="std_prior",
            shape=(),
            initializer=tf.constant_initializer(self.std_prior_value),
            trainable=False,
        )

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

    def get_config(self):
        """
        Retrieves the configuration of the Gaussian Distribution layer.

        Returns:
            Dictionary containing layer configuration.
        """

        base_config = super().get_config()

        config = {
            "shape": self.shape,
            "mu_prior": float(self.mu_prior.numpy()),
            "std_prior": float(self.std_prior.numpy()),
            "mu_init": self.mu_init,
            "rho_init": self.rho_init,
        }

        return {**base_config, **config}

    def sample(self) -> tf.Tensor:
        """
        Samples from the distribution using the current parameters.

        Returns:
            A sampled tensor.
        """

        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sampled array. If None, a sample is generated.

        Returns:
            The log probability of the sample as a tensor.
        """

        if x is None:
            x = self.sample()

        pi: tf.Tensor = tf.convert_to_tensor(math.pi)

        log_prior = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        log_posteriors: tf.Tensor = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        log_probs = tf.math.reduce_sum(log_posteriors) - tf.math.reduce_sum(log_prior)

        return log_probs

    @property
    def num_params(self) -> int:
        """
        Returns the number of parameters in the module.

        Returns:
            The number of parameters as an integer.
        """

        return int(tf.size(self.mu))
