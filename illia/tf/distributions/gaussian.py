"""
This module contains the code for the Gaussian distribution.
"""

# Standard libraries
import math
from typing import Optional

# 3pps
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
        **kwargs,
    ) -> None:
        """
        Constructor for GaussianDistribution.

        Args:
            shape: The shape of the distribution.
            mu_prior: The mean for the prior distribution.
            std_prior: The standard deviation for the prior distribution.
            mu_init: The initial mean for the distribution.
            rho_init: The initial value for the rho parameter.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set parameters
        self.shape = shape
        self.mu_prior_value = mu_prior
        self.std_prior_value = std_prior
        self.mu_init = mu_init
        self.rho_init = rho_init

        # Call build method
        self.build(shape)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the Gaussian Distribution layer.

        Args:
            input_shape: Input shape of the layer.
        """

        # Define non-trainable priors variables
        self.mu_prior = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.mu_prior_value),
            trainable=False,
            name="mu_prior",
        )

        self.std_prior = self.add_weight(
            shape=(),
            initializer=tf.constant_initializer(self.std_prior_value),
            trainable=False,
            name="std_prior",
        )

        # Define trainable parameters
        self.mu = self.add_weight(
            shape=self.shape,
            initializer=tf.random_normal_initializer(mean=self.mu_init, stddev=0.1),
            trainable=True,
            name="mu",
        )

        self.rho = self.add_weight(
            shape=self.shape,
            initializer=tf.random_normal_initializer(mean=self.rho_init, stddev=0.1),
            trainable=True,
            name="rho",
        )

        # Call super-class build method
        super().build(input_shape)

    def get_config(self):
        """
        Retrieves the configuration of the Gaussian Distribution layer.

        Returns:
            Dictionary containing layer configuration.
        """

        base_config = super().get_config()

        config = {
            "shape": self.shape,
            "mu_prior": self.mu_prior_value,
            "std_prior": self.std_prior_value,
            "mu_init": self.mu_init,
            "rho_init": self.rho_init,
        }

        return {**base_config, **config}

    def sample(self) -> tf.Tensor:
        """
        This method samples a tensor from the distribution.

        Returns:
            Sampled tensor. Dimensions: [*] (same ones as the mu and
                std parameters).
        """

        # Sampling with reparametrization trick
        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        This method computes the log prob of the distribution.

        Args:
            x: Output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution.

        Returns:
            Log prob calculated as a tensor. Dimensions: [].
        """

        # Sample if x is None
        if x is None:
            x = self.sample()

        # Define pi variable
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
        This method computes the number of parameters of the
        distribution.

        Returns:
            Number of parameters.
        """

        return int(tf.size(self.mu))
