"""
This module contains the code for the Gaussian distribution.

Implements a learnable Gaussian distribution in TensorFlow using
Keras layers and supports serialization and log-probability
evaluation.
"""

# Standard libraries
import math
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.base import DistributionModule


@saving.register_keras_serializable(
    package="BayesianModule", name="GaussianDistribution"
)
class GaussianDistribution(DistributionModule):
    """
    Implements a learnable Gaussian distribution for TensorFlow models.

    The distribution uses trainable parameters `mu` and `rho`, where
    the standard deviation is obtained via a softplus transformation
    of `rho`. Supports KL divergence via the `log_prob` method.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Gaussian distribution layer.

        Args:
            shape: Shape of the learnable parameters.
            mu_prior: Prior mean for KL divergence computation.
            std_prior: Prior std for KL divergence computation.
            mu_init: Initial value for the mean.
            rho_init: Initial value for the rho parameter.
            **kwargs: Additional arguments passed to the base class.
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
        Builds trainable and non-trainable parameters.

        Args:
            input_shape: Input shape used to trigger layer building.
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

    def get_config(self) -> dict:
        """
        Returns the configuration for serialization.

        Returns:
            A dictionary containing the layer's config.
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
        Draws a sample from the distribution using reparameterization.

        Returns:
            A sample tensor with the same shape as the parameters.
        """

        # Sampling with reparametrization trick
        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes KL divergence between posterior and prior.

        If no sample is provided, one is drawn from the current
        distribution.

        Args:
            x: Optional input sample tensor.

        Returns:
            A scalar tensor representing the KL divergence.
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
        Returns the number of learnable parameters.

        Returns:
            Total number of parameters in the distribution.
        """

        return int(tf.size(self.mu))
