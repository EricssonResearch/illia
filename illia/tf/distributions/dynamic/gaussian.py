# Libraries
import math

import tensorflow as tf

from . import DynamicDistribution


class GaussianDistribution(DynamicDistribution):
    """
    This class models a Gaussian distribution, allowing for sampling
    and probability calculations based on learned mean and standard
    deviation.

    Args:
        shape: Dimensions of the distribution parameters.
        mu_init: Initial mean value for the distribution.
        rho_init: Initial mean value for the log-standard deviation.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initialize a Gaussian Distribution object with trainable
        parameters mu and rho.

        Args:
            shape: The shape of the distribution parameters.
            mu_init: The initial mean value for mu.
            rho_init: The initial mean value for rho.
        """

        # Call super class constructor
        super().__init__()

        # Parameter for standard deviation
        self._std = 0.1

        # Set attributes
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init
        self.mu: tf.Variable = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.mu_init, stddev=self._std)
        )
        self.rho: tf.Variable = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.rho_init, stddev=self._std)
        )

    def get_config(self) -> dict:
        """
        Retrieve the configuration of the Gaussian distribution,
        combining base and custom parameters.

        Returns:
            Dictionary with the Gaussian distribution's configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "shape": self.shape,
            "mu_init": self.mu_init,
            "rho_init": self.rho_init,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def sample(self) -> tf.Tensor:
        """
        Generate a sample from the Gaussian distribution using current
        parameters.

        Returns:
            Tensor representing a sample from the distribution.
        """

        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the log probability of provided input data based on the
        Gaussian distribution.

        Args:
            x: Input tensor for log probability computation.

        Returns:
            Tensor representing the log probability of the input data.
        """

        if x is None:
            x = self.sample()

        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        log_posteriors: tf.Tensor = (
            -tf.math.log(tf.math.sqrt(2 * tf.constant(math.pi)))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        return tf.math.reduce_sum(log_posteriors)

    @property
    def num_params(self) -> int:
        """
        Get the total number of parameters in the Gaussian distribution.

        Returns:
            Integer representing the number of distribution parameters.
        """

        return tf.size(tf.reshape(self.mu, [-1]))
