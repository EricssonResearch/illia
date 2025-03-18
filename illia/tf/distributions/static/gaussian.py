# Libraries
import math

import tensorflow as tf

from . import StaticDistribution


class GaussianDistribution(StaticDistribution):
    """
    Gaussian distribution with fixed mean and standard deviation. This
    class models a Gaussian distribution and allows for probability
    calculations based on provided parameters.

    Args:
        mu: Mean of the Gaussian distribution.
        std: Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mu: float, std: float) -> None:
        """
        Initialize a Gaussian distribution with given mean (mu) and
        standard deviation (std).

        Args:
            mu: The mean of the Gaussian distribution.
            std: The standard deviation of the Gaussian distribution.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.mu = tf.constant(mu, dtype=tf.float32)
        self.std = tf.constant(std, dtype=tf.float32)

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
            "mu": self.mu,
            "std": self.std,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute the log probability of provided input data based on the
        Gaussian distribution.

        Args:
            x: Tensor of input data for log probability computation.

        Returns:
            Tensor representing the log probability of the input data.
        """

        # Compute log probs
        log_probs = (
            -tf.math.log(tf.constant(math.sqrt(2 * math.pi), dtype=tf.float32))
            - tf.math.log(self.std)
            - ((x - self.mu) ** 2) / (2 * self.std**2)
            - 0.5
        )

        return tf.math.reduce_sum(log_probs)
