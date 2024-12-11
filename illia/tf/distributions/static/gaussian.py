# Libraries
import math

import tensorflow as tf  # type: ignore

from . import StaticDistribution


class GaussianDistribution(StaticDistribution):

    def __init__(self, mu: float, std: float) -> None:
        """
        Initialize a Gaussian distribution with given mean (mu) and standard deviation (std).

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
        Get the configuration of the Gaussian Distribution object. This method retrieves the base
        configuration of the parent class and combines it with custom configurations specific to
        the Gaussian Distribution.

        Returns:
            A dictionary containing the combined configuration of the Gaussian Distribution.
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
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters mu and rho, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x: Input data for which the log PDF needs to be calculated.

        Returns:
            The log probability density function (PDF) of the input data or sample.
        """

        # Compute log probs
        log_probs = (
            -tf.math.log(tf.constant(math.sqrt(2 * math.pi), dtype=tf.float32))
            - tf.math.log(self.std)
            - ((x - self.mu) ** 2) / (2 * self.std**2)
            - 0.5
        )

        return tf.math.reduce_sum(log_probs)
