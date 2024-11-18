# Libraries
import math

import tensorflow as tf

from illia.distributions.static.base import StaticDistribution


class GaussianDistribution(StaticDistribution):

    def __init__(self, mu: float, std: float) -> None:
        """
        This method is the constrcutor of the class.

        Args:
            mu: mu parameter.
            std: standard deviation parameter.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.mu = tf.constant(mu)
        self.std = tf.constant(std)

    def get_config(self):
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
        This method computes the log probabilities.

        Args:
            x: _description_

        Returns:
            output tensor. Dimensions:
        """

        # Compute log probs
        log_probs = (
            -tf.math.log(tf.sqrt(2 * tf.constant(math.pi)))
            - tf.math.log(self.std)
            - (((x - self.mu) ** 2) / (2 * self.std**2))
            - 0.5
        )

        return tf.math.reduce_sum(log_probs)
