# Libraries
import math
from typing import Tuple, Optional

import tensorflow as tf

from illia.distributions.dynamic.base import DynamicDistribution


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initializes a set of parameters for a distribution.

        Args:
            shape (Tuple[int, ...]): The shape of the parameter tensors.
            mu_init (float, optional): The initial value for the parameter mu. Defaults to 0.0.
            rho_init (float, optional): The initial value for the parameter rho. Defaults to -7.0.
        """

        # Call super class constructor
        super().__init__()

        # Parameter for standard deviation
        self.std = 0.1

        # Set attributes
        self.shape = shape
        self.mu_init = mu_init
        self.rho_init = rho_init
        self.mu: tf.Variable = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.mu_init, stddev=self.std)
        )
        self.rho: tf.Variable = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.rho_init, stddev=self.std)
        )

    def get_config(self):
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
        Samples a tensor from the distribution using the reparameterization trick.

        Returns:
            tf.Tensor: A sample tensor from the distribution.
        """

        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[tf.Tensor]) -> tf.Tensor:
        """
        Computes the log probability of a given tensor under the distribution.

        Args:
            x (Optional[tf.Tensor]): The tensor for which to compute the log probability.
                                        If None, a sample is drawn from the distribution.

        Returns:
            tf.Tensor: The log probability of the tensor under the distribution.
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
        Returns the number of parameters in the distribution.

        This is calculated as the total number of elements in the tensor representing mu.

        Returns:
            int: The total number of parameters.
        """

        return tf.size(tf.reshape(self.mu, [-1])).numpy()
