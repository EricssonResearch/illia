# Libraries
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf

from illia.distributions.dynamic.tf.base import DynamicDistribution

# Static variables
PI: tf.Tensor = tf.math.acos(tf.zeros(1)) * 2


class GaussianDistribution(DynamicDistribution):

    def __init__(
        self, shape: Tuple[int, ...], mu_init: float = 0.0, rho_init: float = -7.0
    ) -> None:
        """
        Initializes a set of parameters for a distribution.

        Args:
            shape (Tuple[int, ...]): The shape of the parameter tensors.
            mu_init (float, optional): The initial value for the parameter mu. Defaults to 0.0.
            rho_init (float, optional): The initial value for the parameter rho. Defaults to -7.0.
        """

        # Call super class constructor
        super(GaussianDistribution, self).__init__()

        # Set attributes
        self.mu: tf.Variable = tf.Variable(
            np.random.normal(mu_init, 0.1, shape), dtype=tf.float32
        )
        self.rho: tf.Variable = tf.Variable(
            np.random.normal(rho_init, 0.1, shape), dtype=tf.float32
        )

    def sample(self) -> tf.Tensor:
        """
        Samples a tensor from the distribution using the reparameterization trick.

        Returns:
            tf.Tensor: A sample tensor from the distribution.
        """

        eps: tf.Tensor = tf.random.normal(self.rho.shape)
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
            -tf.math.log(tf.math.sqrt(2 * PI))
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
        
        return len(self.mu.view(-1))
