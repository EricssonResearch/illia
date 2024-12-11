# Libraries
import math
from typing import Tuple

import tensorflow as tf  # type: ignore

from . import DynamicDistribution


class GaussianDistribution(DynamicDistribution):
    def __init__(
        self,
        shape: Tuple[int, ...],
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        Initialize a Gaussian Distribution object with trainable parameters mu and rho.
        The parameters are initialized with a normal distribution with mean mu_init and rho_init,
        and a standard deviation of 0.1.

        Args:
            shape: The shape of the distribution parameters.
            mu_init: The initial mean value for mu.
            rho_init: The initial mean value for rho.
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
            "shape": self.shape,
            "mu_init": self.mu_init,
            "rho_init": self.rho_init,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def sample(self) -> tf.Tensor:
        """
        Generate a sample from the Gaussian distribution using the current parameters
        mu and rho. The sample is obtained by adding a random noise (epsilon) to the mean (mu),
        where the noise is scaled by the standard deviation (sigma).

        Returns:
            A tensor representing a sample from the Gaussian distribution.
        """

        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: tf.Tensor) -> tf.Tensor:
        """
        Calculate the log probability density function (PDF) of the given input data.

        If no input data is provided, a sample is generated using the current parameters.
        The log PDF is calculated using the current parameters mu and rho, which represent
        the mean and standard deviation of the Gaussian distribution, respectively.

        Args:
            x: Input data for which the log PDF needs to be calculated.
                If None, a sample is generated using the current parameters.

        Returns:
            The log probability density function (PDF) of the input data or sample.
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
        Calculate the total number of parameters in the Gaussian Distribution, which is the product
        of the dimensions of the mean (mu) parameter.

        Returns:
            The total number of parameters in the Gaussian Distribution.
        """

        return tf.size(tf.reshape(self.mu, [-1]))
