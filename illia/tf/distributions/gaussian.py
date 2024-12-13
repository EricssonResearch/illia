# standard libraries
import math
from typing import Optional

# 3pp
import tensorflow as tf

# own modules
from illia.tf.distributions.base import (
    Distribution,
)


class GaussianDistribution(Distribution):
    """
    This class is the implementation of GaussianDistribution in 
    tensorflow.

    Attr:
        Distribution: _description_
    """
    
    # overriding method
    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
    ) -> None:
        """
        This class is the constructor for GaussianDistribution.

        Args:
            shape: shape of the distribution.
            mu_prior: mu for the prior distribution. Defaults to 0.0.
            std_prior: std for the prior distribution. Defaults to 0.1.
            mu_init: init value for mu. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here. Defaults to 0.0.
            rho_init: init value for rho. This tensor will be initialized
                with a normal distribution with std 0.1 and the mean is
                the parameter specified here. Defaults to -7.0.
        """

        # call super-class constructor
        super().__init__()

        # define priors
        self.mu_prior: tf.Tensor = tf.constant(mu_prior, dtype=tf.float32)
        self.std_prior: tf.Tensor = tf.constant(std_prior, dtype=tf.float32)

        # define initial mu and rho
        self.mu: tf.Tensor = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.mu_init, stddev=0.1)
        )
        self.rho: tf.Tensor = tf.Variable(
            tf.random.normal(shape=self.shape, mean=self.rho_init, stddev=0.1)
        )

    # overriding method
    def sample(self) -> tf.Tensor:
        """
        This method generates a sample from the Gaussian distribution
        using the current parameters mu and rho. The sample is obtained
        by adding a random noise (epsilon) to the mean (mu), where the
        noise is scaled by the standard deviation (sigma).

        Returns:
            sampled tensor from the Gaussian distribution.
                Dimensions: [*] (same ones as the mu and std
                parameters).
        """

        # sampling with reparametrization trick
        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    # overriding method
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        This method calculates the log probability density function
        (PDF) of the given input data.

        If no input data is provided, a sample is generated using the
        current parameters. The log PDF is calculated using the current
        parameters mu and rho, which represent the mean and standard
        deviation of the Gaussian distribution, respectively.

        Args:
            x: output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution. Defaults to None.

        Returns:
            The log probability density function (PDF) of the input
                data or sample.. Dimensions: [].
        """

        # sample if x is None
        if x is None:
            x = self.sample()

        # define pi
        pi: tf.Tensor = tf.constant(math.pi)

        # compute log priors
        log_prior = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # compute sigma
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        # compute log posteriors
        log_posteriors: tf.Tensor = (
            -tf.math.log(tf.math.sqrt(2 * pi))
            - tf.math.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # compute final log probs
        log_probs = tf.math.reduce_sum(log_posteriors) - tf.math.reduce_sum(log_prior)

        return log_probs

    @property
    def num_params(self) -> int:
        """
        Calculate the total number of parameters in the Gaussian
        Distribution, which is the product of the dimensions of the
        mean (mu) parameter.

        Returns:
            The total number of parameters in the Gaussian Distribution.
        """

        return tf.size(tf.reshape(self.mu, [-1]))
