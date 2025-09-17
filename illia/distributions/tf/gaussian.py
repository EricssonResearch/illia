"""
Defines a Gaussian (Normal) distribution using Tensorflow with trainable
mean and standard deviation parameters. Supports sampling and
computing log-probabilities of inputs.
"""

# Standard libraries
import math
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.base import DistributionModule


@saving.register_keras_serializable(package="illia", name="GaussianDistribution")
class GaussianDistribution(DistributionModule):
    """
    Learnable Gaussian distribution with diagonal covariance.

    Represents a Gaussian with trainable mean and standard deviation.
    Standard deviation is derived from `rho` using a softplus
    transformation to ensure positivity.

    Notes:
        Assumes diagonal covariance. KL divergence can be computed
        using log-probability differences from `log_prob`.
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
            mu_prior: Mean of the Gaussian prior.
            std_prior: Standard deviation of the prior.
            mu_init: Initial value for the learnable mean.
            rho_init: Initial value for the learnable rho parameter.
            **kwargs: Additional arguments passed to the base class.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.shape = shape
        self.mu_prior_value = mu_prior
        self.std_prior_value = std_prior
        self.mu_init = mu_init
        self.rho_init = rho_init

        # Call build method
        self.build(self.shape)

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds trainable and non-trainable parameters.

        Args:
            input_shape: Input shape used to trigger layer building.

        Returns:
            None.
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
        Generate a sample from the Gaussian distribution.

        Returns:
            Array containing a sample matching the distribution shape.
        """

        # Sampling with reparametrization trick
        eps: tf.Tensor = tf.random.normal(shape=self.rho.shape)
        sigma: tf.Tensor = tf.math.log1p(tf.math.exp(self.rho))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute the log-probability of a given sample. If no sample is
        provided, a new one is drawn internally from the distribution.

        Args:
            x: Optional sample tensor to evaluate.

        Returns:
            Scalar array containing the log-probability.

        Notes:
            Supports both user-supplied and internally generated
                samples.
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
        Return the total number of learnable parameters in the
        distribution.

        Returns:
            Integer count of all learnable parameters.
        """

        return int(tf.size(self.mu))
