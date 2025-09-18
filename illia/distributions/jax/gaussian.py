# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax.base import DistributionModule


class GaussianDistribution(DistributionModule):
    """
    Learnable Gaussian distribution with diagonal covariance.
    Represents a Gaussian with trainable mean and standard
    deviation. The standard deviation is derived from `rho`
    using a softplus transformation to ensure positivity.

    Notes:
        Assumes diagonal covariance. KL divergence can be
        estimated via log-probability differences from
        `log_prob`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        rngs: Rngs = nnx.Rngs(0),
        **kwargs: Any,
    ) -> None:
        """
        Initialize a learnable Gaussian distribution module.

        Args:
            shape: Shape of the learnable parameters.
            mu_prior: Mean of the Gaussian prior.
            std_prior: Standard deviation of the prior.
            mu_init: Initial value for the learnable mean.
            rho_init: Initial value for the learnable rho.
            rngs: RNG container for parameter initialization.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None
        """

        # Call super-class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.shape = shape
        self.mu_prior = mu_prior
        self.std_prior = std_prior
        self.mu_init = mu_init
        self.rho_init = rho_init

        # Define initial mu and rho
        self.mu = nnx.Param(
            self.mu_init + 0.1 * jax.random.normal(rngs.params(), self.shape)
        )
        self.rho = nnx.Param(
            self.rho_init + 0.1 * jax.random.normal(rngs.params(), self.shape)
        )

    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Draw a sample from the Gaussian distribution.

        Args:
            rngs: RNG container used for sampling.

        Returns:
            jax.Array: A sample drawn from the distribution.

        Notes:
            Sampling is reproducible with the same RNG.
        """

        # Compute epsilon and sigma
        eps: jax.Array = jax.random.normal(rngs.params(), self.rho.shape)
        sigma: jax.Array = jnp.log1p(jnp.exp(jnp.asarray(self.rho)))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        Compute the log-probability of a given sample. If no
        sample is provided, one is drawn internally.

        Args:
            x: Optional input sample to evaluate. If None,
                a new sample is drawn from the distribution.

        Returns:
            jax.Array: Scalar log-probability value.

        Notes:
            Supports both user-supplied and internally drawn
            samples.
        """

        # Sample if x is None
        if x is None:
            x = self.sample()

        # Define pi variable
        pi: jax.Array = jnp.acos(jnp.zeros(1)) * 2

        # Compute log priors
        log_prior = (
            -jnp.log(jnp.sqrt(2 * pi))
            - jnp.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # Compute sigma
        sigma: jax.Array = jnp.log1p(jnp.exp(jnp.asarray(self.rho)))

        # Compute log posteriors
        log_posteriors = (
            -jnp.log(jnp.sqrt(2 * pi))
            - jnp.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # Compute final log probs
        log_probs = log_posteriors.sum() - log_prior.sum()

        return log_probs

    @property
    def num_params(self) -> int:
        """
        Return the number of learnable parameters in the
        distribution.

        Returns:
            int: Total count of learnable parameters.
        """

        return len(self.mu.reshape(-1))

    def __call__(self) -> jax.Array:
        """
        Perform a forward pass by drawing a sample.

        Returns:
            jax.Array: A sample from the distribution.
        """

        return self.sample()
