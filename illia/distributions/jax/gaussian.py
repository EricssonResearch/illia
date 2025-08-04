"""
This module contains the code for the Gaussian distribution.

Defines a learnable Gaussian distribution with methods for
sampling and computing log-probabilities, built on Flax's
nnx.Module system.
"""

# Standard libraries
from typing import Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax.base import DistributionModule


class GaussianDistribution(DistributionModule):
    """
    Implements a learnable Gaussian distribution using Flax's nnx API.

    The distribution is parameterized by a learnable mean and standard
    deviation derived from `rho`, which is transformed via softplus.

    Notes:
        The class assumes a diagonal Gaussian distribution and computes
        KL divergence via log-prob differences in `log_prob`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        mu_prior: float = 0.0,
        std_prior: float = 0.1,
        mu_init: float = 0.0,
        rho_init: float = -7.0,
        rngs: Rngs = nnx.Rngs(0),
    ) -> None:
        """
        Initializes the GaussianDistribution module.

        Args:
            shape: Shape of the learnable parameters.
            mu_prior: Mean of the Gaussian prior.
            std_prior: Standard deviation of the prior.
            mu_init: Initial value for the learnable mean.
            rho_init: Initial value for the learnable rho parameter.
            rngs: RNG container for parameter initialization.
        """

        # Call super-class constructor
        super().__init__()

        # Define priors
        self.mu_prior = mu_prior
        self.std_prior = std_prior

        # Define initial mu and rho
        self.mu = nnx.Param(mu_init + 0.1 * jax.random.normal(rngs.params(), shape))
        self.rho = nnx.Param(rho_init + 0.1 * jax.random.normal(rngs.params(), shape))

    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Draws a sample from the Gaussian distribution.

        Args:
            rngs: RNG container used for sampling.

        Returns:
            A sample drawn from the distribution as a JAX array.
        """

        # Compute epsilon and sigma
        eps: jax.Array = jax.random.normal(rngs.params(), self.rho.shape)
        sigma: jax.Array = jnp.log1p(jnp.exp(jnp.asarray(self.rho)))

        return self.mu + sigma * eps

    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        Computes the KL divergence between posterior and prior.

        If no input is provided, a sample is generated from the
        current distribution.

        Args:
            x: Optional sample for evaluating the log-probability.

        Returns:
            The KL divergence as a scalar JAX array.
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
        sigma: jax.Array = jnp.log1p(jnp.exp(self.rho))  # type: ignore

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
        Returns the number of learnable parameters.

        Returns:
            The total number of parameters in the distribution.
        """

        return len(self.mu.reshape(-1))

    def __call__(self) -> jax.Array:
        """
        Performs a forward pass by sampling from the distribution.

        Returns:
            A sample from the distribution.
        """

        return self.sample()
