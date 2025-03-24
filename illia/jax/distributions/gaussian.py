"""
This module contains the code for the gaussian distribution.
"""

# Standard libraries
from typing import Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.jax.distributions.base import Distribution


class GaussianDistribution(Distribution):
    """
    This is the class to implement a learnable gausssian distribution.
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
        Initializes the GaussianDistribution with given priors and
        initial parameters.

        Args:
            shape: The shape of the parameters.
            mu_prior: The mean prior value.
            std_prior: The standard deviation prior value.
            mu_init: The initial mean value.
            rho_init: The initial rho value, which affects the initial
                standard deviation.
            rngs: Nnx rng container. Defaults to nnx.Rngs(0).
        """

        # Call super-class constructor
        super().__init__()

        # Define priors
        self.mu_prior = mu_prior
        self.std_prior = std_prior

        # Define initial mu and rho
        self.mu = nnx.Param(
            mu_init + rho_init * jax.random.normal(rngs.params(), shape)
        )
        self.rho = nnx.Param(
            mu_init + rho_init * jax.random.normal(rngs.params(), shape)
        )

    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Samples from the distribution using the current parameters.

        Args:
            rngs: Nnx rng container. Defaults to nnx.Rngs(0).

        Returns:
            A sampled JAX array.
        """

        # Compute epsilon and sigma
        eps: jax.Array = jax.random.normal(rngs.params(), self.rho.shape)
        sigma: jax.Array = jnp.log1p(jnp.exp(self.rho))  # type: ignore

        return self.mu + sigma * eps

    def __call__(self) -> jax.Array:
        """
        Performs the forward pass of the module.

        Returns:
            A sampled JAX array.
        """

        return self.sample()

    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sampled array. If None, a sample is
                generated.

        Returns:
            The log probability of the sample as a JAX array.
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
        Returns the number of parameters in the module.

        Returns:
            The number of parameters as an integer.
        """

        return len(self.mu.view(-1))
