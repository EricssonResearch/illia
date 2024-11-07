# standard libraries
from typing import Optional

# 3pp
import jax
import jax.numpy as jnp
from flax import linen as nn

# own modules
from ..base import (
    Distribution,
)


class GaussianDistribution(Distribution, nn.Module):
    """
    This is the class to implement a learnable gausssian distribution
    in jax and flax.

    Attr:
        shape: shape of the distribution.
        mu_prior_value: mu prior value.
        std_prior_value: float = 0.1
        mu_init: float = 0.0
        rho_init: float = -7.0

    Returns:
        _description_
    """

    shape: tuple[int, ...]
    mu_prior: float = 0.0
    std_prior: float = 0.1
    mu_init: float = 0.0
    rho_init: float = -7.0

    def setup(self) -> None:
        """
        This is the setup method to define the parameters of the module.

        Returns:
            None.
        """

        # call super-class constructor
        super().__init__()

        # define initial mu and rho
        self.mu: jax.Array = self.mu_init + self.param(
            "mu", nn.initializers.normal(self.rho_init), self.shape
        )
        self.rho: jax.Array = self.mu_init + self.param(
            "rho", nn.initializers.normal(self.rho_init), self.shape
        )

        return None

    # overriding method
    def sample(self, seed: int = 0, **kwargs) -> jax.Array:
        """
        This method is to sample from parameters.

        Returns:
            sampled jax array.
        """

        # get key
        key = jax.random.key(seed)

        # compute epsilon and sigma
        eps: jax.Array = jax.random.normal(key, self.rho.shape)
        sigma: jax.Array = jnp.log1p(jnp.exp(self.rho))

        return self.mu + sigma * eps

    def __call__(self) -> jax.Array:
        """
        This method if the forward pass of the module.

        Returns:
            sampled jax array.
        """

        return self.sample()

    # overriding method
    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        This function computes the log probabilities.

        Args:
            x: sampled array. Dimensions: [*]. If it is none the tensor
                is sampled. Defaults to None.

        Returns:
            log probs.
        """

        # sample if x is None
        if x is None:
            x = self.sample()

        # define pi variable
        pi: jax.Array = jnp.acos(jnp.zeros(1)) * 2

        # compute log priors
        log_prior = (
            -jnp.log(jnp.sqrt(2 * pi))
            - jnp.log(self.std_prior)
            - (((x - self.mu_prior) ** 2) / (2 * self.std_prior**2))
            - 0.5
        )

        # compute sigma
        sigma: jax.Array = jnp.log1p(jnp.exp(self.rho))

        # compute log posteriors
        log_posteriors = (
            -jnp.log(jnp.sqrt(2 * pi))
            - jnp.log(sigma)
            - (((x - self.mu) ** 2) / (2 * sigma**2))
            - 0.5
        )

        # compute final log probs
        log_probs = log_posteriors.sum() - log_prior.sum()

        return log_probs

    @property
    def num_params(self) -> int:
        """
        This method returns the number of parameters in the module.

        Returns:
            number of parameters.
        """

        return len(self.mu.view(-1))
