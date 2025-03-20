"""
This module contains the base class for the Distributions.
"""

# Standard libraries
from abc import abstractmethod
from typing import Optional

# 3pps
import jax.numpy as jnp
from flax import nnx


class Distribution(nnx.Module):
    """
    Abstract base class for probability distributions.
    """

    @abstractmethod
    def sample(self) -> jnp.ndarray:
        """
        Generates a random sample from the distribution.

        Returns:
            A random sample as a JAX NumPy array.
        """

    @abstractmethod
    def log_prob(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sample for which to compute the log
                probability. If not provided, the method may compute
                the log probability for an internally stored sample.

        Returns:
            The log probability of the sample as a JAX NumPy array.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters of the distribution.

        Returns:
            The number of parameters as an integer.
        """
