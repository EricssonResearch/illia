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
    This class serves as the base class for Distributions modules.
    Any module designed to function as a distribution layer should
    inherit from this class.
    """

    @abstractmethod
    def sample(self) -> jnp.ndarray:
        """
        This method samples a tensor from the distribution.

        Returns:
            Sampled tensor. Dimensions: [*] (same ones as the mu and
                std parameters).
        """

    @abstractmethod
    def log_prob(self, x: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        This method computes the log prob of the distribution.

        Args:
            x: Output already sampled. If no output is introduced,
                first we will sample a tensor from the current
                distribution.

        Returns:
            Log prob calculated as a tensor. Dimensions: [].
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        This method computes the number of parameters of the
        distribution.

        Returns:
            Number of parameters.
        """
