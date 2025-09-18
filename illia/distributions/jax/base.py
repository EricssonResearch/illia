# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import jax
from flax import nnx
from flax.nnx.rnglib import Rngs


class DistributionModule(nnx.Module, ABC):
    """
    Abstract base for probabilistic distribution modules in JAX.
    Defines the required interface for sampling, computing
    log-probabilities, and counting learnable parameters.

    Notes:
        This class is abstract and cannot be instantiated directly.
        All abstract methods must be implemented by subclasses.
    """

    @abstractmethod
    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Draw a sample from the distribution using the given RNG.

        Args:
            rngs: RNG container used for sampling.

        Returns:
            jax.Array: A sample drawn from the distribution.

        Notes:
            Sampling should be reproducible given the same RNG.
        """

    @abstractmethod
    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        Compute the log-probability of a provided sample. If no
        sample is passed, one is drawn internally.

        Args:
            x: Optional sample to evaluate. If None, a new sample is
                drawn from the distribution.

        Returns:
            jax.Array: Scalar log-probability value.

        Notes:
            Works with both user-supplied and internally drawn
            samples.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Return the number of learnable parameters in the
        distribution.

        Returns:
            int: Total count of learnable parameters.
        """
