# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import jax
from flax import nnx
from flax.nnx.rnglib import Rngs


class DistributionModule(nnx.Module, ABC):
    """
    Abstract base for Jax-based probabilistic distribution modules.
    Defines the interface for sampling, computing log-probabilities,
    and retrieving parameter counts. Subclasses must implement all
    abstract methods to provide specific distribution logic.

    Notes:
        This class is abstract and should not be instantiated directly.
        All abstract methods must be implemented by subclasses.
    """

    @abstractmethod
    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Generate a sample from the underlying distribution.

        Args:
            rngs: RNG container used for sampling.

        Returns:
            Array containing a sample matching the distribution shape.

        Notes:
            This method should be deterministic given the same RNG.
        """

    @abstractmethod
    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
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

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Return the total number of learnable parameters in the
        distribution.

        Returns:
            Integer count of all learnable parameters.
        """
