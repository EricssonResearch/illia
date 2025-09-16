"""
Base class for building distribution modules using Flax and JAX.

Provides a standardized interface for sampling, computing log
probabilities, and reporting the number of parameters in custom
probabilistic layers.

Notes:
    This class is abstract and should not be instantiated directly.
    Subclasses must implement all abstract methods to specify
    distribution behavior.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import jax
from flax import nnx
from flax.nnx.rnglib import Rngs

class DistributionModule(nnx.Module, ABC):
    """
    Abstract base for Flax-based probabilistic distribution modules.

    Defines a required interface for sampling, computing log-probabilities,
    and retrieving parameter counts. Subclasses must implement all
    abstract methods to provide specific distribution logic.
        
    Notes:
        Avoid direct instantiation, this serves as a blueprint for
        derived classes.
    """

    @abstractmethod
    def sample(self, rngs: Rngs = nnx.Rngs(0)) -> jax.Array:
        """
        Generates and returns a sample from the underlying distribution.

        Args:
            rngs: RNG container used for sampling.

        Returns:
            Sample array matching the shape and structure defined by
            the distribution parameters.
        """

    @abstractmethod
    def log_prob(self, x: Optional[jax.Array] = None) -> jax.Array:
        """
        Computes the log-probability of an input sample.
        If no sample is provided, a new one is drawn internally from the
        current distribution before computing the log-probability.

        Args:
            x: Optional sample tensor to evaluate.

        Returns:
            Scalar tensor representing the computed log-probability.

        Notes:
            This method supports both user-supplied samples and internally
            generated ones for convenience when evaluating likelihoods.
        """


    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Returns the total number of learnable parameters in the distribution.

        Returns:
            Integer representing the total number of learnable parameters.
        """
