"""
Abstract base class for Bayesian layers using Flax's nnx.
Provides common functionality for identifying Bayesian modules,
freezing/unfreezing parameters, and computing KL divergence costs.
"""

# Standard libraries
from abc import ABC, abstractmethod

# 3pps
import jax
from flax import nnx


class BayesianModule(nnx.Module, ABC):
    """
    Abstract base for Bayesian-aware modules in Flax's nnx framework.
    Any Bayesian layer should inherit from this class. It tracks whether
    the module is Bayesian and provides freezing/unfreezing mechanisms
    for controlling parameter updates.

    Notes:
        Derived classes must implement `freeze` and `kl_cost`.
    """

    def __init__(self) -> None:
        """
        Initialize the module with default Bayesian-specific flags.
        Sets `frozen` to False and `is_bayesian` to True.

        Returns:
            None.
        """

        super().__init__()

        self.frozen: bool = False
        self.is_bayesian: bool = True

    @abstractmethod
    def freeze(self) -> None:
        """
        Freeze the module by setting its `frozen` flag to True.
        Derived classes can use this flag to disable parameter updates.

        Returns:
            None.
        """

    def unfreeze(self) -> None:
        """
        Unfreeze the module by setting its `frozen` flag to False.

        Returns:
            None.
        """
        
        self.frozen = False

    @abstractmethod
    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Compute the KL divergence between posterior and prior distributions.

        Returns:
            Tuple containing:
                - kl_cost: Kullback-Leibler divergence for this module.
                - num_params: Number of parameters contributing to KL.
        """
