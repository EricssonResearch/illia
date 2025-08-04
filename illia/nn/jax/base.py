"""
This module contains the code for the BayesianModule.

Defines an abstract base class for Bayesian layers using Flax's nnx.
"""

# Standard libraries
from abc import ABC, abstractmethod

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx


class BayesianModule(nnx.Module, ABC):
    """
    Abstract base class for all Bayesian modules.

    Any layer intended to function as a Bayesian component should
    inherit from this class and implement the `kl_cost` method.
    """

    def __init__(self) -> None:
        """
        Initializes the BayesianModule.
        Sets default properties for identifying and freezing Bayesian layers.
        """

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    def freeze(self) -> None:
        """
        Freezes the current module by setting its `frozen` flag to True.
        This flag can be used in derived classes to disable updates.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    def unfreeze(self) -> None:
        """
        Unfreezes the current module by setting its `frozen` flag to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @abstractmethod
    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the KL divergence between posterior and prior distributions
        for the module's learnable parameters.

        Returns:
            A tuple containing:
                - kl_cost: The KL divergence as a JAX array.
                - num_params: The number of contributing parameters.
        """

        return jnp.array(0), 0
