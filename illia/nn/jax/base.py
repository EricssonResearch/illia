"""
This module contains the code for the BayesianModule.
"""

# Standard libraries
from abc import ABC, abstractmethod

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx


class BayesianModule(nnx.Module, ABC):
    """
    This class serves as the base class for Bayesian modules.
    Any module designed to function as a Bayesian layer should inherit
    from this class.
    """

    def __init__(self) -> None:
        """
        This method is the constructor for BayesianModule.
        """

        # Call super class constructor
        super().__init__()

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    def freeze(self) -> None:
        """
        Freezes the current layer and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    def unfreeze(self) -> None:
        """
        Unfreezes the current layer and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @abstractmethod
    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        return jnp.array(0), 0
