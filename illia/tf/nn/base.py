"""
This module contains the code for the BayesianModule.
"""

# Standard libraries
from abc import ABC

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="BayesianModule", name="BayesianModule")
class BayesianModule(ABC, layers.Layer):
    """
    This class serves as the base class for Bayesian modules.
    Any module designed to function as a Bayesian layer should inherit
    from this class.
    """

    def __init__(self, **kwargs) -> None:
        """
        This method is the constructor for BayesianModule.

        Args:
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    def unfreeze(self) -> None:
        """
        Unfreezes the current module and all submodules that are
        instances of BayesianModule. Sets the frozen state to False.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        return tf.Tensor([0.0]), 0
