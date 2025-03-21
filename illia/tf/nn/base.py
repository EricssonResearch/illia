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
    This class implements a BayesianModule. This is still an abstract
    class since it does not implement the forward method.
    """

    def __init__(self):
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
        This method freezes the layer.
        """

        # Set frozen indicator to true for current layer
        self.frozen = True

    def unfreeze(self) -> None:
        """
        This method unfreezes the layer.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Abstract method to compute the KL divergence cost.
        Must be implemented by subclasses.

        Returns:
            A tuple containing the KL divergence cost and its
            associated integer value.
        """

        return tf.Tensor([0.0]), 0
