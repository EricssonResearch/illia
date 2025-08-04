"""
This module contains the code for the BayesianModule.

Defines an abstract base class for Bayesian layers using Keras's layers.
"""

# Standard libraries
from abc import ABC

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="BayesianModule", name="BayesianModule")
class BayesianModule(layers.Layer, ABC):
    """
    Abstract base class for Bayesian layers in Keras.

    Any custom Bayesian layer should inherit from this class and
    implement the `kl_cost()` method.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the BayesianModule.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """

        # Call super class constructor
        super().__init__(**kwargs)

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

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the KL divergence between posterior and prior distributions
        for the module's learnable parameters.

        Returns:
            A tuple containing:
                - kl_cost: The KL divergence as a JAX array.
                - num_params: The number of contributing parameters.
        """

        return tf.Tensor([0.0]), 0
