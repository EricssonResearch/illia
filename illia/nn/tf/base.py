"""
This module defines an abstract base class for Bayesian layers using
Tensorflow. It facilitates identifying, freezing, and computing KL
costs for Bayesian-aware modules.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import Any

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="illia", name="BayesianModule")
class BayesianModule(layers.Layer, ABC):
    """
    Abstract base for Bayesian-aware modules in Flax's nnx framework.
    Any Bayesian layer should inherit from this class.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the module with default Bayesian-specific flags.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set freeze false by default
        self.frozen: bool = False

        # Create attribute to know is a bayesian layer
        self.is_bayesian: bool = True

    @abstractmethod
    def freeze(self) -> None:
        """
        Freezes the current module by setting its `frozen` flag to True.
        This flag can be used in derived classes to disable updates.

        Returns:
            None.
        """

    def unfreeze(self) -> None:
        """
        Unfreezes the current module by setting its `frozen` flag to False.

        Returns:
            None.
        """

        # Set frozen indicator to false for current layer
        self.frozen = False

    @abstractmethod
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler divergence between
        posterior and prior distributions for the module's
        learnable parameters.

        Returns:
            A tuple containing:
                - kl_cost: The Kullback-Leibler divergence.
                - num_params: The number of contributing parameters.
        """
