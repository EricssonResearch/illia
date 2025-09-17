"""
Abstract base class for Bayesian layers using TensorFlow.
Provides common functionality for identifying Bayesian modules,
freezing/unfreezing parameters, and computing KL divergence costs.
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
    Abstract base for Bayesian-aware modules in TensorFlow.
    Any Bayesian layer should inherit from this class. It tracks
    whether the module is Bayesian and provides freezing/unfreezing
    mechanisms for controlling parameter updates.

    Notes:
        Derived classes must implement `freeze` and `kl_cost`.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the module with default Bayesian-specific flags.
        Sets `frozen` to False and `is_bayesian` to True.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.

        Returns:
            None.
        """

        super().__init__(**kwargs)

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
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Compute the KL divergence between posterior and prior distributions.

        Returns:
            Tuple containing:
                - kl_cost: Kullback-Leibler divergence for this module.
                - num_params: Number of parameters contributing to KL.
        """
