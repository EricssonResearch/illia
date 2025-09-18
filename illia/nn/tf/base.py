# Standard libraries
from abc import ABC, abstractmethod
from typing import Any

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="illia", name="BayesianModule")
class BayesianModule(layers.Layer, ABC):
    """
    Abstract base for Bayesian-aware modules in Tensorflow.
    Provides mechanisms to track if a module is Bayesian and control
    parameter updates through freezing/unfreezing.

    Notes:
        All derived classes must implement `freeze` and `kl_cost` to
        handle parameter management and compute the KL divergence cost.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the Bayesian module with default flags.
        Sets `frozen` to False and `is_bayesian` to True.

        Args:
            **kwargs: Additional keyword arguments passed to the base class.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.frozen: bool = False
        self.is_bayesian: bool = True

    @abstractmethod
    def freeze(self) -> None:
        """
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.

        Notes:
            Must be implemented by all subclasses.
        """

    def unfreeze(self) -> None:
        """
        Unfreeze the module by setting its `frozen` flag to False.
        Allows parameters to be sampled and updated again.

        Returns:
            None.
        """

        self.frozen = False

    @abstractmethod
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[tf.Tensor, int]: A tuple containing the KL divergence
                cost and the total number of parameters in the layer.

        Notes:
            Must be implemented by all subclasses.
        """
