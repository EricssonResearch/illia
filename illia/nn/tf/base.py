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
        Freezes the layer parameters by stopping gradient computation.
        If the weights or bias are not already sampled, they are sampled
        before freezing. Once frozen, no further sampling occurs.

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
        Computes the KL divergence cost for weights and bias.

        Returns:
            A tuple containing:
                - KL divergence cost.
                - Total number of parameters in the layer.
        """
