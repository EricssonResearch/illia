"""
This module contains the base class for distribution modules.

It defines a standard interface for sampling, evaluating log
probabilities, and querying the number of parameters in
distribution layers built with Keras's `layers.Layer`.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="BayesianModule", name="DistributionModule")
class DistributionModule(layers.Layer, ABC):
    """
    Base class for all distribution modules using Keras's layers API.

    Any subclass must implement sampling, log-probability computation,
    and report the number of learnable parameters in the distribution.

    Notes:
        This class is abstract and should not be instantiated directly.
    """

    @abstractmethod
    def sample(self) -> tf.Tensor:
        """
        Samples a tensor from the distribution.

        Returns:
            A sampled tensor with the same shape as the distribution's
            parameters (e.g., mean and std).
        """

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the log-probability of a given sample.

        If no input is provided, a sample is drawn internally from the
        distribution before computing its log-probability.

        Args:
            x: Optional sample tensor.

        Returns:
            A scalar tensor representing the log-probability.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Returns the number of learnable parameters in the distribution.

        Returns:
            An integer representing the number of parameters.
        """
