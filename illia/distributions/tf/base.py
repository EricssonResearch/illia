"""
Base class for distribution modules built using Tensorflow.

Provides a consistent interface for sampling, evaluating log
probabilities, and retrieving the number of parameters in custom
distribution layers.

Methods should be implemented by subclasses to define the actual
sampling behavior, probability evaluation, and parameter reporting.
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
    Abstract base for Tensorflow-based probabilistic distribution modules.

    Defines a required interface for sampling, computing log-probabilities,
    and retrieving parameter counts. Subclasses must implement all
    abstract methods to provide specific distribution logic.
        
    Notes:
        Avoid direct instantiation, this serves as a blueprint for
        derived classes.
    """

    @abstractmethod
    def sample(self) -> tf.Tensor:
        """
        Generates and returns a sample from the underlying distribution.

        Returns:
            Sample array matching the shape and structure defined by
            the distribution parameters.
        """

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the log-probability of an input sample.
        If no sample is provided, a new one is drawn internally from the
        current distribution before computing the log-probability.

        Args:
            x: Optional sample tensor to evaluate.

        Returns:
            Scalar tensor representing the computed log-probability.

        Notes:
            This method supports both user-supplied samples and internally
            generated ones for convenience when evaluating likelihoods.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Returns the total number of learnable parameters in the distribution.

        Returns:
            Integer representing the total number of learnable parameters.
        """
