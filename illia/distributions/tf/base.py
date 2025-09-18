# Standard libraries
from abc import ABC, abstractmethod
from typing import Optional

# 3pps
import tensorflow as tf
from keras import layers, saving


@saving.register_keras_serializable(package="illia", name="DistributionModule")
class DistributionModule(layers.Layer, ABC):
    """
    Abstract base for probabilistic distribution modules in Tensorflow.
    Defines the required interface for sampling, computing
    log-probabilities, and counting learnable parameters.

    Notes:
        This class is abstract and cannot be instantiated directly.
        All abstract methods must be implemented by subclasses.
    """

    @abstractmethod
    def sample(self) -> tf.Tensor:
        """
        Draw a sample from the distribution.

        Returns:
            tf.Tensor: A sample drawn from the distribution.
        """

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute the log-probability of a provided sample. If no
        sample is passed, one is drawn internally.

        Args:
            x: Optional sample to evaluate. If None, a new sample is
                drawn from the distribution.

        Returns:
            tf.Tensor: Scalar log-probability value.

        Notes:
            Works with both user-supplied and internally drawn
            samples.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Return the number of learnable parameters in the
        distribution.

        Returns:
            int: Total count of learnable parameters.
        """
