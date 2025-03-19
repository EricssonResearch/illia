from abc import abstractmethod
from typing import Optional

import tensorflow as tf
from keras import Model


class Distribution(Model):
    """
    Abstract base class for probability distributions.
    """

    @abstractmethod
    def sample(self) -> tf.Tensor:
        """
        Generates a random sample from the distribution.

        Returns:
            A random sample as a tensor.
        """

    @abstractmethod
    def log_prob(self, x: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Computes the log probability of a given sample.

        Args:
            x: An optional sample for which to compute the log
                probability. If not provided, the method may compute
                the log probability for an internally stored sample.

        Returns:
            The log probability of the sample as a tensor.
        """

    @property
    @abstractmethod
    def num_params(self) -> int:
        """
        Number of parameters of the distribution.

        Returns:
            The number of parameters as an integer.
        """
