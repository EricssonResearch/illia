"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Callable

# 3rd party libraries
import tensorflow as tf
from keras import Model, losses, saving

# Own modules
from illia.losses.tf import KLDivergenceLoss


@saving.register_keras_serializable(package="BayesianModule", name="ELBOLoss")
class ELBOLoss(losses.Loss):
    """
    Computes the Evidence Lower Bound (ELBO) loss, combining a
    likelihood loss and KL divergence.
    """

    def __init__(
        self,
        loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        **kwargs,
    ) -> None:
        """
        Initializes the ELBO loss with specified likelihood loss
        function, sample count, and KL weight.

        Args:
            loss_function: Loss function for computing likelihood loss.
            num_samples: Number of samples for Monte Carlo approximation.
            kl_weight: Scaling factor for the KL divergence component.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the ELBO loss.

        Returns:
            Dictionary containing ELBO loss configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add custom configurations
        custom_config = {
            "loss_function": self.loss_function,
            "num_samples": self.num_samples,
            "kl_weight": self.kl_weight,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, model: Model) -> tf.Tensor:
        """
        Computes the ELBO loss, averaging over multiple samples.

        Args:
            y_true: True labels.
            y_pred: Predictions from the model.
            model: TensorFlow model containing Bayesian layers.

        Returns:
            Average ELBO loss across samples.
        """

        loss_value: tf.Tensor = tf.constant(0.0, dtype=tf.float32)

        for _ in range(self.num_samples):
            current_loss = self.loss_function(y_true, y_pred) + self.kl_loss(model)
            loss_value += current_loss

        # Average the loss across samples
        loss_value = tf.divide(loss_value, tf.cast(self.num_samples, tf.float32))

        return loss_value
