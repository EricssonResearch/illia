"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Literal

# 3pps
import tensorflow as tf
from keras import Model, saving, losses

# Own modules
from illia.tf.nn.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="KLDivergenceLoss")
class KLDivergenceLoss(Model):
    """
    Computes the KL divergence loss for Bayesian modules within a model.
    """

    def __init__(self, reduction: Literal["mean"] = "mean", weight: float = 1.0):
        """
        Initializes the KL divergence loss with specified reduction
        method and weight.

        Args:
            reduction: Method to reduce the loss, currently only "mean"
                is supported.
            weight: Scaling factor for the KL divergence loss.
        """

        # Call super class constructor
        super().__init__()

        # Set atributtes
        self.reduction = reduction
        self.weight = weight

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the KL divergence loss.

        Returns:
            Dictionary containing loss configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "reduction": self.reduction,
            "weight": self.weight,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, model: Model) -> tf.Tensor:
        """
        Computes the KL divergence loss across all Bayesian layers in
        the model.

        Args:
            model: TensorFlow model containing Bayesian layers.

        Returns:
            KL divergence cost scaled by the specified weight.
        """

        kl_global_cost = tf.constant(0.0, dtype=tf.float32)
        num_params_global = 0

        # Iterate through the model's layers
        for layer in model.layers:
            if isinstance(layer, BayesianModule):
                kl_cost, num_params = layer.kl_cost()
                kl_global_cost += kl_cost
                num_params_global += num_params

        kl_global_cost = tf.divide(
            kl_global_cost, tf.cast(num_params_global, tf.float32)
        )
        kl_global_cost = tf.multiply(kl_global_cost, self.weight)

        return kl_global_cost


@saving.register_keras_serializable(package="illia_tf", name="ELBOLoss")
class ELBOLoss(Model):
    """
    Computes the Evidence Lower Bound (ELBO) loss, combining a
    likelihood loss and KL divergence.
    """

    def __init__(
        self,
        loss_function: losses.Loss,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ):
        """
        Initializes the ELBO loss with specified likelihood loss
        function, sample count, and KL weight.

        Args:
            loss_function: Loss function for computing likelihood loss.
            num_samples: Number of samples for Monte Carlo
                approximation.
            kl_weight: Scaling factor for the KL divergence component.
        """

        # Call super class constructor
        super().__init__()

        # Set atributtes
        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

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

        loss_value = tf.constant(0.0, dtype=tf.float32)

        for _ in range(self.num_samples):
            current_loss = self.loss_function(y_true, y_pred) + self.kl_loss(model)
            loss_value += current_loss

        loss_value = tf.divide(loss_value, tf.cast(self.num_samples, tf.float32))

        return loss_value

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the ELBO loss.

        Returns:
            Dictionary containing ELBO loss configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "loss_function": self.loss_function,
            "num_samples": self.num_samples,
            "kl_weight": self.kl_weight,
        }

        # Combine both configurations
        return {**base_config, **custom_config}
