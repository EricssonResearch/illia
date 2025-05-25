"""
This module contains the code for the Losses.
"""

# Standard libraries
from typing import Literal

# 3rd party libraries
import tensorflow as tf
from keras import Model, losses, saving

# Own modules
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="KLDivergenceLoss")
class KLDivergenceLoss(losses.Loss):
    """
    Computes the KL divergence loss for Bayesian modules within a model.
    """

    def __init__(
        self, reduction: Literal["mean"] = "mean", weight: float = 1.0, **kwargs
    ) -> None:
        """
        Initializes the KL divergence loss with specified reduction
        method and weight.

        Args:
            reduction: Method to reduce the loss, currently only "mean"
                is supported.
            weight: Scaling factor for the KL divergence loss.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set parameters
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

        # Add custom configurations
        custom_config = {"reduction": self.reduction, "weight": self.weight}

        # Combine both configurations
        return {**base_config, **custom_config}

    def __call__(self, model: Model) -> tf.Tensor:
        """
        Computes the KL divergence loss across all Bayesian layers in
        the model.

        Args:
            model: TensorFlow model containing Bayesian layers.

        Returns:
            KL divergence cost scaled by the specified weight.
        """

        kl_global_cost: tf.Tensor = tf.constant(0.0, dtype=tf.float32)
        num_params_global: int = 0

        # Iterate through the model's layers
        for layer in model.layers:
            if isinstance(layer, BayesianModule):
                kl_cost, num_params = layer.kl_cost()
                kl_global_cost += kl_cost
                num_params_global += num_params

        # Compute mean KL cost and scale by weight
        kl_global_cost = tf.divide(
            kl_global_cost, tf.cast(num_params_global, tf.float32)
        )
        kl_global_cost = tf.multiply(kl_global_cost, self.weight)

        return kl_global_cost
