"""
This module implements the Kullback-Leibler (KL) divergence
loss for Bayesian neural networks in Tensorflow.
"""

# Standard libraries
from typing import Any, Literal

# 3pps
import tensorflow as tf
from keras import losses, saving

# Own modules
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="illia", name="KLDivergenceLoss")
class KLDivergenceLoss(losses.Loss):
    """
    Computes the Kullback-Leibler divergence loss across
    all Bayesian modules.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Kullback-Leibler divergence loss computation.

        Args:
            reduction: Reduction method for the loss.
            weight: Scaling factor applied to the total KL loss.
            **kwargs: Additional keyword arguments.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def get_config(self) -> dict:
        """
        Returns the configuration of the Kullback-Leibler
        divergence loss.

        Returns:
            Dictionary containing configuration values.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add custom configurations
        custom_config = {"reduction": self.reduction, "weight": self.weight}

        # Combine both configurations
        return {**base_config, **custom_config}

    def __call__(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        """
        Computes Kullback-Leibler divergence for all Bayesian
        modules in the model.

        Args:
            *args: Unused positional arguments.
            **kwargs: Must include 'model' as a keyword argument.

        Returns:
            Scaled Kullback-Leibler divergence loss as a scalar array.
        """

        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model must be provided as a keyword argument")

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
