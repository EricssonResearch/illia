# Libraries
from typing import Literal

import tensorflow as tf

from illia.nn.torch.base import BayesianModule


class KLDivergenceLoss(tf.keras.layers.Layer):

    reduction: Literal["mean"]
    weight: float

    def __init__(self, reduction: Literal["mean"] = "mean", weight: float = 1.0):

        super().__init__()

        self.reduction = reduction
        self.weight = weight

    def get_config(self):

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "reduction": self.reduction,
            "weight": self.weight,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, model: tf.keras.Model) -> tf.Tensor:
        """
        This method computes the forward for KLDivergenceLoss

        Args:
            model: tensorflow model.
            reduction: type of reduction for the loss. Defaults to "mean".
            weight: weight for the loss. Defaults to 1.0.

        Returns:
            kl divergence cost
        """
        kl_global_cost = tf.constant(0.0, dtype=tf.float32)
        num_params_global = 0

        # iterate through the model's layers
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


class ELBOLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        loss_function: tf.keras.losses.Loss,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.num_samples = num_samples
        self.kl_weight = kl_weight
        self.kl_loss = KLDivergenceLoss(weight=kl_weight)

    def call(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, model: tf.keras.Model
    ) -> tf.Tensor:
        loss_value = tf.constant(0.0, dtype=tf.float32)

        for _ in range(self.num_samples):
            current_loss = self.loss_function(y_true, y_pred) + self.kl_loss(model)
            loss_value += current_loss

        loss_value = tf.divide(loss_value, tf.cast(self.num_samples, tf.float32))

        return loss_value

    def get_config(self):
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
