"""
This module contains the code for the Losses.

Implements the KL divergence loss and the Evidence Lower Bound
(ELBO) loss for Bayesian neural networks in TensorFlow.
"""

# Standard libraries
from typing import Any, Callable, Literal

# 3pps
import tensorflow as tf
from keras import losses, saving

# Own modules
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="KLDivergenceLoss")
class KLDivergenceLoss(losses.Loss):
    """
    Computes the KL divergence loss across all Bayesian modules
    in a model.

    Supports configurable reduction and scaling.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the KL divergence loss.

        Args:
            reduction: Reduction method for loss. Only "mean"
                currently supported.
            weight: Weight to scale the KL divergence.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def get_config(self) -> dict:
        """
        Returns the configuration of the KL divergence loss.

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
        Computes KL divergence across Bayesian modules in the model.

        Args:
            *args: Unused positional arguments.
            **kwargs: Must include 'model' as a keyword argument.

        Returns:
            Scalar tensor representing KL loss.
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


@saving.register_keras_serializable(package="BayesianModule", name="ELBOLoss")
class ELBOLoss(losses.Loss):
    """
    Computes the Evidence Lower Bound (ELBO) loss.

    Combines a reconstruction loss and KL divergence, optionally
    using Monte Carlo sampling.
    """

    def __init__(
        self,
        loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ELBO loss function.

        Args:
            loss_function: Callable computing likelihood loss.
            num_samples: Number of samples for MC estimation.
            kl_weight: Scaling factor for KL divergence term.
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
        Returns the configuration of the ELBO loss.

        Returns:
            Dictionary with configuration values.
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

    def __call__(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, *args: Any, **kwargs: Any
    ) -> tf.Tensor:
        """
        Computes the ELBO loss using KL regularization and reconstruction error.

        Args:
            y_true: Ground truth targets.
            y_pred: Model predictions.
            *args: Unused positional arguments.
            **kwargs: Must include 'model' containing Bayesian layers.

        Returns:
            Scalar tensor representing the total ELBO loss.
        """

        model = kwargs.get("model")
        if model is None:
            raise ValueError("Model must be provided as a keyword argument")

        loss_value: tf.Tensor = tf.constant(0.0, dtype=tf.float32)

        for _ in range(self.num_samples):
            current_loss = self.loss_function(y_true, y_pred) + self.kl_loss(model)
            loss_value += current_loss

        # Average the loss across samples
        loss_value = tf.divide(loss_value, tf.cast(self.num_samples, tf.float32))

        return loss_value
