# Standard libraries
from typing import Any, Callable

# 3pps
import tensorflow as tf
from keras import losses, saving

# Own modules
from illia.losses.tf.kl import KLDivergenceLoss


@saving.register_keras_serializable(package="illia", name="ELBOLoss")
class ELBOLoss(losses.Loss):
    """
    Computes the Evidence Lower Bound (ELBO) loss for Bayesian networks.
    The ELBO loss combines a reconstruction loss with a KL divergence
    term. Monte Carlo sampling can be used to estimate the expected
    reconstruction loss over the model's stochastic layers.

    Notes:
        The KL term is weighted by `kl_weight`. This module assumes the
        model contains Bayesian layers compatible with
        `KLDivergenceLoss`.
    """

    def __init__(
        self,
        loss_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ELBO loss with sampling and KL regularization.

        Args:
            loss_function: Function for computing reconstruction loss.
            num_samples: Number of Monte Carlo samples for estimation.
            kl_weight: Weight applied to the KL divergence term.
            **kwargs: Additional arguments passed to the base class.

        Returns:
            None.
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
        Compute the ELBO loss using Monte Carlo sampling and KL
        regularization.

        Args:
            y_true: Ground truth targets.
            y_pred: Model predictions.
            *args: Unused positional arguments.
            **kwargs: Must include 'model' containing Bayesian layers.

        Returns:
            Scalar array representing the average ELBO loss.

        Notes:
            The loss is averaged over `num_samples` Monte Carlo draws.
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
