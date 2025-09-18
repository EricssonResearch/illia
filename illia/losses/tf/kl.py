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
    Compute Kullback-Leibler divergence across Bayesian modules.
    This loss sums the KL divergence from all Bayesian layers in
    the model. It can be reduced by averaging and scaled by a
    weight factor.

    Notes:
        Assumes the model contains submodules derived from
        `BayesianModule`.
    """

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the KL divergence loss.

        Args:
            reduction: Method used to reduce the KL loss.
            weight: Scaling factor for the KL divergence.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.reduction = reduction
        self.weight = weight

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        Returns:
            dict: Dictionary containing the KL loss configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add custom configurations
        custom_config = {"reduction": self.reduction, "weight": self.weight}

        # Combine both configurations
        return {**base_config, **custom_config}

    def __call__(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        """
        Compute KL divergence for all Bayesian modules in a model.

        Args:
            *args: Unused positional arguments.
            **kwargs: Must include 'model' containing Bayesian
                layers.

        Returns:
            tf.Tensor: Weighted KL divergence loss.

        Notes:
            The KL loss is averaged over the number of parameters
            and scaled by the `weight` attribute.
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
