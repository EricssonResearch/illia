# Standard libraries
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.gaussian import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="illia", name="Linear")
class Linear(BayesianModule):
    """
    Bayesian linear layer (fully connected) with optional weight and bias
    distributions. Can be frozen to stop gradient updates and fix
    parameters.
    """

    bias_distribution: Optional[GaussianDistribution] = None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian Linear layer.

        Args:
            input_size: Number of input features.
            output_size: Number of output features.
            weights_distribution: Distribution for the weights.
            bias_distribution: Distribution for the bias.
            use_bias: Whether to include a bias term.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        # Call super-class constructor
        super().__init__(**kwargs)

        # Set parameters
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution((output_size, input_size))
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if self.use_bias:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution((output_size,))
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build trainable and non-trainable parameters.

        Args:
            input_shape: Input shape used to trigger layer build.

        Returns:
            None
        """

        # Register non-trainable variables
        self.w = self.add_weight(
            name="weights",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=(self.output_size, self.input_size),
            trainable=False,
        )

        if self.use_bias and self.bias_distribution is not None:
            self.b = self.add_weight(
                name="bias",
                initializer=tf.constant_initializer(
                    self.bias_distribution.sample().numpy()
                ),
                shape=(self.output_size,),
                trainable=False,
            )

        # Call super-class build method
        super().build(input_shape)

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        Returns:
            dict: Dictionary with the layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def freeze(self) -> None:
        """
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w = self.weights_distribution.sample()

        # Sample bias is they are undefined
        if self.use_bias and self.b is None and self.bias_distribution is not None:
            self.b = self.bias_distribution.sample()

        # Stop gradient computation
        self.w = tf.stop_gradient(self.w)
        if self.use_bias:
            self.b = tf.stop_gradient(self.b)

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[tf.Tensor, int]: A tuple containing the KL divergence
                cost and the total number of parameters in the layer.
        """

        # Compute log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(self.w)

        # Add bias log probs only if using bias
        if self.use_bias and self.b is not None and self.bias_distribution is not None:
            log_probs += self.bias_distribution.log_prob(self.b)

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution is not None:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs forward pass using current weights and bias.

        Samples parameters if layer is not frozen. Raises an error if
        frozen weights are undefined.

        Args:
            inputs: Input tensor of shape [batch, features].

        Returns:
            Output tensor after linear transformation.
        
        Raises:
            ValueError: If the layer is frozen but weights or bias are
                undefined.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.w = self.weights_distribution.sample()

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution is not None:
                self.b = self.bias_distribution.sample()
        elif self.w is None or (self.use_bias and self.b is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        outputs: tf.Tensor = tf.linalg.matmul(inputs, self.w, transpose_b=True)

        # Add bias only if using bias
        if self.use_bias and self.b is not None:
            outputs = tf.nn.bias_add(outputs, self.b)

        return outputs
