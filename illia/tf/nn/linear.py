"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.tf.nn import BayesianModule
from illia.tf.distributions import (
    Distribution,
    GaussianDistribution,
)


@saving.register_keras_serializable(package="BayesianModule", name="Linear")
class Linear(BayesianModule):
    """
    Bayesian Linear layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        This is the constructor of the Linear class.

        Args:
            input_size: Input size of the linear layer.
            output_size: Output size of the linear layer.
            weights_distribution: Distribution for the weights of the
                layer.
            bias_distribution: Distribution for the bias of the layer.
        """

        # Call super-class constructor
        super().__init__()

        # Set parameters
        self.input_size = input_size
        self.output_size = output_size
        self.w: tf.Variable
        self.b: tf.Variable

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                (input_size, output_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution((output_size,))
        else:
            self.bias_distribution = bias_distribution

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the linear layer.

        Args:
            input_shape: The shape of the input tensor.
        """

        # Register non-trainable variables
        self.w = self.add_weight(
            name="weights",
            initializer=tf.constant_initializer(self.weights_distribution.sample()),
            shape=(self.input_size, self.output_size),
            trainable=False,
        )

        self.b = self.add_weight(
            name="bias",
            initializer=tf.constant_initializer(self.bias_distribution.sample()),
            shape=(self.output_size,),
            trainable=False,
        )

        super().build(input_shape)

    def get_config(self) -> dict:
        """
        Retrieves the configuration of the Linear layer.

        Returns:
            Dictionary containing layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weights_distribution": self.weights_distribution,
            "bias_distribution": self.bias_distribution,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the Bayesian Linear layer.

        Samples weights and bias from their posterior distributions if
        the layer is not frozen. If frozen and not initialized, samples
        them once.

        Args:
            inputs: input tensor. Dimensions: [batch, *].

        Raises:
            ValueError: Module has been frozen with undefined weights.

        Returns:
            Output tensor after linear transformation.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.w.assign(self.weights_distribution.sample())
            self.b.assign(self.bias_distribution.sample())
        elif self.w is None or self.b is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Compute outputs
        lin_output = tf.linalg.matmul(inputs, self.w)
        outputs: tf.Tensor = tf.nn.bias_add(lin_output, self.b)

        return outputs

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        log_probs: tf.Tensor = self.weights_distribution.log_prob(
            self.w
        ) + self.bias_distribution.log_prob(self.b)

        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
