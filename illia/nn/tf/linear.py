"""
This module contains the code for Linear Bayesian layer.
"""

# Standard libraries
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="Linear")
class Linear(BayesianModule):
    """
    This class is the bayesian implementation of the Linear class.
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
        This is the constructor of the Linear class.

        Args:
            input_size: Input size of the linear layer.
            output_size: Output size of the linear layer.
            weights_distribution: The Gaussian distribution for the
                weights, if applicable.
            bias_distribution: The Gaussian distribution for the bias,
                if applicable.
            **kwargs: Additional keyword arguments.
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
        Builds the Linear layer.

        Args:
            input_shape: Input shape of the layer.
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

        if self.use_bias and self.bias_distribution:
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
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w = self.weights_distribution.sample()

        # Sample bias is they are undefined
        if self.use_bias and self.bias_distribution:
            self.b = self.bias_distribution.sample()

        # Stop gradient computation (more similar to detach) weights and bias
        self.w = tf.stop_gradient(self.w)
        if self.use_bias:
            self.b = tf.stop_gradient(self.b)

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(self.w)

        # Add bias log probs only if using bias
        if self.use_bias and self.bias_distribution:
            log_probs += self.bias_distribution.log_prob(self.b)

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

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
            self.w = self.weights_distribution.sample()

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution:
                self.b = self.bias_distribution.sample()
        elif self.w is None or self.b is None:
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        outputs: tf.Tensor = tf.linalg.matmul(inputs, self.w, transpose_b=True)

        # Add bias only if using bias
        if self.use_bias is not None:
            outputs = tf.nn.bias_add(outputs, self.b)

        return outputs
