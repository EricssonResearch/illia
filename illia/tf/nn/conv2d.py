"""
This module contains the code for the bayesian Conv2D.
"""

# Standard libraries
from typing import Optional, Union

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.tf.nn.base import BayesianModule
from illia.tf.distributions import GaussianDistribution


@saving.register_keras_serializable(package="BayesianModule", name="Conv2D")
class Conv2D(BayesianModule):
    """
    This class is the bayesian implementation of the Conv2D class.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, list[int]],
        stride: Union[int, list[int]] = 1,
        padding: Union[str, list[int]] = "VALID",
        dilation: Union[int, list[int]] = 1,
        num_groups: int = 1,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Deafults to 1.
            padding: Padding added to all four sides of the input.
                Defaults to 0.
            dilation: Spacing between kernel elements.
            num_groups: Number of blocked connections from input channels
                to output channels. Defaults to 1.
            weights_distribution: The distribution for the weights.
            bias_distribution: The distribution for the bias.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.num_groups = num_groups

        # Check if kernel_size is a list and unpack it if necessary
        kernel_shape = (
            kernel_size if isinstance(kernel_size, list) else [kernel_size, kernel_size]
        )
        self.shape = (input_channels // num_groups, *kernel_shape, output_channels)

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(shape=self.shape)
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution = GaussianDistribution((output_channels,))
        else:
            self.bias_distribution = bias_distribution

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the Conv2D layer.

        Args:
            input_shape: Input shape of the layer.
        """

        # Register non-trainable variables
        self.w = self.add_weight(
            name="weights",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=self.shape,
            trainable=False,
        )
        self.b = self.add_weight(
            name="bias",
            initializer=tf.constant_initializer(
                self.bias_distribution.sample().numpy()
            ),
            shape=(self.output_channels,),
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
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "num_groups": self.num_groups,
            "weights_distribution": self.weights_distribution,
            "bias_distribution": self.bias_distribution,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def _conv2d(
        self,
        inputs: tf.Tensor,
        weight: tf.Tensor,
        stride: Union[int, list[int]],
        padding: Union[str, list[int]],
        dilation: Union[int, list[int]],
    ) -> tf.Tensor:
        """
        Applies a 2D convolution operation to the input tensor.

        Args:
            inputs: The input tensor of shape
                [batch_size, height, width, channels].
            weight: The convolutional kernel weights.
            stride: The stride of the convolution.
            padding: The padding to be applied to the input tensor.
            dilation: The dilation rate of the convolution.

        Returns:
            The output tensor of shape
            [batch_size, height, width, output_channels].
        """

        output: tf.Tensor = tf.nn.conv2d(
            input=inputs,
            filters=weight,
            strides=stride,
            padding=padding,
            dilations=dilation,
        )

        return output

    def freeze(self) -> None:
        """
        This method is to freeze the layer.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.w is None:
            self.w = self.weights_distribution.sample()  # pylint: disable=W0201

        # Sample bias is they are undefined
        if self.b is None:
            self.b = self.bias_distribution.sample()  # pylint: disable=W0201

    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs
        log_probs: tf.Tensor = self.weights_distribution.log_prob(
            self.w
        ) + self.bias_distribution.log_prob(self.b)

        # Compute the number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs a forward pass through the Bayesian Convolution 2D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input tensor to the layer. Dimensions: [batch,
                input channels, input width, input height].

        Returns:
            Output tensor after passing through the layer. Dimensions:
                [batch, output channels, output width, output height].
        """

        # Check if layer is frozen
        if not self.frozen:
            self.w = self.weights_distribution.sample()
            self.b = self.bias_distribution.sample()
        elif self.w is None or self.b is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Compute outputs
        outputs: tf.Tensor = self._conv2d(
            inputs=inputs,
            weight=self.w,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Add bias
        outputs = tf.nn.bias_add(outputs, self.b)

        return outputs
