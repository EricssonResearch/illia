# Libraries
from typing import Optional, Union

import tensorflow as tf
from keras import saving

from . import (
    StaticDistribution,
    DynamicDistribution,
    StaticGaussianDistribution,
    DynamicGaussianDistribution,
    BayesianModule,
)


@saving.register_keras_serializable(package="illia_tf", name="Conv2d")
class Conv2d(BayesianModule):
    """
    Bayesian 2D Convolution layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    input_channels: int
    output_channels: int
    weights_posterior: DynamicDistribution
    weights_prior: StaticDistribution
    bias_posterior: DynamicDistribution
    bias_prior: StaticDistribution
    padding: Union[list[list[int]], str]
    weights: tf.Tensor
    bias: tf.Tensor

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int], str] = "valid",
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1,
        weights_prior: Optional[StaticDistribution] = None,
        bias_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        bias_posterior: Optional[DynamicDistribution] = None,
        channel_first: Optional[bool] = True,
    ) -> None:
        """
        Initializes a Bayesian 2D Convolution layer with specified
        parameters and distributions.

        Args:
            input_channels: Number of channels in the input image.
            output_channels: Number of channels produced by the
                convolution.
            kernel_size: Size of the convolving kernel, can be a single
                integer or a tuple of two integers.
            stride: Stride of the convolution, can be a single integer
                or a tuple of two integers.
            padding: Padding added to all four sides of the input, can
                be an integer, tuple, or a string ('valid' or 'same').
            dilation: Spacing between kernel elements, can be a single
                integer or a tuple of two integers.
            groups: Number of blocked connections from input channels
                to output channels.
            weights_prior: Prior distribution for the weights.
            bias_prior: Prior distribution for the bias.
            weights_posterior: Posterior distribution for the weights.
            bias_posterior: Posterior distribution for the bias.
            channel_first: Whether the input and output tensors should
                be channel-first.
        """

        super().__init__()

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = self.output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.data_format = "NCHW" if channel_first else "NHWC"

        # Checks in the groups
        if groups > 1:
            assert (
                input_channels % groups == 0
            ), "Input channels must be divisible by groups"
            assert (
                output_channels % groups == 0
            ), "Output channels must be divisible by groups"

        if isinstance(padding, int):
            self.padding = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
            pad_height, pad_width = padding
            self.padding = [
                [0, 0],
                [0, 0],
                [pad_height, pad_height],
                [pad_width, pad_width],
            ]
        elif isinstance(padding, str):
            assert padding.upper() in (
                "SAME",
                "VALID",
            ), 'Padding arg must be either "SAME" or "VALID"'
            self.padding = padding.upper()
        else:
            raise TypeError(
                "Padding must be an int, tuple, or string 'SAME' or 'VALID'"
            )

        parameters = {"mean": 0, "std": 0.1}

        # Prior
        self.weights_prior = (
            StaticGaussianDistribution(mu=parameters["mean"], std=parameters["std"])
            if weights_prior is None
            else weights_prior
        )
        self.bias_prior = (
            StaticGaussianDistribution(mu=parameters["mean"], std=parameters["std"])
            if bias_prior is None
            else bias_prior
        )

        # Posterior
        if weights_posterior is None:
            if isinstance(kernel_size, int):
                self.weights_posterior = DynamicGaussianDistribution(
                    (output_channels, input_channels // groups, kernel_size)
                )
            else:
                self.weights_posterior = DynamicGaussianDistribution(
                    (output_channels, input_channels // groups, *kernel_size)
                )
        else:
            self.weights_posterior = weights_posterior

        self.bias_posterior = (
            DynamicGaussianDistribution((output_channels,))
            if bias_posterior is None
            else bias_posterior
        )

    def __conv__(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.groups > 1:
            # Split input and filters into groups
            input_splits = tf.split(inputs, num_or_size_splits=self.groups, axis=1)
            filter_splits = tf.split(
                self.weights, num_or_size_splits=self.groups, axis=1
            )

            # Perform convolution for each group
            group_convolutions = [
                tf.nn.conv2d(
                    input_split,
                    filters=filter_split,
                    strides=self.stride,
                    padding=self.padding,
                    dilations=self.dilation,
                    data_format=self.data_format,
                )
                for input_split, filter_split in zip(input_splits, filter_splits)
            ]

            conv_output = tf.concat(group_convolutions, axis=-1)
        else:
            conv_output = tf.nn.conv2d(
                inputs,
                filters=self.weights,
                strides=self.stride,
                padding=self.padding,
                data_format=self.data_format,
                dilations=self.dilation,
            )

        outputs: tf.Tensor = tf.nn.bias_add(conv_output, self.bias)

        return outputs

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Performs forward pass to compute conv2d based on input tensor.

        Args:
            inputs: Input tensor for the layer.

        Returns:
            Tensor containing the layer's output.
        """

        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        elif self.weights is None or self.bias is None:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()

        return self.__conv__(inputs)

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            tuple containing KL divergence cost and total number of
            parameters.
        """

        log_posterior: tf.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: tf.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params

    def get_config(self):
        """
        Retrieves the configuration of the Conv layer.

        Returns:
            Dictionary containing layer configuration.
        """

        # Get the base configuration
        base_config = super().get_config()

        # Add the custom configurations
        custom_config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "weights_prior": self.weights_prior,
            "weights_posterior": self.weights_posterior,
            "bias_prior": self.bias_prior,
            "bias_posterior": self.bias_posterior,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilatation": self.dilation,
            "groups": self.groups,
            "data_format": self.data_format,
        }

        # Combine both configurations
        return {**base_config, **custom_config}
