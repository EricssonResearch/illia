# Libraries
from typing import Optional, Tuple, Union

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable  # type: ignore

from . import (
    Distribution,
    GaussianDistribution,
    BayesianModule,
)


@register_keras_serializable(package="BayesianModule", name="Conv2d")
class Conv2d(BayesianModule):
    """
    This class is the bayesian implementation of the Conv2d class with backend TF.

    Attr:
        weights_distribution: distribution for the weights of the
            layer.
            Dimensions: [output channels, input channels // groups, kernel size, kernel size].
        bias_distribution: distribution of the bias layer.
            Dimensions: [output channels].
        weights: sampled weights of the layer. They are registered in
            the buffer.
            Dimensions: [output channels, input channels // groups, kernel size, kernel size].
        bias: sampled bias of the layer. They are registered in
            the buffer.
            Dimensions: [output channels].

        input_channels: Number of channels in the input image.
        output_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Padding added to all four sides of the input.
        dilation_rate: Spacing between kernel elements.
        groups: Number of blocked connections from input channels
            to output channels.
        channel_first: Structure of data to train with. If True, NCHW. Else, NHWC.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = "valid",
        dilation_rate: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
        data_format: Optional[str] = "NHWC",
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels produced by the convolution.
            kernel_size (Union[int, Tuple[int, int]]):
                Size of the convolving kernel.
            stride (Union[int, Tuple[int, int]]):
                Stride of the convolution.
            padding (Union[int, Tuple[int, int], str]):
                Either the `string` `"SAME"` or `"VALID"` or explicity padding.
                When explicit padding is used and data_format is
                `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
                [pad_left, pad_right], [0, 0]]`
                Default configuration valid.
            dilation_rate (Union[int, Tuple[int, int]]):
                Spacing between kernel elements.
            groups (int, optional):
                Number of blocked connections from input channels to output channels.
                Defaults to 1.
            weights_distribution (Optional[Distribution], optional):
                The distribution for the weights.
                Defaults to None.
            bias_distribution (Optional[Distribution], optional):
                The distribution for the bias.
                Defaults to None.
            data_format (str, optional):
                Whether the input and output tensors should be channel-first.
                Defaults to False.

        Note:
            Data format internally will always operate with channel last (default in TF).
            https://www.tensorflow.org/api_docs/python/tf/nn/convolution

        """

        super().__init__()

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = self.output_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.stride = stride
        self.groups = groups
        self.data_format = data_format

        if groups > 1:
            assert (
                input_channels % groups == 0
            ), "Input channels must be divisible by groups"

            # ## IF PASSED DISTRIBUTION (BY DEFAULT THE SHAPE IS ALREADY DIVIDED)
            # if weights_distribution is None:
            assert self.filters % groups == 0, "Filters must be divisible by groups"

        if isinstance(padding, int):
            self.padding = [[0, 0], [0, 0], [padding, padding], [padding, padding]]
        elif isinstance(padding, tuple):
            padH, padW = padding
            self.padding = [[0, 0], [0, 0], [padH, padH], [padW, padW]]
        else:
            assert padding.upper() in (
                "SAME",
                "VALID",
            ), f'Padding arg must be either "SAME" or "VALID"'
            self.padding = padding.upper()

        # Distribution Initialization (parameters by default)
        weights_distribution_shape = (
            *self.kernel_size,
            input_channels // groups,
            output_channels,
        )
        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                weights_distribution_shape, name="weights_distr"
            )
        else:
            assert (
                weights_distribution.sample().shape == weights_distribution_shape
            ), f"""Expected shape  {weights_distribution_shape}, sampled shape {weights_distribution.sample().shape}"""
            self.weights_distribution = weights_distribution

        bias_distribution_shape = (output_channels,)
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution(
                bias_distribution_shape, name="bias_distr"
            )
        else:
            assert (
                bias_distribution.sample().shape == bias_distribution_shape
            ), f"""Expected shape  {bias_distribution_shape}, sampled shape {bias_distribution.sample().shape}"""

            self.bias_distribution = bias_distribution

        # Sample initial distributions
        self.kernels = self.add_weight(
            name="kernels",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=weights_distribution_shape,
            trainable=False,
        )
        self.bias = self.add_weight(
            name="bias",
            initializer=tf.constant_initializer(
                self.bias_distribution.sample().numpy()
            ),
            shape=bias_distribution_shape,
            trainable=False,
        )

    @staticmethod
    def channels_first_to_last(shape: list):
        """
        Given a shape with channels first, returns a shape with channels last.
        Args:
            shape: A sequence with each element corresponding to an axis in a shape.
        Returns:
            The transposed shape.
        """
        return shape[:1] + shape[2:] + shape[1:2]

    @staticmethod
    def channels_last_to_first(shape: list):
        """
        Given a shape with channels last, returns a shape with channels first.
        Args:
            shape:
        Return:
            The transposed shape.
        """
        return shape[:1] + shape[-1:] + shape[1:-1]

    def maybe_transpose_tensor(self, tensor: tf.Tensor):
        """Transpose if data format does not start with NC (channel-first).
        Args:
            Tensor of 4D
        Returns:
            Channel last tensor.
        """
        if self.data_format in ("NCHW", "channel_first"):
            order = self.channels_last_to_first(list(range(tensor.shape.rank)))
            return tf.transpose(a=tensor, perm=order)
        else:
            return tensor

    def __conv__(self, inputs: tf.Tensor) -> tf.Tensor:
        # Unfortunately 'channels_first' data format is not implemented on the CPU
        inputs = self.maybe_transpose_tensor(inputs)
        conv_output = tf.nn.convolution(
            inputs,
            filters=self.kernels,
            strides=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilations=self.dilation_rate,
        )

        outputs: tf.Tensor = tf.nn.bias_add(conv_output, self.bias)

        return outputs

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self.frozen:
            self.kernels.assign(self.weights_distribution.sample())
            self.bias.assign(self.bias_distribution.sample())
        else:
            if self.kernels is None:
                w = self.weights_distribution.sample()
                self.kernels = self.add_weight(
                    name="kernel",
                    initializer=tf.constant_initializer(w.numpy()),
                    shape=w.shape,
                    trainable=False,
                )
            if self.bias is None:
                b = self.bias_distribution.sample()
                self.add_weight(
                    name="bias",
                    initializer=tf.constant_initializer(b.numpy()),
                    shape=b.shape,
                    trainable=False,
                )
        return self.__conv__(inputs)

    @tf.function
    def kl_cost(self) -> Tuple[tf.Tensor, int]:
        log_posterior: tf.Tensor = self.weights_distribution.log_prob(
            self.kernels
        ) + self.bias_distribution.log_prob(self.bias)

        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )
        return log_posterior, num_params

    def get_config(self):
        """
        TODO: Review what are the necessary parameters (appart from init)
        """
        base_config = super().get_config()
        custom_config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "weights_distribution": self.weights_distribution,
            "bias_distribution": self.bias_distribution,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation_rate": self.dilation_rate,
            "groups": self.groups,
            "data_format": self.data_format,
        }
        # Combine both configurations
        return {**base_config, **custom_config}
