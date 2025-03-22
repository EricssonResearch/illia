"""
This module contains the code for the bayesian Conv2d.
"""

# Standard libraries
from typing import Optional, Union

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.tf.nn import BayesianModule
from illia.tf.distributions import (
    Distribution,
    GaussianDistribution,
)


@saving.register_keras_serializable(package="BayesianModule", name="Conv2d")
class Conv2d(BayesianModule):
    """
    Bayesian 2D Convolution layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int], str] = "valid",
        dilation: Union[int, tuple[int, int]] = 1,
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
            kernel_size (Union[int, tuple[int, int]]):
                Size of the convolving kernel.
            stride (Union[int, tuple[int, int]]):
                Stride of the convolution.
            padding (Union[int, tuple[int, int], str]):
                Either the `string` `"SAME"` or `"VALID"` or explicity padding.
                When explicit padding is used and data_format is
                `"NHWC"`, this should be in the form `[[0, 0], [pad_top, pad_bottom],
                [pad_left, pad_right], [0, 0]]`
                Default configuration valid.
            dilation (Union[int, tuple[int, int]]):
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
            Data format internally will always operate with channel last
            https://www.tensorflow.org/api_docs/python/tf/nn/convolution
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filters = self.output_channels
        self.dilation = dilation
        self.stride = stride
        self.groups = groups
        self.data_format = data_format

        # Set kernel size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        # Check if groups are valid
        if groups > 1:
            assert (
                input_channels % groups == 0
            ), "Input channels must be divisible by groups"
            assert self.filters % groups == 0, "Filters must be divisible by groups"

        # Check padding type
        if isinstance(padding, int):
            self.padding_explicit: list[list[int]] = [
                [0, 0],
                [0, 0],
                [padding, padding],
                [padding, padding],
            ]
            self.padding = "EXPLICIT"
        elif isinstance(padding, tuple):
            pad_h, pad_w = padding
            self.padding_explicit = [[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]]
            self.padding = "EXPLICIT"
        else:
            assert padding.upper() in (
                "SAME",
                "VALID",
            ), 'Padding arg must be either "SAME" or "VALID"'
            self.padding = padding.upper()

        # Distribution initialization
        self._weights_distribution_shape = (
            *self.kernel_size,
            input_channels // groups,
            output_channels,
        )

        if weights_distribution is None:
            self.weights_distribution: Distribution = GaussianDistribution(
                self._weights_distribution_shape
            )
        else:
            assert (
                weights_distribution.sample().shape == self._weights_distribution_shape
            ), (
                f"Expected shape  {self._weights_distribution_shape}, "
                f"sampled shape {weights_distribution.sample().shape}"
            )
            self.weights_distribution = weights_distribution

        self._bias_distribution_shape = (output_channels,)
        if bias_distribution is None:
            self.bias_distribution: Distribution = GaussianDistribution(
                self._bias_distribution_shape
            )
        else:
            assert bias_distribution.sample().shape == self._bias_distribution_shape, (
                f"Expected shape  {self._bias_distribution_shape}, "
                f"sampled shape {bias_distribution.sample().shape}"
            )
            self.bias_distribution = bias_distribution

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the conv2d layer.

        Args:
            input_shape: The shape of the input tensor.
        """

        # Register non-trainable variables
        self.w = self.add_weight(
            initial_value=tf.constant_initializer(self.weights_distribution.sample()),
            trainable=False,
            name="weights",
            shape=self._weights_distribution_shape,
        )

        self.b = self.add_weight(
            initial_value=tf.constant_initializer(self.bias_distribution.sample()),
            trainable=False,
            name="bias",
            shape=self._bias_distribution_shape,
        )

        super().build(input_shape)

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
            dilations=self.dilation,
        )

        outputs: tf.Tensor = tf.nn.bias_add(conv_output, self.bias)

        return outputs

    @tf.function
    def kl_cost(self) -> tuple[tf.Tensor, int]:
        log_posterior: tf.Tensor = self.weights_distribution.log_prob(
            self.kernels
        ) + self.bias_distribution.log_prob(self.bias)

        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )
        return log_posterior, num_params

    def get_config(self):
        base_config = super().get_config()

        config = {
            "input_channels": self.input_channels,
            "output_channels": self.output_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
            "weights_distribution": self.weights_distribution,
            "bias_distribution": self.bias_distribution,
            "data_format": self.data_format,
        }

        # Combine both configurations
        return {**base_config, **config}

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        The call function is responsible for performing a forward pass
        through the Bayesian Convolutional layer.

        Args:
            inputs: The input tensor to the layer.

        Returns:
            The output tensor after the convolution operation.

        Raises:
            ValueError: If the layer is frozen and the weights or bias
                are not initialized.
        """

        if not self.frozen:
            self.kernels.assign(self.weights_distribution.sample())
            self.bias.assign(self.bias_distribution.sample())
        elif self.kernels is None or self.bias is None:
            raise ValueError("Module has been frozen with undefined weights")

        return self.__conv__(inputs)
