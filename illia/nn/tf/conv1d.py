"""
This module contains the code for the bayesian Conv1D.
"""

# Standard libraries
from typing import Any, Optional, Union

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="BayesianModule", name="Conv1D")
class Conv1D(BayesianModule):
    """
    This class is the bayesian implementation of the Conv1D class.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: Union[int, list[int]] = 1,
        padding: str = "VALID",
        dilation: Union[int, list[int]] = 1,
        groups: int = 1,
        data_format: Optional[str] = "NWC",
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian Conv1D layer.

        Args:
            input_channels: The number of channels in the input image.
            output_channels: The number of channels produced by the
                convolution.
            kernel_size: The size of the convolving kernel.
            stride: The stride of the convolution.
            padding: The padding added to all four sides of the input.
                Can be 'VALID' or 'SAME'.
            dilation: The spacing between kernel elements.
            groups: The number of blocked connections from input
                channels to output channels.
            data_format: The data format for the convolution, either
                'NWC' or 'NCW'.
            weights_distribution: The Gaussian distribution for the
                weights, if applicable.
            bias_distribution: The Gaussian distribution for the bias,
                if applicable.
            **kwargs: Additional keyword arguments.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Check data format
        self._check_params(kernel_size, groups, stride, dilation, data_format)

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # Adjust the weights distribution based on the channel format
        self.data_format = (
            "NWC" if data_format is None or data_format == "NWC" else "NCW"
        )

        # Get the weights distribution shape, needs to be channel last
        self._weights_distribution_shape = (
            input_channels // groups,
            kernel_size,
            output_channels,
        )

        # Set weights distribution
        if weights_distribution is None:
            self.weights_distribution = GaussianDistribution(
                self._weights_distribution_shape
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            self.bias_distribution = GaussianDistribution((output_channels,))
        else:
            self.bias_distribution = bias_distribution

    def _check_params(self, kernel_size, groups, stride, dilation, data_format) -> None:
        """
        Checks the validity of the parameters for the convolution
        operation.

        Args:
            kernel_size: The size of the convolving kernel.
            groups: The number of blocked connections from input
                channels to output channels.
            stride: The stride of the convolution.
            dilation: The spacing between kernel elements.
            data_format: The data format for the convolution, either
                "NHWC" or "NCHW".

        Raises:
            ValueError: If the kernel size is invalid, the groups is
                invalid, the stride is invalid, the dilation is
                invalid, or the data format is invalid.
        """

        if kernel_size is not None and (kernel_size <= 0 or kernel_size % groups != 0):
            raise ValueError(
                f"Invalid `kernel_size`: {kernel_size}. Must be > 0 "
                f"and divisible by `groups` {groups}."
            )
        if groups <= 0:
            raise ValueError(f"Invalid `groups`: {groups}. Must be > 0.")
        if isinstance(stride, list):
            if any(s == 0 for s in stride):
                raise ValueError(f"`stride` {stride} cannot contain 0.")
            if max(stride) > 1 and isinstance(dilation, list) and max(dilation) > 1:
                raise ValueError(
                    f"`stride` {stride} > 1 not allowed with `dilation` {dilation} > 1."
                )
        if data_format not in {"NWC", "NCW"}:
            raise ValueError(
                f"Invalid `data_format`: {data_format}. Must be 'NWC' or 'NCW'."
            )

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Builds the Conv1D layer.

        Args:
            input_shape: Input shape of the layer.
        """

        # Register non-trainable variables
        self.w = self.add_weight(
            name="weights",
            initializer=tf.constant_initializer(
                self.weights_distribution.sample().numpy()
            ),
            shape=self._weights_distribution_shape,
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
        Retrieves the configuration of the Conv1D layer.

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
            "groups": self.groups,
            "data_format": self.data_format,
        }

        # Combine both configurations
        return {**base_config, **custom_config}

    def _conv1d(
        self,
        inputs: tf.Tensor,
        weight: tf.Tensor,
        stride: Union[int, list[int]],
        padding: str,
        data_format: Optional[str] = "NWC",
        dilation: Optional[Union[int, list[int]]] = None,
    ) -> tf.Tensor:
        """
        Applies a 1D convolution operation to the input tensor.

        Args:
            inputs: The input tensor.
            weight: The convolutional kernel weights.
            stride: The stride of the convolution.
            padding: The padding strategy to be applied, either
                'VALID' or 'SAME'.
            data_format: The data format for the input tensor, either
                'NWC' or 'NCW'.
            dilation: The dilation rate for spacing between kernel
                elements.

        Returns:
            The output tensor after applying the 1D convolution.
        """

        output: tf.Tensor = tf.nn.conv1d(
            input=inputs,
            filters=weight,
            stride=stride,
            padding=padding,
            data_format=data_format,
            dilations=dilation,
        )

        return output

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
        if self.b is None:
            self.b = self.bias_distribution.sample()

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
        Performs a forward pass through the Bayesian Convolution 1D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input tensor to the layer.

        Returns:
            Output tensor after passing through the layer.
        """

        # Check if layer is frozen
        if not self.frozen:
            self.w = self.weights_distribution.sample()
            self.b = self.bias_distribution.sample()
        elif self.w is None or self.b is None:
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        outputs: tf.Tensor = self._conv1d(
            inputs=inputs,
            weight=self.w,
            stride=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilation=self.dilation,
        )

        # Add bias
        outputs = tf.nn.bias_add(
            value=outputs,
            bias=self.b,
            data_format="N..C" if self.data_format == "NWC" else "NC..",
        )

        return outputs
