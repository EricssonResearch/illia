# Standard libraries
from typing import Any, Optional

# 3pps
import tensorflow as tf
from keras import saving

# Own modules
from illia.distributions.tf.gaussian import GaussianDistribution
from illia.nn.tf.base import BayesianModule


@saving.register_keras_serializable(package="illia", name="Conv1d")
class Conv1d(BayesianModule):
    """
    Bayesian 1D convolutional layer with optional weight and bias priors.
    Behaves like a standard Conv1d but treats weights and bias as random
    variables sampled from specified distributions. Parameters become fixed
    when the layer is frozen.
    """

    bias_distribution: Optional[GaussianDistribution] = None

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "VALID",
        dilation: int = 1,
        groups: int = 1,
        data_format: Optional[str] = "NWC",
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian 1D convolutional layer.

        Args:
            input_channels: Number of channels in the input.
            output_channels: Number of channels produced by the conv.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding type, 'VALID' or 'SAME'.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections between input/output.
            data_format: 'NWC' or 'NCW' format for input data.
            weights_distribution: Distribution for weights sampling.
            bias_distribution: Distribution for bias sampling.
            use_bias: Whether to include a bias term.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        super().__init__(**kwargs)

        # Check data format
        self._check_params(kernel_size, groups, stride, dilation, data_format)

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

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
        if self.use_bias:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution((output_channels,))
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None

    def _check_params(
        self,
        kernel_size: int,
        groups: int,
        stride: int,
        dilation: int,
        data_format: Optional[str],
    ) -> None:
        """
        Validates convolution parameters for correctness.

        Args:
            kernel_size: Convolution kernel size.
            groups: Number of blocked connections.
            stride: Convolution stride.
            dilation: Spacing between kernel elements.
            data_format: 'NWC' or 'NCW' for input tensor.

        Raises:
            ValueError: If any parameter is invalid.
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
            shape=self._weights_distribution_shape,
            trainable=False,
        )

        if self.use_bias and self.bias_distribution is not None:
            self.b = self.add_weight(
                name="bias",
                initializer=tf.constant_initializer(
                    self.bias_distribution.sample().numpy()
                ),
                shape=(self.output_channels,),
                trainable=False,
            )

        super().build(input_shape)

    def get_config(self) -> dict:
        """
        Return the configuration dictionary for serialization.

        Returns:
            dict: Dictionary with the layer configuration.
        """

        base_config = super().get_config()

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

        return {**base_config, **custom_config}

    def _conv1d(
        self,
        inputs: tf.Tensor,
        weight: tf.Tensor,
        stride: int | list[int],
        padding: str,
        data_format: Optional[str] = "NWC",
        dilation: Optional[int | list[int]] = None,
    ) -> tf.Tensor:
        """
        Performs a 1D convolution using provided weights.

        Args:
            inputs: Input tensor.
            weight: Convolutional kernel tensor.
            stride: Convolution stride.
            padding: Padding strategy 'VALID' or 'SAME'.
            data_format: 'NWC' or 'NCW' input format.
            dilation: Spacing between kernel elements.

        Returns:
            Tensor after 1D convolution.
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
        Performs a forward pass through the Bayesian Convolution 1D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input tensor to the layer with shape
                (batch, length, output_channels) if 'data_format' is
                'NWC' or (batch, output_channels, length) if
                'data_format' is 'NCW'

        Returns:
            Output tensor after convolution with optional bias added.

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
        outputs: tf.Tensor = self._conv1d(
            inputs=inputs,
            weight=self.w,
            stride=self.stride,
            padding=self.padding,
            data_format=self.data_format,
            dilation=self.dilation,
        )

        # Add bias only if using bias
        if self.use_bias and self.b is not None:
            outputs = tf.nn.bias_add(
                value=outputs,
                bias=self.b,
                data_format="N..C" if self.data_format == "NWC" else "NC..",
            )

        return outputs
