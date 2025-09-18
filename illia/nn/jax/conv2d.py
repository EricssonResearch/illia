# Standard libraries
from typing import Any, Optional

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax.gaussian import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Conv2d(BayesianModule):
    """
    Bayesian 2D convolutional layer with optional weight and bias priors.
    Behaves like a standard Conv2d but treats weights and bias as random
    variables sampled from specified distributions. Parameters become fixed
    when the layer is frozen.
    """

    bias_distribution: Optional[GaussianDistribution] = None
    bias: Optional[nnx.Param] = None

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        rngs: Rngs = nnx.Rngs(0),
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian 2D convolutional layer.

        Args:
            input_channels: Number of input feature channels.
            output_channels: Number of output feature channels.
            kernel_size: Convolution kernel size. Int is converted to tuple.
            stride: Stride of the convolution operation.
            padding: Tuple specifying zero-padding for height and width.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections between input and output.
            weights_distribution: Distribution to initialize weights.
            bias_distribution: Distribution to initialize bias.
            use_bias: Whether to include a bias term.
            rngs: Random number generators for reproducibility.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.rngs = rngs

        # Set weights distribution
        if weights_distribution is None:
            # Extend kernel if we only have 1 value
            if isinstance(self.kernel_size, int):
                self.kernel_size = (self.kernel_size, self.kernel_size)

            self.weights_distribution: GaussianDistribution = GaussianDistribution(
                shape=(
                    self.output_channels,
                    self.input_channels // self.groups,
                    *self.kernel_size,
                ),
                rngs=self.rngs,
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias prior
        if self.use_bias:
            if bias_distribution is None:
                # Define weights distribution
                self.bias_distribution = GaussianDistribution(
                    shape=(self.output_channels,),
                    rngs=self.rngs,
                )
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None

        # Sample initial weights
        self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample initial bias only if using bias
        if self.use_bias and self.bias_distribution is not None:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        else:
            self.bias = None

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
        if self.weights is None:
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias is None and self.bias_distribution is not None:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Stop gradient computation
        self.weights = jax.lax.stop_gradient(self.weights)
        if self.use_bias:
            self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[jax.Array, int]: A tuple containing the KL divergence
                cost and the total number of parameters in the layer.
        """

        # Compute log probs for weights
        log_probs: jax.Array = self.weights_distribution.log_prob(
            jnp.asarray(self.weights)
        )

        # Add bias log probs only if using bias
        if (
            self.use_bias
            and self.bias is not None
            and self.bias_distribution is not None
        ):
            log_probs += self.bias_distribution.log_prob(jnp.asarray(self.bias))

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias and self.bias_distribution is not None:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Performs a forward pass through the Bayesian Convolution 2D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input array with shape (batch, channels, height,
                width).

        Returns:
            Output array after convolution with optional bias addition.

        Raises:
            ValueError: If the layer is frozen but weights or bias are
                undefined.
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution is not None:
                self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        elif self.weights is None or (self.use_bias and self.bias is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute ouputs
        outputs = jax.lax.conv_general_dilated(
            lhs=inputs,
            rhs=jnp.asarray(self.weights),
            window_strides=self.stride,
            padding=[self.padding, self.padding],
            lhs_dilation=[1, 1],
            rhs_dilation=self.dilation,
            dimension_numbers=(
                "NCHW",  # Input
                "OIHW",  # Kernel
                "NCHW",  # Output
            ),
            feature_group_count=self.groups,
        )

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += jnp.reshape(
                a=jnp.asarray(self.bias), shape=(1, self.output_channels, 1, 1)
            )

        return outputs
