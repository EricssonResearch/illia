"""
This module contains the code for the bayesian Conv2D.
"""

# Standard libraries
from typing import Optional, Union

# 3pps
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.rnglib import Rngs

# Own modules
from illia.distributions.jax import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Conv2D(BayesianModule):
    """
    This class is the bayesian implementation of the Conv2D class.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: list[int] = [1, 1],
        padding: tuple[int, int] = (0, 0),
        dilation: list[int] = [1, 1],
        groups: int = 1,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        rngs: Rngs = nnx.Rngs(0),
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels: Number of input feature channels.
            output_channels: Number of output feature channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride of the convolution operation.
            padding: Tuple for zero-padding on both sides.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections between input and output.
            weights_distribution: Distribution to initialize weights.
            bias_distribution: Distribution to initialize bias.
            use_bias: Whether to include a bias term.
            rngs: Random number generators for reproducibility.
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
                self.bias_distribution: GaussianDistribution = GaussianDistribution(
                    shape=(self.output_channels,),
                    rngs=self.rngs,
                )
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None  # type: ignore

        # Sample initial weights
        self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample initial bias only if using bias
        if self.use_bias:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))
        else:
            self.bias = None  # type: ignore

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias is None:
            self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

        # Stop gradient computation (more similar to detach) weights and bias
        self.weights = jax.lax.stop_gradient(self.weights)
        if self.use_bias and self.bias is not None:
            self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs for weights
        log_probs: jax.Array = self.weights_distribution.log_prob(
            jnp.asarray(self.weights)
        )

        # Add bias log probs only if using bias
        if self.use_bias and self.bias is not None:
            log_probs += self.bias_distribution.log_prob(jnp.asarray(self.bias))

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params
        if self.use_bias:
            num_params += self.bias_distribution.num_params

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """
        Applies the convolution operation to the inputs using current weights
        and bias. If the model is not frozen, samples new weights and bias
        before computation.

        Args:
            inputs: Input array to be convolved.

        Returns:
            Output array after applying convolution and bias.
        """

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = nnx.Param(self.weights_distribution.sample(self.rngs))

            # Sample bias only if using bias
            if self.use_bias:
                self.bias = nnx.Param(self.bias_distribution.sample(self.rngs))

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
