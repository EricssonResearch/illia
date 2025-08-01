"""
This module contains the code for the bayesian Conv2D.
"""

# Standard libraries
from typing import Any, Optional, Union

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

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.backend_params: dict[str, Any] = {
            "use_bias": use_bias,
        }

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rngs = rngs

        # Set weights distribution
        if weights_distribution is None:
            # Extend kernel if we only have 1 value
            if isinstance(self.kernel_size, int):
                self.kernel_size = (self.kernel_size, self.kernel_size)

            # Define weights distribution
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

        # Set bias distribution
        if bias_distribution is None:
            # Define weights distribution
            self.bias_distribution: GaussianDistribution = GaussianDistribution(
                shape=(self.output_channels,),
                rngs=self.rngs,
            )
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        self.weights = self.weights_distribution.sample(self.rngs)
        self.bias = self.bias_distribution.sample(self.rngs) if use_bias else None

    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = self.weights_distribution.sample(self.rngs)

        # Sample bias is they are undefined
        if self.bias is None and self.backend_params["use_bias"]:
            self.bias = self.bias_distribution.sample(self.rngs)

        # Stop gradient computation (more similar to detach) weights and bias
        self.weights = jax.lax.stop_gradient(self.weights)
        self.bias = jax.lax.stop_gradient(self.bias)

    def kl_cost(self) -> tuple[jax.Array, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs
        log_probs: jax.Array = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # Compute number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params

    def __call__(self, inputs: jax.Array) -> jax.Array:

        # Sample if model not frozen
        if not self.frozen:
            # Sample weights
            self.weights = self.weights_distribution.sample(self.rngs)

            # Sample bias
            if self.backend_params["use_bias"]:
                self.bias = self.bias_distribution.sample(self.rngs)

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

        if self.backend_params["use_bias"] and self.bias is not None:
            outputs += jnp.reshape(a=self.bias, shape=(1, self.output_channels, 1, 1))

        return outputs
