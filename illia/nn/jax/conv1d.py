"""
This module contains the code for the bayesian Conv1D.
"""

# Standard libraries
from typing import Any, Optional

# 3pps
import jax

# Own modules
from illia.distributions.jax import GaussianDistribution
from illia.nn.jax.base import BayesianModule


class Conv1D(BayesianModule):
    """
    This class is the bayesian implementation of the Conv1D class.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
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

        # Set weights prior
        if weights_distribution is None:
            self.weights_distribution: GaussianDistribution = GaussianDistribution(
                (
                    self.input_channels // self.groups,
                    self.kernel_size,
                    self.output_channels,
                )
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias prior
        if bias_distribution is None:
            self.bias_distribution: GaussianDistribution = GaussianDistribution(
                (self.output_channels,)
            )
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        self.weights = self.weights_distribution.sample()
        self.bias = self.bias_distribution.sample()

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
            self.weights = self.weights_distribution.sample()

            # Sample bias
            if self.backend_params["use_bias"]:
                self.bias = self.bias_distribution.sample()

        # Compute ouputs
        outputs = jax.lax.conv_general_dilated(
            lhs=inputs,
            rhs=self.weights,
            window_strides=[self.stride],
            padding=[(self.padding, self.padding)],
            lhs_dilation=[1],
            rhs_dilation=[self.dilation],
            dimension_numbers=(
                "NHC",
                "HIO",
                "NHC",
            ),  # (input, kernel, output) dimension
            feature_group_count=self.groups,
        )
        if self.backend_params["use_bias"]:
            outputs += self.bias

        return outputs
