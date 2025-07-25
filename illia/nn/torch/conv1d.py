"""
This module contains the code for the bayesian Conv1D.
"""

# Standard libraries
from typing import Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.distributions.torch import GaussianDistribution
from illia.nn.torch.base import BayesianModule


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
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels: Number of channels in the input image.
            output_channels: Number of channels produced by the
                convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Deafults to 1.
            padding: Padding added to all four sides of the input.
                Defaults to 0.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels
                to output channels. Defaults to 1.
            weights_distribution: The distribution for the weights.
            bias_distribution: The distribution for the bias.
        """

        # Call super class constructor
        super().__init__()

        # Set attributes
        self.conv_params: tuple[int, ...] = (stride, padding, dilation, groups)

        # Set weights distribution
        if weights_distribution is None:
            # Define weights distribution
            self.weights_distribution: GaussianDistribution = GaussianDistribution(
                (output_channels, input_channels // groups, kernel_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            # Define weights distribution
            self.bias_distribution: GaussianDistribution = GaussianDistribution(
                (output_channels,)
            )
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        weights = self.weights_distribution.sample()
        bias = self.bias_distribution.sample()

        # Register buffers
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the current module and all submodules that are instances
        of BayesianModule. Sets the frozen state to True.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:  # type: ignore
            self.weights = self.weights_distribution.sample()

        # Sample bias is they are undefined
        if self.bias is None:  # type: ignore
            self.bias = self.bias_distribution.sample()

        # Detach weights and bias
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # Compute number of parameters
        num_params: int = (
            self.weights_distribution.num_params() + self.bias_distribution.num_params()
        )

        return log_probs, num_params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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

        # Forward depending of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()
        elif self.weights is None or self.bias is None:
            raise ValueError("Module has been frozen with undefined weights")

        # Execute torch forward
        return F.conv1d(
            inputs, self.weights, self.bias, *self.conv_params  # type: ignore
        )
