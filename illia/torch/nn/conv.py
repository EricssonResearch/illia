from typing import Optional, Union

import torch
import torch.nn.functional as F

from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import (
    Distribution,
    GaussianDistribution,
)


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
        padding: Union[int, tuple[int, int], str] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        groups: int = 1,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        Initializes a Bayesian 2D Convolution layer with specified
        parameters and distributions.

        Args:
            input_channels: Number of channels in the input image.
            output_channels: Number of channels produced by the
                convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output
                channels.
            weights_distribution: The distribution for the weights.
            bias_distribution: The distribution for the bias.
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

        # Set weights distribution
        if weights_distribution is None:
            # Extend kernel if we only have 1 value
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            # Define weights distribution
            self.weights_distribution: Distribution = GaussianDistribution(
                (output_channels, input_channels // groups, *kernel_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if bias_distribution is None:
            # Define weights distribution
            self.bias_distribution: Distribution = GaussianDistribution(
                (output_channels,)
            )
        else:
            self.bias_distribution = bias_distribution

        # Sample initial weights
        self.weights = self.weights_distribution.sample()
        self.bias = self.bias_distribution.sample()

        # Register buffers
        self.register_buffer("weights", self.weights)
        self.register_buffer("bias", self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian 2D Convolution
        layer.

        Samples weights and bias from their posterior distributions if
        the layer is not frozen. If frozen and not initialized, samples
        them once.

        Args:
            inputs: Input tensor to the layer.

        Returns:
            Output tensor after convolution operation.
        """

        # Forward depeding of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()
        elif self.weights is None or self.bias is None:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()

        # Run torch forward
        outputs: torch.Tensor = F.conv2d(
            inputs,
            weight=self.weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return outputs

    @torch.jit.export
    def freeze(self) -> None:
        """
        This method is to freeze the layer.

        Returns:
            None.
        """

        # set indicator
        self.frozen = True

        # sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()

        # sample bias is they are undefined
        if self.bias is None:
            self.bias = self.bias_distribution.sample()

        # detach weights and bias
        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    @torch.jit.export
    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            tuple containing KL divergence cost and total number of
            parameters.
        """

        # Compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # Compute number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
