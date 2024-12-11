# Libraries
from typing import Optional, Union

# 3pp
import torch
import torch.nn.functional as F

# own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.distributions import (
    Distribution,
    GaussianDistribution,
)


class Conv2d(BayesianModule):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]],
        padding: Union[int, tuple[int, int]],
        dilation: Union[int, tuple[int, int]],
        groups: int = 1,
        weights_distribution: Optional[Distribution] = None,
        bias_distribution: Optional[Distribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels: Number of channels in the input image.
            output_channels: Number of channels produced by the
                convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels
                to output channels.
            weights_prior: The prior distribution for the weights.
            bias_prior: The prior distribution for the bias.
            weights_posterior: The posterior distribution for the
                weights.
            bias_posterior: The posterior distribution for the bias.
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

        # set weights distribution
        if weights_distribution is None:
            # extend kernel if we only have 1 value
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)

            # define weights distribution
            self.weights_distribution: Distribution = GaussianDistribution(
                (output_channels, input_channels // groups, *kernel_size)
            )
        else:
            self.weights_distribution = weights_distribution

        # sample initial weights
        weights = self.weights_distribution.sample()
        bias = self.bias_distribution.sample()

        # register buffers
        self.register_buffer("weights", weights)
        self.register_buffer("bias", bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Bayesian Convolution 2D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective posterior distributions. If the layer is
        frozen and the weights or bias are not initialized, it samples
        them from their respective posterior distributions.

        Args:
            inputs: Input tensor to the layer.

        Returns:
            Output tensor after passing through the layer.
        """

        # forward depending of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()
        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_distribution.sample()
                self.bias = self.bias_distribution.sample()

        # Run torch forward
        return F.conv2d(
            inputs,
            weight=self.weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def kl_cost(self) -> tuple[torch.Tensor, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the
        weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights
                and bias, and the total number of parameters.
        """

        log_posterior: torch.Tensor = self.weights_posterior.log_prob(
            self.weights
        ) + self.bias_posterior.log_prob(self.bias)
        log_prior: torch.Tensor = self.weights_prior.log_prob(
            self.weights
        ) + self.bias_prior.log_prob(self.bias)

        num_params: int = (
            self.weights_posterior.num_params + self.bias_posterior.num_params
        )

        return log_posterior - log_prior, num_params
