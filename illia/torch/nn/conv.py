# standard libraries
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
            weights_distribution: The distribution for the weights.
            bias_distribution: The distribution for the bias.
        """

        # Call super class constructor
        super().__init__()

        # set attributes
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

        # set bias distribution
        if bias_distribution is None:
            # define weights distribution
            self.bias_distribution: Distribution = GaussianDistribution(
                (output_channels,)
            )
        else:
            self.bias_distribution = bias_distribution

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

        # forward depending of frozen state
        if not self.frozen:
            self.weights = self.weights_distribution.sample()
            self.bias = self.bias_distribution.sample()
        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_distribution.sample()
                self.bias = self.bias_distribution.sample()

        # execute torch forward
        return F.conv2d(
            inputs,
            weight=self.weights,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

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
        Calculate the Kullback-Leibler (KL) divergence cost for the
        weights and bias of the layer.

        Returns:
            KL divergence cost. Dimensions: [].
            number of parameters.
        """

        # compute log probs
        log_probs: torch.Tensor = self.weights_distribution.log_prob(
            self.weights
        ) + self.bias_distribution.log_prob(self.bias)

        # compute number of parameters
        num_params: int = (
            self.weights_distribution.num_params + self.bias_distribution.num_params
        )

        return log_probs, num_params
