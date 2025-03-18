# Libraries
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from . import (
    StaticDistribution,
    DynamicDistribution,
    StaticGaussianDistribution,
    DynamicGaussianDistribution,
    BayesianModule,
)


class Conv2d(BayesianModule):
    """
    Bayesian 2D Convolution layer with trainable weights and biases,
    supporting prior and posterior distributions.
    """

    input_channels: int
    output_channels: int
    weights_posterior: DynamicDistribution
    weights_prior: StaticDistribution
    bias_posterior: DynamicDistribution
    bias_prior: StaticDistribution
    weights: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
        groups: int = 1,
        weights_prior: Optional[StaticDistribution] = None,
        bias_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        bias_posterior: Optional[DynamicDistribution] = None,
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
            weights_prior: Prior distribution for weights.
            bias_prior: Prior distribution for bias.
            weights_posterior: Posterior distribution for weights.
            bias_posterior: Posterior distribution for bias.
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

        # Set parameters
        parameters = {"mean": 0, "std": 0.1}

        if weights_prior is None:
            self.weights_prior = StaticGaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        if bias_prior is None:
            self.bias_prior = StaticGaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.bias_prior = bias_prior

        if weights_posterior is None:
            if isinstance(kernel_size, int):
                self.weights_posterior = DynamicGaussianDistribution(
                    (output_channels, input_channels // groups, kernel_size)
                )
            else:
                self.weights_posterior = DynamicGaussianDistribution(
                    (output_channels, input_channels // groups, *kernel_size)
                )
        else:
            self.weights_posterior = weights_posterior

        if bias_posterior is None:
            self.bias_posterior = DynamicGaussianDistribution((output_channels,))
        else:
            self.bias_posterior = bias_posterior

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
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()
        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_posterior.sample()
                self.bias = self.bias_posterior.sample()

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

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
        """
        Computes the Kullback-Leibler (KL) divergence cost for the
        layer's weights and bias.

        Returns:
            Tuple containing KL divergence cost and total number of
            parameters.
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
