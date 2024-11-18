# Libraries
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import illia.distributions.static as static
import illia.distributions.dynamic as dynamic
from illia.nn.torch.base import BayesianModule


class Conv2d(BayesianModule):

    input_channels: int
    output_channels: int
    weights_posterior: dynamic.DynamicDistribution
    weights_prior: static.StaticDistribution
    bias_posterior: dynamic.DynamicDistribution
    bias_prior: static.StaticDistribution
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
        weights_prior: Optional[static.StaticDistribution] = None,
        bias_prior: Optional[static.StaticDistribution] = None,
        weights_posterior: Optional[dynamic.DynamicDistribution] = None,
        bias_posterior: Optional[dynamic.DynamicDistribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels (int): Number of channels in the input image.
            output_channels (int): Number of channels produced by the convolution.
            kernel_size (Union[int, Tuple[int, int]]): Size of the convolving kernel.
            stride (Union[int, Tuple[int, int]]): Stride of the convolution.
            padding (Union[int, Tuple[int, int]]): Padding added to all four sides of the input.
            dilation (Union[int, Tuple[int, int]]): Spacing between kernel elements.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights. Defaults to None.
            bias_prior (Optional[StaticDistribution], optional): The prior distribution for the bias. Defaults to None.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights. Defaults to None.
            bias_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the bias. Defaults to None.
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

        parameters = {"mean": 0, "std": 0.1}

        if weights_prior is None:
            self.weights_prior = static.GaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.weights_prior = weights_prior

        if bias_prior is None:
            self.bias_prior = static.GaussianDistribution(
                mu=parameters["mean"], std=parameters["std"]
            )
        else:
            self.bias_prior = bias_prior

        if weights_posterior is None:
            if isinstance(kernel_size, int):
                self.weights_posterior = dynamic.GaussianDistribution(
                    (output_channels, input_channels // groups, kernel_size)
                )
            else:
                self.weights_posterior = dynamic.GaussianDistribution(
                    (output_channels, input_channels // groups, *kernel_size)
                )
        else:
            self.weights_posterior = weights_posterior

        if bias_posterior is None:
            self.bias_posterior = dynamic.GaussianDistribution((output_channels,))
        else:
            self.bias_posterior = bias_posterior

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            self.weights = self.weights_posterior.sample()
            self.bias = self.bias_posterior.sample()

        else:
            if self.weights is None or self.bias is None:
                self.weights = self.weights_posterior.sample()
                self.bias = self.bias_posterior.sample()

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

    def kl_cost(self) -> Tuple[torch.Tensor, int]:
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
