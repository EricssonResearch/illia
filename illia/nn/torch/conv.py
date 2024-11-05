# Libraries
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

import illia.distributions.static as static
from illia.nn import conv
from illia.nn.torch.base import BayesianModule
from illia.distributions.static import StaticDistribution
from illia.distributions.dynamic import (
    DynamicDistribution,
    GaussianDistribution,
)


class Conv2d(conv.Conv2d, BayesianModule):
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
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        parameters = {"mean": 0, "std": 0.1}

        if weights_prior is None:
            self.weights_prior = static.GaussianDistribution(parameters)
        else:
            self.weights_prior = weights_prior

        if bias_prior is None:
            self.bias_prior = static.GaussianDistribution(parameters)
        else:
            self.bias_prior = bias_prior

        if weights_posterior is None:
            if isinstance(kernel_size, int):
                self.weights_posterior = GaussianDistribution(
                    (output_channels, input_channels // groups, kernel_size)
                )
            else:
                self.weights_posterior = GaussianDistribution(
                    (output_channels, input_channels // groups, *kernel_size)
                )
        else:
            self.weights_posterior = weights_posterior

        if bias_posterior is None:
            self.bias_posterior = GaussianDistribution((output_channels,))
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
