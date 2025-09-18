# Standard libraries
from typing import Any, Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.distributions.torch.gaussian import GaussianDistribution
from illia.nn.torch.base import BayesianModule


class Conv2d(BayesianModule):
    """
    This class is the bayesian implementation of the Conv2d class.
    """

    weights: torch.Tensor
    bias: torch.Tensor

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        weights_distribution: Optional[GaussianDistribution] = None,
        bias_distribution: Optional[GaussianDistribution] = None,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian 2D convolutional layer.

        Args:
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution. Deafults to 1.
            padding: Padding added to all four sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels
                to output channels.
            weights_distribution: The distribution for the weights.
            bias_distribution: The distribution for the bias.
            use_bias: Whether to include a bias term.
            **kwargs: Additional keyword arguments.

        Returns:
            None.

        Notes:
            If no distributions are provided, Gaussian distributions are
            used by default.
        """

        # Call super class constructor
        super().__init__(**kwargs)

        # Set attributes
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias

        # Set weights distribution
        if weights_distribution is None:
            # Extend kernel if we only have 1 value
            if isinstance(self.kernel_size, int):
                self.kernel_size = (self.kernel_size, self.kernel_size)

            # Define weights distribution
            self.weights_distribution = GaussianDistribution(
                (
                    self.output_channels,
                    self.input_channels // self.groups,
                    *self.kernel_size,
                )
            )
        else:
            self.weights_distribution = weights_distribution

        # Set bias distribution
        if self.use_bias:
            if bias_distribution is None:
                self.bias_distribution = GaussianDistribution((self.output_channels,))
            else:
                self.bias_distribution = bias_distribution
        else:
            self.bias_distribution = None  # type: ignore

        # Sample initial weights
        weights = self.weights_distribution.sample()

        # Register buffers
        self.register_buffer("weights", weights)

        if self.use_bias and self.bias_distribution is not None:
            bias = self.bias_distribution.sample()
            self.register_buffer("bias", bias)
        else:
            self.bias = None  # type: ignore

    @torch.jit.export
    def freeze(self) -> None:
        """
        Freezes the layer parameters by stopping gradient computation.
        If the weights or bias are not already sampled, they are sampled
        before freezing. Once frozen, no further sampling occurs.

        Returns:
            None.
        """

        # Set indicator
        self.frozen = True

        # Sample weights if they are undefined
        if self.weights is None:
            self.weights = self.weights_distribution.sample()

        # Sample bias if they are undefined and bias is used
        if self.use_bias and self.bias_distribution is not None:
            if not hasattr(self, "bias") or self.bias is None:
                self.bias = self.bias_distribution.sample()
            self.bias = self.bias.detach()

        # Detach weights and bias
        self.weights = self.weights.detach()

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
        log_probs: torch.Tensor = self.weights_distribution.log_prob(self.weights)

        # Add bias log probs if bias is used
        if self.use_bias and self.bias_distribution is not None:
            log_probs += self.bias_distribution.log_prob(self.bias)

        # Compute number of parameters
        num_params: int = self.weights_distribution.num_params()
        if self.use_bias and self.bias_distribution is not None:
            num_params += self.bias_distribution.num_params()

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

        # Check if layer is frozen
        if not self.frozen:
            self.weights = self.weights_distribution.sample()

            # Sample bias only if using bias
            if self.use_bias and self.bias_distribution is not None:
                self.bias = self.bias_distribution.sample()
        elif self.weights is None or (self.use_bias and self.bias is None):
            raise ValueError(
                "Module has been frozen with undefined weights and/or bias."
            )

        # Compute outputs
        # pylint: disable=E1102
        outputs: torch.Tensor = F.conv2d(
            input=inputs,
            weight=self.weights,
            bias=self.bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += torch.reshape(
                input=self.bias, shape=(1, self.output_channels, 1, 1)
            )

        return outputs
