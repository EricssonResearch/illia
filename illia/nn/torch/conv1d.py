# Standard libraries
from typing import Any, Optional

# 3pps
import torch
import torch.nn.functional as F

# Own modules
from illia.distributions.torch.gaussian import GaussianDistribution
from illia.nn.torch.base import BayesianModule


class Conv1d(BayesianModule):
    """
    Bayesian 1D convolutional layer with optional weight and bias priors.
    Behaves like a standard Conv1d but treats weights and bias as random
    variables sampled from specified distributions. Parameters become fixed
    when the layer is frozen.
    """

    weights: torch.Tensor
    bias: torch.Tensor

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
        **kwargs: Any,
    ) -> None:
        """
        Initializes a Bayesian 1D convolutional layer.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            kernel_size: Size of the convolution kernel.
            stride: Stride of the convolution.
            padding: Padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections.
            weights_distribution: Distribution for the weights.
            bias_distribution: Distribution for the bias.
            use_bias: Whether to include a bias term.
            **kwargs: Extra arguments passed to the base class.

        Returns:
            None.

        Notes:
            Gaussian distributions are used by default if none are
            provided.
        """

        super().__init__(**kwargs)

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
            # Define weights distribution
            self.weights_distribution = GaussianDistribution(
                (
                    self.output_channels,
                    self.input_channels // self.groups,
                    self.kernel_size,
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
        Freeze the module's parameters to stop gradient computation.
        If weights or biases are not sampled yet, they are sampled first.
        Once frozen, parameters are not resampled or updated.

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
        Compute the KL divergence cost for all Bayesian parameters.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the KL
                divergence cost and the total number of parameters in
                the layer.
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
        Performs a forward pass through the Bayesian Convolution 1D
        layer. If the layer is not frozen, it samples weights and bias
        from their respective distributions. If the layer is frozen
        and the weights or bias are not initialized, it also performs
        sampling.

        Args:
            inputs: Input tensor to the layer with shape (batch,
                input channels, input width, input height).

        Returns:
            Output tensor after passing through the layer with shape
                (batch, output channels, output width, output height).

        Raises:
            ValueError: If the layer is frozen but weights or bias are
                undefined.
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
        outputs: torch.Tensor = F.conv1d(
            input=inputs,
            weight=self.weights,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Add bias only if using bias
        if self.use_bias and self.bias is not None:
            outputs += torch.reshape(
                input=self.bias, shape=(1, self.output_channels, 1)
            )

        return outputs
