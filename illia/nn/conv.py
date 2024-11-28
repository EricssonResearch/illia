# Libraries
from typing import Optional, Tuple, Any, Union

from illia.distributions.dynamic import DynamicDistribution
from illia.distributions.static import StaticDistribution
from illia.nn.base import BayesianModule

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from illia.nn.torch import conv
elif backend() == "tf":
    from illia.nn.tf import conv


class Conv2d(BayesianModule):

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
        Definition of a Bayesian Convolution 2D layer.

        Args:
            input_channels: Number of channels in the input image.
            output_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Padding added to all four sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input channels to output channels.
            weights_prior: The prior distribution for the weights.
            bias_prior: The prior distribution for the bias.
            weights_posterior: The posterior distribution for the weights.
            bias_posterior: The posterior distribution for the bias.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super().__init__()

        # Define layer based on the imported library
        self.layer = conv.Conv2d(
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weights_prior=weights_prior,
            bias_prior=bias_prior,
            weights_posterior=weights_posterior,
            bias_posterior=bias_posterior,
        )

    def __call__(self, inputs: Any) -> Any:
        """
        Call the underlying layer with the given inputs to apply the layer operation.

        Args:
            inputs: The input data to the layer.

        Returns:
            The output of the layer operation.
        """

        return self.layer(inputs)

    def kl_cost(self) -> Tuple[Any, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Returns:
            A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        return self.layer.kl_cost()
