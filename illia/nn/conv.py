# Libraries
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Union

from illia.distributions.dynamic import DynamicDistribution
from illia.distributions.static import StaticDistribution


class Conv2d(ABC):

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
        backend: Optional[str] = "torch",
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
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super(Conv2d, self).__init__()

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import conv
        else:
            raise ValueError("Invalid backend value")
        
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

    def __call__(self, inputs):
        return self.layer(inputs)
        
    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
