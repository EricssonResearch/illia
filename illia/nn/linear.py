# Libraries
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

from illia.distributions.dynamic import DynamicDistribution
from illia.distributions.static import StaticDistribution


class Linear(ABC):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_prior: Optional[StaticDistribution] = None,
        bias_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        bias_posterior: Optional[DynamicDistribution] = None,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Definition of a Bayesian Linear layer.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights.
            bias_prior (Optional[StaticDistribution], optional): The prior distribution for the bias.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights.
            bias_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the bias.
            backend (Optional[str], optional): The backend to use.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import linear
        else:
            raise ValueError("Invalid backend value")

        # Define layer based on the imported library
        self.layer = linear.Linear(
            input_size=input_size,
            output_size=output_size,
            weights_prior=weights_prior,
            bias_prior=bias_prior,
            weights_posterior=weights_posterior,
            bias_posterior=bias_posterior,
        )

    def __call__(self, inputs):
        """
        Call the underlying layer with the given inputs to apply the layer operation.

        Args:
            inputs (Any): The input data to the layer.

        Returns:
            output (Any): The output of the layer operation.
        """

        return self.layer(inputs)

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        """
        Calculate the Kullback-Leibler (KL) divergence cost for the weights and bias of the layer.

        Returns:
            Tuple[Any, int]: A tuple containing the KL divergence cost for the weights and bias, and the total number of parameters.
        """

        pass
