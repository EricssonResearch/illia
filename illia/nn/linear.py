# Libraries
from typing import Optional, Tuple, Any

from illia.distributions.dynamic import DynamicDistribution
from illia.distributions.static import StaticDistribution
from illia.nn.base import BayesianModule

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from illia.nn.torch import linear
elif backend() == "tf":
    from illia.nn.tf import linear


class Linear(BayesianModule):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_prior: Optional[StaticDistribution] = None,
        bias_prior: Optional[StaticDistribution] = None,
        weights_posterior: Optional[DynamicDistribution] = None,
        bias_posterior: Optional[DynamicDistribution] = None,
    ) -> None:
        """
        Definition of a Bayesian Linear layer.

        Args:
            input_size: Size of each input sample.
            output_size: Size of each output sample.
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
        self.layer = linear.Linear(
            input_size=input_size,
            output_size=output_size,
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
