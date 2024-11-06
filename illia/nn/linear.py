# Libraries
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

from illia.distributions import dynamic
from illia.distributions import static


class Linear(ABC):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        weights_prior: Optional[static.StaticDistribution] = None,
        bias_prior: Optional[static.StaticDistribution] = None,
        weights_posterior: Optional[dynamic.DynamicDistribution] = None,
        bias_posterior: Optional[dynamic.DynamicDistribution] = None,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Definition of a Bayesian Linear layer.

        Args:
            input_size (int): Size of each input sample.
            output_size (int): Size of each output sample.
            weights_prior (Optional[StaticDistribution], optional): The prior distribution for the weights. Defaults to None.
            bias_prior (Optional[StaticDistribution], optional): The prior distribution for the bias. Defaults to None.
            weights_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the weights. Defaults to None.
            bias_posterior (Optional[DynamicDistribution], optional): The posterior distribution for the bias. Defaults to None.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend value is provided.
        """
        
        # Call super class constructor
        super(Linear, self).__init__()

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
        return self.layer(inputs)

    @abstractmethod
    def kl_cost(self) -> Tuple[Any, int]:
        pass
