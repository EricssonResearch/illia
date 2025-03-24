"""
This module contains auxiliary code for tests.torch.nn.
"""

# 3pps
import torch

# Own modules
from illia.torch.nn.base import BayesianModule


class ComposedModel(torch.nn.Module):
    """
    This class implements a composed model from an initial one.
    """

    def __init__(self, model: torch.nn.Module, num_models: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            model: Initial model to construct the composed one.
            num_models: Number of models to use.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set up models
        self.models = torch.nn.Sequential(*[model for _ in range(num_models)])

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, *].

        Returns:
            Outputs tensor. Dimensions: [batch, *].
        """

        outputs: torch.Tensor = torch.sum(
            torch.stack([model(inputs) for _, model in enumerate(self.models)]), dim=1
        )
        return outputs


class BayesianComposedModel(BayesianModule):
    """
    This class implements a composed bayesian model from an initial one.
    """

    def __init__(self, model: torch.nn.Module, num_models: int) -> None:
        """
        This method is the constructor of the class.

        Args:
            model: Initial model to construct the composed one.
            num_models: Number of models to use.

        Returns:
            None.
        """

        # Call super class constructor
        super().__init__()

        # Set up models
        self.models = torch.nn.Sequential(*[model for _ in range(num_models)])

        return None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: Inputs tensor. Dimensions: [batch, *].

        Returns:
            Outputs tensor. Dimensions: [batch, *].
        """

        outputs: torch.Tensor = torch.sum(
            torch.stack([model(inputs) for _, model in enumerate(self.models)]), dim=1
        )
        return outputs
