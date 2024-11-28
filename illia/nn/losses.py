# Libraries
from typing import Literal, Any

# Illia backend selection
from illia.backend import backend

if backend() == "torch":
    from torch.nn import Module as BackendModule
    from illia.nn.torch import losses
elif backend() == "tf":
    from tensorflow.keras import Model as BackendModule
    from illia.nn.tf import losses


class KLDivergenceLoss(BackendModule):

    reduction: Literal["mean"]
    weight: float

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
    ):
        """
        Definition of the KL Divergence Loss function.

        Args:
            reduction: Specifies the reduction to apply to the output.
            weight: Weight for the loss.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Call super class constructor
        super().__init__()

        # Define layer based on the imported library
        self.loss = losses.KLDivergenceLoss(reduction=reduction, weight=weight)

    def __call__(self, model: Any) -> Any:
        """
        Call the underlying layer with the given inputs to apply the loss operation.

        Args:
            model: The model used to apply the loss.

        Returns:
            The output of the loss operation.
        """

        return self.loss(model)


class ELBOLoss(BackendModule):

    def __init__(
        self,
        loss_function: Any,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
    ) -> None:
        """
        Initializes the Evidence Lower Bound (ELBO) loss function.

        Args:
            loss_function: The loss function to be used for computing the reconstruction loss.
            num_samples: The number of samples to draw for estimating the ELBO.
            kl_weight: The weight applied to the KL divergence.

        Raises:
            ValueError: If an invalid backend is provided.
        """

        # Call super class constructor
        super().__init__()

        # Define layer based on the imported library
        self.loss = losses.ELBOLoss(
            loss_function=loss_function,
            num_samples=num_samples,
            kl_weight=kl_weight,
        )

    def __call__(self, y_true: Any, y_pred: Any, y_model: Any) -> Any:
        """
        Calls the ELBO loss function with the provided inputs.

        Args:
            y_true: The true values of the data.
            y_pred: The predicted values of the data.
            y_model: The model's output.

        Returns:
            The computed ELBO loss value.
        """

        return self.loss(y_true, y_pred, y_model)
