# Libraries
from abc import ABC
from typing import Literal, Optional, Any


class KLDivergenceLoss(ABC):

    def __init__(
        self,
        reduction: Literal["mean"] = "mean",
        weight: float = 1.0,
        backend: Optional[str] = "torch",
    ):
        """
        Definition of the KL Divergence Loss function.

        Args:
            reduction (Literal[&quot;mean&quot;], optional): Specifies the reduction to apply to the output. Defaults to "mean".
            weight (float, optional): Weight for the loss. Defaults to 1.0.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend value is provided.
        """

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import losses  # type: ignore
        elif self.backend == "tf":
            # Import tensorflow part
            from illia.nn.tf import losses  # type: ignore
        else:
            raise ValueError("Invalid backend value")

        # Define layer based on the imported library
        self.loss = losses.KLDivergenceLoss(reduction=reduction, weight=weight)

    def __call__(self, inputs):
        return self.loss(inputs)


class ELBOLoss(ABC):

    def __init__(
        self,
        loss_function: Any,
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Initializes the Evidence Lower Bound (ELBO) loss function.

        Args:
            loss_function (Union[Any, Any]): The loss function to be used for computing the reconstruction loss.
            num_samples (int, optional): The number of samples to draw for estimating the ELBO. Defaults to 1.
            kl_weight (float, optional): The weight applied to the KL divergence. Defaults to 1e-3.
            backend (Optional[str], optional): The backend to use. Defaults to 'torch'.

        Raises:
            ValueError: If an invalid backend is provided.
        """

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import losses  # type: ignore
        elif self.backend == "tf":
            # Import tensorflow part
            from illia.nn.tf import losses  # type: ignore
        else:
            raise ValueError("Invalid backend value")

        # Define layer based on the imported library
        self.loss = losses.ELBOLoss(
            loss_function=loss_function,
            num_samples=num_samples,
            kl_weight=kl_weight,
        )

    def __call__(self, inputs):
        return self.loss(inputs)
