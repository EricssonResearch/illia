# Libraries
from abc import ABC
from typing import Literal, Optional, Union

from tensorflow.keras.losses import Loss
from torch.nn import Module

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
            reduction (Literal[&quot;mean&quot;], optional): _description_. Defaults to "mean".
            weight (float, optional): _description_. Defaults to 1.0.
            backend (Optional[str], optional): _description_. Defaults to "torch".

        Raises:
            ValueError: _description_
        """
        # Call super class constructor
        super(KLDivergenceLoss, self).__init__()

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import losses
        elif self.backend == "tf":
            # Import tensorflow part
            from illia.nn.tf import losses
        else:
            raise ValueError("Invalid backend value")
        
        # Define layer based on the imported library
        self.loss = losses.KLDivergenceLoss(
            reduction=reduction, 
            weight=weight
        )

    def __call__(self, inputs):
        return self.layer(inputs)


class ELBOLoss(ABC):

    def __init__(
        self,
        loss_function: Union[Loss, Module],
        num_samples: int = 1,
        kl_weight: float = 1e-3,
        backend: Optional[str] = "torch",
    ) -> None:
        """
        Definition of the ELBO Loss function.

        Args:
            loss_function (Union[Loss, Module]): _description_
            num_samples (int, optional): _description_. Defaults to 1.
            kl_weight (float, optional): _description_. Defaults to 1e-3.
            backend (Optional[str], optional): _description_. Defaults to "torch".

        Raises:
            ValueError: _description_
        """
    
        # Call super class constructor
        super(ELBOLoss, self).__init__()

        # Set attributes
        self.backend = backend

        # Choose backend
        if self.backend == "torch":
            # Import torch part
            from illia.nn.torch import losses
        elif self.backend == "tf":
            # Import tensorflow part
            from illia.nn.tf import losses
        else:
            raise ValueError("Invalid backend value")
        
        # Define layer based on the imported library
        self.loss = losses.ELBOLoss(
            loss_function=loss_function, 
            num_samples=num_samples,
            kl_weight=kl_weight
        )

    def __call__(self, inputs):
        return self.layer(inputs)
