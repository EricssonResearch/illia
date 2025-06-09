"""
This module specifies the loss function imports used in PyTorch.
"""

from illia.losses.torch.elbo import ELBOLoss, KLDivergenceLoss

# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
