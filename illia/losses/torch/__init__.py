"""
This module specifies the loss function imports used in PyTorch.
"""

# Own modules
from illia.losses.torch.elbo import ELBOLoss, KLDivergenceLoss

# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
