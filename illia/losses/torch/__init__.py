"""
This module defines the imports for illia.losses.torch.
"""

# Own modules
from illia.losses.torch.elbo import ELBOLoss
from illia.losses.torch.kl import KLDivergenceLoss

# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
