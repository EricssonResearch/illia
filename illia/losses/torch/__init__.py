"""
This module consolidates and exposes losses-related classes
implemented in PyTorch. It imports core base classes and specific
losses implementations for easier access in other modules.
"""

# Own modules
from illia.losses.torch.elbo import ELBOLoss
from illia.losses.torch.kl import KLDivergenceLoss


# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
