"""
This module specifies the loss function imports used in Jax.
"""

# Own modules
from illia.losses.jax.elbo import ELBOLoss, KLDivergenceLoss


# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
