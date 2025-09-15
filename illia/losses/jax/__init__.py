"""
This module specifies the loss function imports used in Jax.
"""

# Own modules
from illia.losses.jax.elbo import ELBOLoss
from illia.losses.jax.kl import KLDivergenceLoss


# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
