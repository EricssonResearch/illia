"""
This module consolidates and exposes losses-related classes
implemented in JAX. It imports core base classes and specific
losses implementations for easier access in other modules.
"""


# Own modules
from illia.losses.jax.elbo import ELBOLoss
from illia.losses.jax.kl import KLDivergenceLoss


# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
