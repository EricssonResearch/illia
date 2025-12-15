"""
This module consolidates and exposes losses-related classes
implemented in Tensorflow. It imports core base classes and specific
losses implementations for easier access in other modules.
"""

# Own modules
from illia.losses.tf.elbo import ELBOLoss
from illia.losses.tf.kl import KLDivergenceLoss


__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
