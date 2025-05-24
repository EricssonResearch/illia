"""
This module defines the imports for illia.losses.tf.
"""

# Own modules
from illia.losses.tf.elbo import ELBOLoss
from illia.losses.tf.kl import KLDivergenceLoss

# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
