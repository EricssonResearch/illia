"""
This module specifies the loss function imports used in Tensorflow.
"""

from illia.losses.tf.elbo import ELBOLoss, KLDivergenceLoss

# Define all names to be imported
__all__: list[str] = [
    "ELBOLoss",
    "KLDivergenceLoss",
]
