"""
This module defines the imports for illia.tf.nn.
"""

# Own modules
from illia.tf.nn.conv1d import Conv1d
from illia.tf.nn.conv2d import Conv2d
from illia.tf.nn.embedding import Embedding
from illia.tf.nn.linear import Linear
from illia.tf.nn.losses import KLDivergenceLoss, ELBOLoss

# Define all names to be imported
__all__: list[str] = [
    "Conv1d",
    "Conv2d",
    "Embedding",
    "Linear",
    "KLDivergenceLoss",
    "ELBOLoss",
]
