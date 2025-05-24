"""
This module defines the imports for illia.nn.tf.
"""

# Own modules
from illia.nn.tf.conv1d import Conv1D
from illia.nn.tf.conv2d import Conv2D
from illia.nn.tf.embedding import Embedding
from illia.nn.tf.linear import Linear

# Define all names to be imported
__all__: list[str] = [
    "Conv1D",
    "Conv2D",
    "Embedding",
    "Linear",
]
