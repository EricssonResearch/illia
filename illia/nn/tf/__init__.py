"""
This module contains the import statements for NN layers used with Tensorflow.
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
