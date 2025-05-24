"""
This module defines the imports for illia.nn.torch.
"""

# Own modules
from illia.nn.torch.conv1d import Conv1D
from illia.nn.torch.conv2d import Conv2D
from illia.nn.torch.embedding import Embedding
from illia.nn.torch.linear import Linear

# Define all names to be imported
__all__: list[str] = [
    "Conv1D",
    "Conv2D",
    "Embedding",
    "Linear",
]
