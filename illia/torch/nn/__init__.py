"""
This module defines the imports for illia.torch.nn.
"""

# Own modules
from illia.torch.nn.conv1d import Conv1d
from illia.torch.nn.conv2d import Conv2d
from illia.torch.nn.embedding import Embedding
from illia.torch.nn.linear import Linear

# Define all names to be imported
__all__: list[str] = ["Conv1d", "Conv2d", "Embedding", "Linear"]
