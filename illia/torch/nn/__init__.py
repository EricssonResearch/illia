"""
This module defines the imports for illia.torch.nn.
"""

# Own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.nn.linear import Linear
from illia.torch.nn.embedding import Embedding
from illia.torch.nn.conv1d import Conv1d
from illia.torch.nn.conv2d import Conv2d

# Define all names to be imported
__all__: list[str] = ["BayesianModule", "Linear", "Embedding", "Conv1d", "Conv2d"]
