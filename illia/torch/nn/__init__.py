"""
This module defines the imports for illia.torch.nn.
"""

# Own modules
from illia.torch.nn.base import BayesianModule
from illia.torch.nn.linear import Linear
from illia.torch.nn.embedding import Embedding
from illia.torch.nn.conv import Conv2d

# Define all names to be imported
__all__: list[str] = ["BayesianModule", "Linear", "Embedding", "Conv2d"]
