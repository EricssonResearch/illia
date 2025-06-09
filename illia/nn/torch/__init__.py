"""
This module contains the import statements for NN layers used with PyTorch.
"""

from illia.nn.torch.base import BayesianModule
from illia.nn.torch.conv1d import Conv1D
from illia.nn.torch.conv2d import Conv2D
from illia.nn.torch.embedding import Embedding
from illia.nn.torch.linear import Linear

# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Conv1D",
    "Conv2D",
    "Embedding",
    "Linear",
]
