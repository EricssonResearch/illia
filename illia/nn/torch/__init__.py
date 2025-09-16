"""
This module consolidates and exposes layers-related classes
implemented in PyTorch. It imports core base classes and specific
layers implementations for easier access in other modules.
"""

# Own modules
from illia.nn.torch.base import BayesianModule
from illia.nn.torch.conv1d import Conv1d
from illia.nn.torch.conv2d import Conv2d
from illia.nn.torch.embedding import Embedding
from illia.nn.torch.linear import Linear
from illia.nn.torch.lstm import LSTM


# Define all names to be imported
__all__: list[str] = [
    "BayesianModule",
    "Conv1d",
    "Conv2d",
    "Embedding",
    "Linear",
    "LSTM",
]
