"""
This module consolidates and exposes layers-related classes
implemented in PyTorch Geometric. It imports core base classes
and specific layers implementations for easier access in
other modules.
"""

# Own modules
from illia.nn.pyg.cg_conv import CGConv


# Define all names to be imported
__all__: list[str] = [
    "CGConv",
]
