"""
Backend-agnostic interface for neural network layers.
"""

# Standard libraries
from typing import Any

# Own modules
from illia import BackendManager


def __getattr__(name: str) -> Any:
    """
    Dynamically import a class from backend.

    Args:
        name: Name of the class to be imported.

    Returns:
        The requested layer/module class.
    """
    backend = BackendManager.get_backend()
    module = BackendManager.get_backend_module(backend, "nn")
    layer_class = BackendManager.get_class(backend, name, "nn", module)

    globals()[name] = layer_class
    return layer_class
