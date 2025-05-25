"""
Main module that exposes backend-agnostic neural network layers.
"""

# Standard libraries
from typing import Any

# Own modules
from illia import BackendManager


def Conv1D(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic 1D convolutional layer."""
    return _load_layer("Conv1D", *args, **kwargs)


def Conv2D(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic 2D convolutional layer."""
    return _load_layer("Conv2D", *args, **kwargs)


def Embedding(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic Embedding layer."""
    return _load_layer("Embedding", *args, **kwargs)


def Linear(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic Linear (Dense) layer."""
    return _load_layer("Linear", *args, **kwargs)


def _load_layer(layer_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Helper function to load a backend-agnostic layer.

    Args:
        layer_name: Name of the layer class to retrieve.
        *args: Positional arguments passed to the layer.
        **kwargs: Keyword arguments passed to the layer.

    Returns:
        Instantiated backend-specific layer class.

    Raises:
        ValueError: If the layer is not available in the selected backend.
    """
    backend_name = BackendManager.get_backend()
    try:
        layer_class = BackendManager.get_layer_class(backend_name, layer_name)
        return layer_class(*args, **kwargs)
    except ValueError as e:
        available_backends = BackendManager.get_all_available_backends_for_class(
            layer_name, "nn"
        )
        raise ValueError(
            f"{e}. "
            f"This layer is available in the following backends: {available_backends}"
        ) from e
