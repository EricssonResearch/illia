"""
Main module that exposes backend-agnostic loss layers.
"""

# Standard libraries
from typing import Any

# Own modules
from illia import BackendManager


def KLDivergenceLoss(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic Kullback–Leibler Divergence loss."""

    return _load_loss("KLDivergenceLoss", *args, **kwargs)


def ELBOLoss(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic Evidence Lower Bound (ELBO) loss."""

    return _load_loss("ELBOLoss", *args, **kwargs)


def _load_loss(loss_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Helper function to load a backend-agnostic loss class.

    Args:
        loss_name: Name of the loss class to retrieve.
        *args: Positional arguments passed to the loss class.
        **kwargs: Keyword arguments passed to the loss class.

    Returns:
        Instantiated backend-specific loss class.

    Raises:
        ValueError: If the loss is not available in the selected backend.
    """

    backend_name = BackendManager.get_backend()
    try:
        loss_class = BackendManager.get_loss_class(backend_name, loss_name)
        return loss_class(*args, **kwargs)
    except ValueError as e:
        available_backends = BackendManager.get_all_available_backends_for_class(
            loss_name, "losses"
        )
        raise ValueError(
            f"{e}. Loss available in the following backends: {available_backends}"
        ) from e
