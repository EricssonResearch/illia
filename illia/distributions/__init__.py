"""
Main module that exposes backend-agnostic distribution layers.
"""

# Standard libraries
from typing import Any

# Own modules
from illia import BackendManager


def GaussianDistribution(*args: Any, **kwargs: Any) -> Any:
    """Backend-agnostic GaussianDistribution."""

    return _load_distribution("GaussianDistribution", *args, **kwargs)


def _load_distribution(distribution_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Helper function to load a backend-agnostic distribution.

    Args:
        distribution_name: Name of the distribution class to retrieve.
        *args: Positional arguments passed to the distribution.
        **kwargs: Keyword arguments passed to the distribution.

    Returns:
        Instantiated backend-specific distribution class.

    Raises:
        ValueError: If the distribution is not available in the selected backend.
    """

    backend_name = BackendManager.get_backend()

    try:
        distribution_class = BackendManager.get_distribution_class(
            backend_name, distribution_name
        )
        return distribution_class(*args, **kwargs)
    except ValueError as e:
        available_backends = BackendManager.get_all_available_backends_for_class(
            distribution_name, "distributions"
        )
        raise ValueError(
            f"{e}. "
            f"Distribution available in the following backends: {available_backends}"
        ) from e
