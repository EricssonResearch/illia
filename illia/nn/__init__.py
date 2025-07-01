"""
Backend-agnostic interface for neural network layers.
"""

# Standard libraries
from typing import Any, Union

# Own modules
from illia import BackendManager


# Obtain the library to import
def __getattr__(name: str) -> None:
    """
    Dynamically import a class from backend distributions.

    Args:
        name: Name of the class to be imported.
    """

    # Obtain parameters for nn
    module_type: str = "nn"
    backend: str = BackendManager.get_backend()
    module_path: Union[Any, dict[str, Any]] = BackendManager.get_backend_module(
        backend, module_type
    )

    # Set class to global namespace
    globals()[name] = BackendManager.get_class(
        backend_name=backend,
        class_name=name,
        module_type=module_type,
        module_path=module_path,
    )
