"""
Backend-agnostic interface for neural network layers.
"""

# Standard libraries
from typing import Any

# Own modules
from illia import BackendManager
from illia.support import NONPARAMETRIC_LAYER_MAP


def __getattr__(name: str) -> Any:
    """
    Dynamically import a class from backend (Bayesian or non-parametric).

    Args:
        name: Name of the class to be imported.

    Returns:
        The requested layer/module class.
    """

    # Obtain parameters for nn
    module_type: str = "nn"
    backend: str = BackendManager.get_backend()

    # Check if this is a non-parametric layer (redirect to native backend)
    if (path := NONPARAMETRIC_LAYER_MAP.get(backend, {}).get(name)) is not None:
        module_name, class_name = path.rsplit(".", 1)
        layer_class = BackendManager.import_external_class(module_name, class_name)
        
        # HACK: Special handling for TensorFlow Activation layers
        if backend == "tf" and class_name == "Activation":
            from functools import partial
            layer_class = partial(layer_class, name.lower())
    else:
        # Otherwise, get Bayesian layer from illia implementation, or failure.
        module = BackendManager.get_backend_module(backend, module_type)
        layer_class = BackendManager.get_class(
            backend_name=backend,
            class_name=name,
            module_type=module_type,
            module_path=module,
        )

    globals()[name] = layer_class
    return layer_class
