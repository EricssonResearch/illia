"""
Backend-agnostic interface for neural network layers.
"""

# Standard libraries
from typing import Any
from functools import partial

# Own modules
from illia import BackendManager
from illia.nonparametric import NONPARAMETRIC_LAYER_MAP, LAYER_CATEGORY_MAP


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

    # Direct O(1) lookup for category
    category = LAYER_CATEGORY_MAP.get(name)
    path = (
        NONPARAMETRIC_LAYER_MAP.get(backend, {}).get(category, {}).get(name)
        if category
        else None
    )

    # Check if this is a non-parametric layer (redirect to native backend)
    if path is not None:
        module_name, class_name = path.rsplit(".", 1)
        layer_class = BackendManager.import_external_class(module_name, class_name)

        # HACK: Special handling for TensorFlow Activation layers
        if backend == "tf" and category == "activation" and class_name == "Activation":
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
