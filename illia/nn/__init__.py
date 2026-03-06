"""
Backend-agnostic interface for neural network layers.
"""

# Standard libraries
from functools import partial
from typing import Any

# Own modules
from illia import BackendManager
from illia.layers import _BAYESIAN_LAYERS, LAYER_CATEGORY_MAP, NONPARAMETRIC_LAYER_MAP


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

    #  get Bayesian layer from illia implementation, or failure.
    if name in _BAYESIAN_LAYERS:
        module = BackendManager.get_backend_module(backend, module_type)
        layer_class = BackendManager.get_class(
            backend_name=backend,
            class_name=name,
            module_type=module_type,
            module_path=module,
        )

    elif (category := LAYER_CATEGORY_MAP.get(name, None)) is not None:
        # Otherwise, this is a non-parametric layer (redirect to native backend)
        module_path = (
            NONPARAMETRIC_LAYER_MAP.get(backend, {}).get(category, {}).get(name)
        )
        if module_path:
            module_name, class_name = module_path.rsplit(".", 1)
            layer_class = BackendManager.import_native_backend_class(
                module_name, class_name
            )

            # HACK: Special handling for TensorFlow Activation layers
            if (
                backend == "tf"
                and category == "activation"
                and class_name == "Activation"
            ):
                layer_class = partial(layer_class, name.lower())
        else:
            raise ImportError(
                f"Module '{module_type}', {name} not available for backend '{backend}'."
            )

    else:
        raise ImportError(
            f"Module '{module_type}', {name} not available for backend '{backend}'."
        )

    globals()[name] = layer_class
    return layer_class
