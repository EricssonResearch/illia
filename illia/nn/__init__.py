"""
Backend-agnostic interface for neural network layers.
"""

# Own modules
from illia import BackendManager
from illia.support import BACKEND_CAPABILITIES

_backend = BackendManager.get_backend()
_nn_module = BackendManager.get_backend_module(_backend, "nn")

# Generate dynamically __all__ with all 'nn' layers available in any backend
__all__ = sorted(
    {
        class_name
        for backend_caps in BACKEND_CAPABILITIES.values()
        for class_name in backend_caps.get("nn", set())
    }
)

for class_name in __all__:
    # Check if the current backend implements a class with that name
    if hasattr(_nn_module, class_name):
        # Dynamically add that class to the current global namespace
        globals()[class_name] = getattr(_nn_module, class_name)
