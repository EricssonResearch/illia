"""
Backend-agnostic interface for loss functions.
"""

# Own modules
from illia import BackendManager
from illia.support import BACKEND_CAPABILITIES

_backend = BackendManager.get_backend()
_loss_module = BackendManager.get_backend_module(_backend, "losses")


# Generate dynamically __all__ with all 'losses' available in any backend
__all__ = sorted(
    {
        class_name
        for backend_caps in BACKEND_CAPABILITIES.values()
        for class_name in backend_caps.get("losses", set())
    }
)

for class_name in __all__:
    # Check if the current backend implements a class with that name
    if hasattr(_loss_module, class_name):
        # Dynamically add that class to the current global namespace
        globals()[class_name] = getattr(_loss_module, class_name)
