"""
Backend-agnostic interface for loss functions.
"""

# Own modules
from illia import BackendManager
from illia.support import BACKEND_CAPABILITIES

# Obtain parameters for losses
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

# Check if the current backend implements a class with that name
# and dynamically add that class to the current global namespace
for class_name in __all__:
    if hasattr(_loss_module, class_name):
        globals()[class_name] = getattr(_loss_module, class_name)
