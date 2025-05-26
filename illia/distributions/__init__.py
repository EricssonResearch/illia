"""
Backend-agnostic interface for probability distributions.
"""

# Own modules
from illia import BackendManager
from illia.support import BACKEND_CAPABILITIES

# Obtain parameters for distributions
_backend = BackendManager.get_backend()
_dist_module = BackendManager.get_backend_module(_backend, "distributions")

# Generate dynamically __all__ with all 'distributions' available in any backend
__all__ = sorted(
    {
        class_name
        for backend_caps in BACKEND_CAPABILITIES.values()
        for class_name in backend_caps.get("distributions", set())
    }
)


# Check if the current backend implements a class with that name
# and dynamically add that class to the current global namespace
for class_name in __all__:
    if hasattr(_dist_module, class_name):
        globals()[class_name] = getattr(_dist_module, class_name)
