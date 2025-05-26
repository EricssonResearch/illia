"""
Backend-agnostic interface for loss functions.
"""

# Own modules
from illia import BackendManager
from illia.support import BACKEND_CAPABILITIES

# Obtain parameters for losses
_module_name = "losses"
_backend = BackendManager.get_backend()
_module_path = BackendManager.get_backend_module(_backend, _module_name)

# Generate dynamically __all__ with all 'losses' available in any backend
__all__ = sorted(
    {class_name for class_name in BACKEND_CAPABILITIES[_backend][_module_name]}
)

# Obtain the library to import
def __getattr__(name: str):
    globals()[name] = BackendManager.get_class(
        backend_name=_backend,
        class_name=name,
        module_name=_module_name,
        module_path=_module_path,
    )
