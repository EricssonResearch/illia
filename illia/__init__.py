"""
This module defines the Illia dynamic backend manager.
"""

import importlib
import os
import warnings
from typing import Any, Optional, Union

from illia.support import (
    AVAILABLE_DNN_BACKENDS,
    AVAILABLE_GNN_BACKENDS,
    BACKEND_CAPABILITIES,
    BACKEND_MODULES,
    VERSION,
)

# Constants
ENV_OS_NAME: str = "ILLIA_BACKEND"
DEFAULT_BACKEND: str = "torch"


class BackendManager:
    """
    Backend manager that dynamically loads modules.
    """

    _backend_modules: dict[str, list[str]] = BACKEND_MODULES
    _backend_capabilities: dict[str, dict[str, set[str]]] = BACKEND_CAPABILITIES
    _loaded_backends: dict[str, dict[str, Any]] = {}
    _active_backend: Optional[str] = None

    @classmethod
    def get_backend_module(
        cls, backend_name: str, module_type: Optional[str] = None
    ) -> Union[Any, dict[str, Any]]:
        """
        Load and return a backend module or all modules of a backend.

        Args:
            backend_name: Backend name.
            module_type: Optional module type to load.

        Returns:
            A specific module or a dictionary with all loaded modules for the backend.

        Raises:
            ValueError: If backend or module is not available.
            ImportError: If the module cannot be imported.
        """

        if cls._active_backend and cls._active_backend != backend_name:
            raise RuntimeError(
                f"Backend already set to '{cls._active_backend}' "
                f"cannot switch to '{backend_name}'. Restart to change."
            )

        if backend_name not in cls._backend_modules:
            raise ImportError(
                f"Backend '{backend_name}' is not available. "
                f"Available backends: {list(cls._backend_modules.keys())}"
            )

        cls._loaded_backends.setdefault(backend_name, {})
        cls._active_backend = backend_name

        if module_type:
            module_key = f"{backend_name}.{module_type}"
            if module_key not in cls._loaded_backends[backend_name]:
                target_module = next(
                    (
                        m
                        for m in cls._backend_modules[backend_name]
                        if f".{module_type}" in m
                    ),
                    None,
                )

                if not target_module:
                    raise ImportError(
                        f"Module '{module_type}' "
                        f"not available for backend '{backend_name}'"
                    )

                try:
                    cls._loaded_backends[backend_name][module_key] = (
                        importlib.import_module(target_module)
                    )
                except ImportError as e:
                    raise ImportError(
                        f"Failed to import module '{module_type}' "
                        f"from backend '{backend_name}'. Error: {e}"
                    ) from e

            return cls._loaded_backends[backend_name][module_key]

        # Load all modules if module_type is not specified
        for module_path in cls._backend_modules[backend_name]:
            module_type_name = module_path.split(".")[-2]
            module_key = f"{backend_name}.{module_type_name}"

            if module_key not in cls._loaded_backends[backend_name]:
                try:
                    cls._loaded_backends[backend_name][module_key] = (
                        importlib.import_module(module_path)
                    )
                except ImportError as e:
                    warnings.warn(f"Could not load module '{module_path}': {e}")
                    continue

        return cls._loaded_backends[backend_name]

    @classmethod
    def is_class_available(
        cls, backend_name: str, class_name: str, module_type: str
    ) -> bool:
        """
        Check if a specific class is available in the backend.
        """

        return (
            backend_name in cls._backend_capabilities
            and module_type in cls._backend_capabilities[backend_name]
            and class_name in cls._backend_capabilities[backend_name][module_type]
        )

    @classmethod
    def get_class(
        cls,
        backend_name: str,
        class_name: str,
        module_type: str,
        module_path: Union[Any, dict[str, Any]],
    ) -> Any:
        """
        Retrieve a class from a specific backend module with availability check.

        Args:
            backend_name: Backend name.
            class_name: Class name to retrieve.
            module_type: Module type.
            module_path: Full module path.

        Returns:
            The class object.

        Raises:
            ValueError: If the class is not available for the backend.
            AttributeError: If the class is not implemented in the loaded module.
        """

        if not cls.is_class_available(backend_name, class_name, module_type):
            available_backends = [
                b
                for b, caps in cls._backend_capabilities.items()
                if class_name in caps.get(module_type, set())
            ]
            msg = f"Class '{class_name}' is not available in backend '{backend_name}'."
            if available_backends:
                msg += f" Available in: {available_backends}"
            raise ImportError(msg)

        return getattr(module_path, class_name)

    @classmethod
    def get_available_classes(cls, backend_name: str, module_type: str) -> set[str]:
        """
        Return all available classes for a given backend and module type.
        """

        return cls._backend_capabilities.get(backend_name, {}).get(module_type, set())

    @classmethod
    def get_all_available_backends_for_class(
        cls, class_name: str, module_type: str
    ) -> list[str]:
        """
        Return all backends that have a specific class available.
        """

        return [
            backend
            for backend, caps in cls._backend_capabilities.items()
            if class_name in caps.get(module_type, set())
        ]

    @classmethod
    def get_backend(
        cls,
    ) -> str:
        """
        Get current backend from environment variable.
        """

        env_backend = os.environ.get(ENV_OS_NAME)

        if cls._active_backend:
            if env_backend and cls._active_backend != env_backend:
                raise RuntimeError(
                    f"Backend already set to '{cls._active_backend}' "
                    f"cannot switch to '{env_backend}'. Restart to change."
                )
            return cls._active_backend

        # If nothing is loaded yet, use environment variable if set, else default
        cls._active_backend = env_backend or DEFAULT_BACKEND
        return cls._active_backend

    @classmethod
    def get_available_backends(
        cls,
    ) -> list[str]:
        """
        Get all available backends.
        """

        return list(AVAILABLE_DNN_BACKENDS | AVAILABLE_GNN_BACKENDS)

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """
        Check if a backend is available for the specified network type.
        """

        available = cls.get_available_backends()
        return backend.lower() in [b.lower() for b in available]

    @classmethod
    def _auto_load_current_backend(
        cls,
    ) -> None:
        """
        Automatically load the current backend on import.
        """

        cls.get_backend_module(cls.get_backend())


# Export methods
__version__ = VERSION
__get_backend__ = BackendManager.get_backend()
__get_available_backends__ = BackendManager.get_available_backends()
is_backend_available = BackendManager.is_backend_available

# Load default backend
BackendManager._auto_load_current_backend()  # pylint: disable=W0212
