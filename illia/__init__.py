"""
This module defines the illia dynamic backend manager.
"""

# Standard libraries
import importlib
import os
import warnings
from functools import lru_cache
from typing import Any, Optional

# Own modules
from illia.support import (
    AVAILABLE_DNN_BACKENDS,
    AVAILABLE_GNN_BACKENDS,
    BACKEND_CAPABILITIES,
    BACKEND_MODULES,
    DEFAULT_BACKEND,
    ENV_OS_NAME,
)


class BackendManager:
    """
    Backend manager that dynamically loads modules.
    """

    # Configuration data
    _backend_modules: dict[str, list[str]] = BACKEND_MODULES
    _backend_capabilities: dict[str, dict[str, set[str]]] = BACKEND_CAPABILITIES

    # Runtime state
    _loaded_backends: dict[str, dict[str, Any]] = {}
    _active_backend: Optional[str] = None
    _module_cache: dict[str, Any] = {}
    _available_backends: Optional[list[str]] = None

    @classmethod
    def get_backend_module(
        cls, backend_name: str, module_type: Optional[str] = None
    ) -> Any | dict[str, Any]:
        """
        Retrieve a backend module or all its modules.

        Args:
            backend_name: Name of the backend to load from.
            module_type: Module type to load. Loads all if omitted.

        Returns:
            The specified module or a dictionary of all modules for the
                backend.

        Raises:
            ValueError: If the backend or module is not found.
            ImportError: If the module import fails.
        """

        # Check if we're trying to switch backends, this isn't allowed
        if cls._active_backend and cls._active_backend != backend_name:
            raise RuntimeError(
                f"Already using '{cls._active_backend}'. "
                f"Can't switch to '{backend_name}'. Restart to change backends."
            )

        # Validate backend exists
        if backend_name not in cls._backend_modules:
            available = list(cls._backend_modules.keys())
            raise ImportError(
                f"Backend '{backend_name}' doesn't exist. "
                f"Available backends: {available}."
            )

        # Initialize storage for this backend
        if backend_name not in cls._loaded_backends:
            cls._loaded_backends[backend_name] = {}

        cls._active_backend = backend_name

        if module_type:
            return cls._load_single_module(backend_name, module_type)

        return cls._load_all_modules(backend_name)

    @classmethod
    def _load_single_module(cls, backend_name: str, module_type: str) -> Any:
        """
        Load a single backend module with caching support.

        Args:
            backend_name: Name of the backend to load from.
            module_type: Type of module to load.

        Returns:
            The loaded module instance.

        Raises:
            ValueError: If the module does not exist for the given backend.
            ImportError: If the module import fails.
        """

        module_key = f"{backend_name}.{module_type}"

        # Check if already loaded
        if module_key in cls._loaded_backends[backend_name]:
            return cls._loaded_backends[backend_name][module_key]

        # Find target module path
        target_module = None
        module_suffix = f".{module_type}"
        for module_path in cls._backend_modules[backend_name]:
            if module_suffix in module_path:
                target_module = module_path
                break

        if not target_module:
            raise ImportError(
                f"Module '{module_type}' not available for backend '{backend_name}'."
            )

        # Load the module with caching
        if target_module in cls._module_cache:
            loaded_module = cls._module_cache[target_module]
        else:
            try:
                loaded_module = importlib.import_module(target_module)
                cls._module_cache[target_module] = loaded_module
            except ImportError as e:
                raise ImportError(
                    f"Failed to import module '{module_type}' "
                    f"from backend '{backend_name}'. Error: {e}."
                ) from e

        cls._loaded_backends[backend_name][module_key] = loaded_module
        return loaded_module

    @classmethod
    def _load_all_modules(cls, backend_name: str) -> dict[str, Any]:
        """
        Load all modules of a backend using optimized batch loading.

        Args:
            backend_name: Name of the backend whose modules should be loaded.

        Returns:
            A dictionary mapping module types to their corresponding
            loaded module instances.

        Raises:
            ValueError: If the backend does not exist or has no modules.
            ImportError: If any module fails to import.
        """

        modules_to_load = []

        # Collect modules that need loading
        for module_path in cls._backend_modules[backend_name]:
            module_type_name = module_path.split(".")[-2]
            module_key = f"{backend_name}.{module_type_name}"

            if module_key not in cls._loaded_backends[backend_name]:
                modules_to_load.append((module_path, module_key, module_type_name))

        # Load modules
        for module_path, module_key, _ in modules_to_load:
            if module_path in cls._module_cache:
                loaded_module = cls._module_cache[module_path]
            else:
                try:
                    loaded_module = importlib.import_module(module_path)
                    cls._module_cache[module_path] = loaded_module
                except ImportError as e:
                    warnings.warn(f"Could not load module '{module_path}': {e}.")
                    continue

            cls._loaded_backends[backend_name][module_key] = loaded_module

        return cls._loaded_backends[backend_name]

    @classmethod
    @lru_cache
    def is_class_available(
        cls, backend_name: str, class_name: str, module_type: str
    ) -> bool:
        """
        Determine if a specific class exists in the backend modules.

        Args:
            backend_name: Name of the backend to search.
            class_name: Name of the class to verify.
            module_type: Module type where the class might be located.

        Returns:
            True if the class is found in the backend's module, else False.

        Raises:
            ValueError: If the backend or module type is not found.
            ImportError: If the module cannot be imported.
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
        module_path: Any | dict[str, Any],
    ) -> Any:
        """
        Retrieve a class from a backend module, checking its availability.

        Args:
            backend_name: Name of the backend for the class.
            class_name: Name of the class to retrieve.
            module_type: Type of module containing the class.
            module_path: Full module path or mapping of module types.

        Returns:
            The requested class object.

        Raises:
            ValueError: If the class is not available for the backend.
            AttributeError: If the class is not implemented in the loaded
                module.
        """

        if not cls.is_class_available(backend_name, class_name, module_type):
            available_backends = cls.get_all_available_backends_for_class(
                class_name, module_type
            )
            msg = f"Class '{class_name}' is not available in backend '{backend_name}'."
            if available_backends:
                msg += f" Available in: {available_backends}."
            raise ImportError(msg)

        return getattr(module_path, class_name)

    @classmethod
    @lru_cache
    def get_available_classes(
        cls, backend_name: str, module_type: str
    ) -> frozenset[str]:
        """
        Get all available classes for a backend's module type with caching.

        Args:
            backend_name: Name of the backend to inspect.
            module_type: Type of module whose classes should be listed.

        Returns:
            A frozenset containing all class names found.

        Raises:
            ValueError: If the backend or module type is not found.
            ImportError: If the module fails to load.
        """

        return frozenset(
            cls._backend_capabilities.get(backend_name, {}).get(module_type, set())
        )

    @classmethod
    @lru_cache
    def get_all_available_backends_for_class(
        cls, class_name: str, module_type: str
    ) -> tuple[str, ...]:
        """
        List all backends providing a given class in a module type.

        Args:
            class_name: Name of the class to search for.
            module_type: Type of module where the class should be located.

        Returns:
            A tuple containing the names of backends that provide the class.

        Raises:
            ImportError: If a backend's module cannot be imported.
        """

        return tuple(
            backend
            for backend, caps in cls._backend_capabilities.items()
            if class_name in caps.get(module_type, set())
        )

    @classmethod
    def get_backend(cls) -> str:
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
    def get_available_backends(cls) -> list[str]:
        """
        Get all available backends.
        """

        if cls._available_backends is None:
            cls._available_backends = list(
                AVAILABLE_DNN_BACKENDS | AVAILABLE_GNN_BACKENDS
            )

        return cls._available_backends.copy()

    @classmethod
    @lru_cache
    def is_backend_available(cls, backend: str) -> bool:
        """
        Verify if a backend is available for a given network type.

        Args:
            backend: Name of the backend to check.

        Returns:
            True if the backend is available, else False.

        Raises:
            ImportError: If there's an issue importing the backend.
        """

        available_backends_lower = {b.lower() for b in cls.get_available_backends()}
        return backend.lower() in available_backends_lower

    @classmethod
    def _auto_load_current_backend(cls) -> None:
        """
        Automatically load the current backend on import.
        """

        cls.get_backend_module(cls.get_backend())


def __getattr__(name: str) -> Any:
    """
    Dynamically retrieve special backend-related attributes.

    Args:
        name: Attribute name to access.

    Returns:
        The corresponding backend manager method when recognized.

    Raises:
        AttributeError: If the attribute name is not recognized.
    """

    if name == "__get_backend__":
        return BackendManager.get_backend()
    if name == "__get_available_backends__":
        return BackendManager.get_available_backends()
    if name == "__version__":
        return importlib.metadata.version("illia")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


is_backend_available = BackendManager.is_backend_available

# Load default backend
BackendManager._auto_load_current_backend()  # pylint: disable=W0212
