# Standard libraries
import os
import importlib
import warnings
from typing import Any, Optional, Union

# Own modules
from illia.support import BACKEND_MODULES, BACKEND_CAPABILITIES, VERSION

# Globals
ENV_OS_NAME = "ILLIA_BACKEND"
DEFAULT_BACKEND = "torch"
AVAILABLES_DNN_BACKENDS = ["jax", "tf", "torch"]
AVAILABLES_GNN_BACKENDS = ["pyg"]


class BackendManager:
    """Backend manager that dynamically loads modules."""

    _backend_modules: dict[str, list[str]] = BACKEND_MODULES
    _backend_capabilities: dict[str, dict[str, set[str]]] = BACKEND_CAPABILITIES
    _loaded_backends: dict[str, dict[str, Any]] = {}

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

        if backend_name not in cls._backend_modules:
            raise ValueError(
                f"Backend '{backend_name}' is not available. "
                f"Available backends: {list(cls._backend_modules.keys())}"
            )

        cls._loaded_backends.setdefault(backend_name, {})

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
                    raise ValueError(
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
        """Check if a specific class is available in the backend."""

        return class_name in cls._backend_capabilities.get(backend_name, {}).get(
            module_type, set()
        )

    @classmethod
    def get_class(cls, backend_name: str, class_name: str, module_type: str) -> Any:
        """
        Retrieve a class from a specific backend module with availability check.

        Args:
            backend_name: Backend name.
            class_name: Class name to retrieve.
            module_type: Module type.

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
            raise ValueError(msg)

        backend_module = cls.get_backend_module(backend_name, module_type)

        if not hasattr(backend_module, class_name):
            raise AttributeError(
                f"Class '{class_name}' is not implemented in module '{module_type}' "
                f"of backend '{backend_name}'"
            )

        return getattr(backend_module, class_name)

    @classmethod
    def get_available_classes(cls, backend_name: str, module_type: str) -> set[str]:
        """Return all available classes for a given backend and module type."""

        return cls._backend_capabilities.get(backend_name, {}).get(module_type, set())

    @classmethod
    def get_all_available_backends_for_class(
        cls, class_name: str, module_type: str
    ) -> list[str]:
        """Return all backends that have a specific class available."""

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

        return os.environ.get(ENV_OS_NAME, DEFAULT_BACKEND)

    @classmethod
    def set_backend(cls, backend: str) -> None:
        """
        set the backend to be used by environment variable.
        """

        os.environ[ENV_OS_NAME] = backend

    @classmethod
    def get_available_backends(cls, network_type: str = "all") -> list[str]:
        """
        Get available backends for specific network type.

        Args:
            network_type: "dnn", "gnn", or "all"
        """

        if network_type.lower() == "dnn":
            return AVAILABLES_DNN_BACKENDS
        if network_type.lower() == "gnn":
            return AVAILABLES_GNN_BACKENDS
        return AVAILABLES_DNN_BACKENDS + AVAILABLES_GNN_BACKENDS

    @classmethod
    def is_backend_available(cls, backend: str, network_type: str = "all") -> bool:
        """
        Check if a backend is available for the specified network type.
        """

        available = cls.get_available_backends(network_type)
        return backend.lower() in [b.lower() for b in available]

    @classmethod
    def version(
        cls,
    ) -> str:
        """
        Check Illia version.
        """

        return VERSION


# Export methods
version = BackendManager.version
get_backend = BackendManager.get_backend
set_backend = BackendManager.set_backend
get_available_backends = BackendManager.get_available_backends
is_backend_available = BackendManager.is_backend_available
