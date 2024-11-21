# Libraries
from typing import Literal


def get_backend(backend_name: Literal["torch", "tf", "jax"] = "torch") -> int:
    """
    This function returns the backend used by this library. It allows
    torch, tensorflow and jax. It represents them with the integers 0,
    1 and 2 respectively.

    Args:
        backend_name: The name of the backend to use. Defaults to torch.

    Raises:
        ValueError: Invalid backend name.

    Returns:
        the integer that represents the backend.
    """

    if backend_name == "torch":
        return 0
    elif backend_name == "tf":
        return 1
    elif backend_name == "jax":
        return 2
    else:
        raise ValueError("Invalid backend name")


def get_geometric_backend(
    backend_name: Literal["torch_geometric", "dgl", "spektral"] = "torch_geometric"
) -> int:
    """
    This function returns the backend used by this library. It allows
    torch_geometric, dgl and spektral. It represents them with the
    integers 0, 1 and 2 respectively.

    Args:
        backend_name: The name of the backend to use. Defaults to torch.

    Raises:
        ValueError: Invalid backend name.

    Returns:
        the integer that represents the backend.
    """

    if backend_name == "torch_geometric":
        return 0
    elif backend_name == "dgl":
        return 1
    elif backend_name == "spektral":
        return 2
    else:
        raise ValueError("Invalid backend name")
