# Libraries
import os

# Available backends for illia
_AVAILABLE_DNN_BACKENDS = ["torch", "tf", "jax"]
_AVAILABLE_GNN_BACKENDS = ["torch_geometric", "dgl", "spektral"]

# Default backend for illia: Torch and Torch Geometric
_DNN_BACKEND = _AVAILABLE_DNN_BACKENDS[0]
_GNN_BACKEND = _AVAILABLE_GNN_BACKENDS[0]

# Set backend based on ILLIA_BACKEND flag, if applicable
if "ILLIA_DNN_BACKEND" in os.environ:
    _backend = os.environ["ILLIA_DNN_BACKEND"]
    if _backend:
        if _backend in _AVAILABLE_DNN_BACKENDS:
            _DNN_BACKEND = _backend
        else:
            raise Exception(
                f"DNN backend not available, the availables backends are {_AVAILABLE_DNN_BACKENDS}."
            )

if "ILLIA_GNN_BACKEND" in os.environ:
    _backend_gnn = os.environ["ILLIA_GNN_BACKEND"]
    if _backend_gnn:
        if _backend_gnn in _AVAILABLE_GNN_BACKENDS:
            _GNN_BACKEND = _backend_gnn
        else:
            raise Exception(
                f"GNN backend not available, the availables backends are {_AVAILABLE_GNN_BACKENDS}."
            )

if _DNN_BACKEND != "tf":
    # https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def backend():
    """
    Publicly accessible method for determining the current backend.

    Returns:
        The name (String) of the DNN backend illia is currently using.
    """

    return _DNN_BACKEND


def gnn_backend():
    """
    Publicly accessible method for determining the current backend.

    Returns:
        The name (String) of the GNN backend illia is currently using.
    """

    return _GNN_BACKEND
