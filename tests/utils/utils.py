# Libraries
from typing import Any

import numpy as np  # type: ignore


def compare_tensors(
    a: Any, b: Any, rtol: float = 1e-1, atol: float = 1e-1, name: str = ""
):
    """
    Compares two tensors for element-wise equality within a specified
    tolerance.

    Args:
        a: The first tensor to compare.
        b: The second tensor to compare.
        rtol: The relative tolerance for equality.
        atol: The absolute tolerance for equality.
        name: A name to identify the tensors in the output message.

    Returns:
        bool: True if all elements in the tensors are within the
            specified tolerance, False otherwise.
    """

    are_close = np.allclose(a, b, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(a - b))
    print(f"Max absolute difference for {name}: {max_diff}")
    return are_close
