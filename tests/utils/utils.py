# Libraries
from typing import Any

import numpy as np


def compare_tensors(
    a: Any, b: Any, rtol: float = 1e-1, atol: float = 1e-1, name: str = ""
):
    """
    Compares two tensors for element-wise equality within a specified tolerance.

    Args:
        a: The first tensor to compare.
        b: The second tensor to compare.
        rtol (float): The relative tolerance for equality. Default is 1e-1.
        atol (float): The absolute tolerance for equality. Default is 1e-1.
        name (str): A name to identify the tensors in the output message. Default is an empty string.

    Returns:
        bool: True if all elements in the tensors are within the specified tolerance, False otherwise.
    """

    are_close = np.allclose(a, b, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(a - b))
    print(f"Max absolute difference for {name}: {max_diff}")
    return are_close