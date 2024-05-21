"""Utility functions."""

from typing import Any, Union

import numpy as np


def eye_1d(
    n: int,
    eye_index: int,
    eye_value: Union[int, float, complex, np.ndarray] = 1,
    fill_value: Union[int, float, complex, np.ndarray] = 0,
    dtype: Any = None,  # noqa: ANN401
) -> np.ndarray:
    """Similar to ``np.full`` but a single index is changed.

    Args:
        n: Vector length.
        eye_index: Index.
        eye_value: Index value.
        fill_value: Fill value.
        dtype: The desired data-type for the array The default, None,
            means ``np.array(f).dtype``.

    Returns:
        1D array with a single index changed.
    """
    a = np.full((n,), fill_value=fill_value, dtype=dtype)
    a[eye_index] = eye_value
    return a
