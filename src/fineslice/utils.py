from typing import Union, Optional, Iterable, Sized

import numpy as np

from fineslice.types import SamplerPoint


def eye_1d(
        n: int,
        eye_index: int,
        eye_value: Union[int, float, complex, np.ndarray] = 1,
        fill_value: Union[int, float, complex, np.ndarray] = 0,
        dtype: Optional[object] = None
):
    """
    Similar to ``np.full`` but a single index is changed.

    :param n: Vector length.
    :param eye_index: Index.
    :param eye_value: Index value.
    :param fill_value: Fill value.
    :param dtype: The desired data-type for the array The default, None, means ``np.array(f).dtype``.
    :return:
    """
    a = np.full((n,), fill_value=fill_value, dtype=dtype)
    a[eye_index] = eye_value
    return a
