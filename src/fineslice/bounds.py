from typing import Union, Tuple

import numpy as np


def bounds_manual(p1, p2):
    return np.vstack((np.vstack((p1, p2)).T, (0, 0)))


def bounds_where(texture_bool: np.ndarray, affine: np.ndarray, margin=0.):
    wpos = np.where(texture_bool)
    wpos = np.vstack(
        wpos + (np.ones(wpos[0].shape[0]),)
    )

    wpos_trans = np.dot(affine, wpos)

    return np.vstack([
        np.min(wpos_trans, axis=1) - margin,
        np.max(wpos_trans, axis=1) + margin
    ]).T


def bounds_cube(
        size: Union[float, Tuple[float, float, float]],
        offset: Union[float, Tuple[float, float, float]] = 0.) -> np.ndarray:
    if isinstance(size, tuple):
        size_x, size_y, size_z = size
    else:
        size_x = size_y = size_z = size
    if isinstance(offset, tuple):
        offset_x, offset_y, offset_z = offset
    else:
        offset_x = offset_y = offset_z = offset

    return np.array([
        [offset_x-size_x, offset_x+size_x],
        [offset_y-size_y, offset_y+size_y],
        [offset_z-size_z, offset_z+size_z],
        [1., 1.]
    ])
