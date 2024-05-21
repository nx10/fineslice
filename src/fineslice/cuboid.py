"""Cuboid utility functions."""

from typing import Iterable

import numpy as np
import numpy.typing as npt

from .types import SamplerPoints


def cuboid(shape: Iterable, dtype: npt.DTypeLike = None) -> np.ndarray:
    """Get vertices of a cuboid defined by shape.

    Args:
        shape: Cuboid dimensions (x, y, z)
        dtype: Output dtype

    Returns:
        vertices slicer points array
    """
    x, y, z = (i - 1 for i in shape)
    return np.array(
        [
            [0, 0, 0, 1],  # 0
            [x, 0, 0, 1],  # 1
            [0, y, 0, 1],  # 2
            [0, 0, z, 1],  # 3
            [x, y, 0, 1],  # 4
            [0, y, z, 1],  # 5
            [x, 0, z, 1],  # 6
            [x, y, z, 1],  # 7
        ],
        dtype=dtype if dtype is not None else np.float64,
    ).T


def cuboid_from_bounds(
    bounds: SamplerPoints, dtype: npt.DTypeLike = None
) -> np.ndarray:
    """Get vertices of a cuboid defined by bounds.

    Args:
        bounds: slicer bounds
        dtype: Output dtype

    Returns:
        vertices slicer points array
    """
    x0, y0, z0 = (i - 1 for i in bounds[0:3, 0])
    x1, y1, z1 = (i - 1 for i in bounds[0:3, 1])
    return np.array(
        [
            [x0, y0, z0, 1],  # 0
            [x1, y0, z0, 1],  # 1
            [x0, y1, x0, 1],  # 2
            [x0, y0, z1, 1],  # 3
            [x1, y1, z0, 1],  # 4
            [x0, y1, z1, 1],  # 5
            [x1, y0, z1, 1],  # 6
            [x1, y1, z1, 1],  # 7
        ],
        dtype=dtype if dtype is not None else np.float64,
    ).T


_CUBOID_EDGES = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [1, 6],
        [2, 4],
        [2, 5],
        [3, 5],
        [3, 6],
        [4, 7],
        [5, 7],
        [6, 7],
    ],
    dtype=int,
)
_CUBOID_EDGES.setflags(write=False)  # TODO attempt at making it immutable


def cuboid_edges() -> np.ndarray:
    """Get edges of cube defined by ``cuboid()``.

    Returns:
        ``[[vertex_from, vertex_to]*]``
    """
    return _CUBOID_EDGES


def _make_cuboid_edges_axes() -> np.ndarray:
    """Get edges and their axes of cube defined by ``cuboid()``."""
    cube = cuboid((2, 2, 2), dtype=int)
    return np.array(
        [
            (v0, v1, np.nonzero(cube.T[v0] - cube.T[v1])[0][0])
            for v0, v1 in cuboid_edges()
        ]
    )


_CUBOID_EDGES_AXES = _make_cuboid_edges_axes()
_CUBOID_EDGES_AXES.setflags(write=False)


def cuboid_edges_axes() -> np.ndarray:
    """Get edges and their axes of cube defined by ``cuboid()``.

    Returns:
        ``[[vertex_from, vertex_to, edge_axis]*]``
    """
    return _CUBOID_EDGES_AXES
