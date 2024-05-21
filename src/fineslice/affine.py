"""Affine transformation functions.

Create, modify, and apply affine transformations.
"""

from typing import Union

import numpy as np

from .types import Affine

NpScalar = Union[int, float, np.ndarray]


def affine_invert(aff: Affine) -> Affine:
    """Invert affine matrix."""
    return np.linalg.inv(aff)


def affine_identity(
    dtype=np.float64,  # noqa: ANN001
) -> Affine:
    """Create identity affine matrix."""
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=dtype
    )


def affine_translate(
    aff: Affine, x: NpScalar = 0, y: NpScalar = 0, z: NpScalar = 0
) -> Affine:
    """Translate affine matrix."""
    affine_translated = np.array(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=aff.dtype
    )
    return aff.dot(affine_translated)


def affine_scale(
    aff: Affine, x: NpScalar = 1, y: NpScalar = 1, z: NpScalar = 1
) -> Affine:
    """Scale affine matrix."""
    affine_scaled = np.array(
        [[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]], dtype=aff.dtype
    )
    return aff.dot(affine_scaled)


def affine_shear(
    aff: Affine,
    xy: NpScalar = 0,
    yx: NpScalar = 0,
    xz: NpScalar = 0,
    zx: NpScalar = 0,
    yz: NpScalar = 0,
    zy: NpScalar = 0,
) -> Affine:
    """Shear affine matrix."""
    affine_scaled = np.array(
        [[1, xy, xz, 0], [yx, 1, yz, 0], [zx, zy, 1, 0], [0, 0, 0, 1]], dtype=aff.dtype
    )
    return aff.dot(affine_scaled)


def affine_rotate(
    aff: Affine, x: NpScalar = 0, y: NpScalar = 0, z: NpScalar = 0
) -> Affine:
    """Rotate affine matrix (in radians)."""
    affine_rx = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(x), np.sin(x), 0],
            [0, -np.sin(x), np.cos(x), 0],
            [0, 0, 0, 1],
        ],
        dtype=aff.dtype,
    )
    affine_ry = np.array(
        [
            [np.cos(y), 0, -np.sin(y), 0],
            [0, 1, 0, 0],
            [np.sin(y), 0, np.cos(y), 0],
            [0, 0, 0, 1],
        ],
        dtype=aff.dtype,
    )
    affine_rz = np.array(
        [
            [np.cos(z), -np.sin(z), 0, 0],
            [np.sin(z), np.cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=aff.dtype,
    )

    return aff.dot(affine_rx).dot(affine_ry).dot(affine_rz)


def affine_rotate_degrees(
    aff: Affine, x: NpScalar = 0, y: NpScalar = 0, z: NpScalar = 0
) -> Affine:
    """Rotate affine matrix in degrees."""
    return affine_rotate(aff, np.radians(x), np.radians(y), np.radians(z))
