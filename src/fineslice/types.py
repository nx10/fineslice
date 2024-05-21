"""Type definitions for fineslice module."""

from typing import Any, Iterable, NamedTuple, Protocol, Tuple, Union

import numpy as np

# Affine matrix

Affine = np.ndarray
"""Affine matrix type."""
AffineLike = Union[np.ndarray, Iterable]
"""Objects convertible to Affine type."""


def as_affine(
    affine_like: AffineLike,
    dtype=np.float64,  # noqa: ANN001
) -> Affine:
    """Converts AffineLike to Affine."""
    a = np.array(affine_like, dtype=dtype)
    if a.shape == (4, 4) and np.allclose(a[3, :], np.array([0, 0, 0, 1], dtype=dtype)):
        return a
    raise Exception("Affine matrix does not have the correct format")


# Texture3D


class Texture3D(Protocol):
    """3D texture type."""

    ndim: int
    shape: Tuple[int, ...]

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401
        """Get item."""
        pass


def check_valid_texture_3d(texture_3d_like: Texture3D) -> None:
    """Check if texture has the correct format."""
    if not texture_3d_like.ndim == 3:
        raise Exception("Texture does not have the correct format")


# Sampler point

SamplerPoint = np.ndarray
"""Sampler point type."""
SamplerPointLike = Union[np.ndarray, Iterable, int, float]
"""Objects convertible to SamplerPoint type."""


def sampler_point_0d(
    dtype=np.float64,  # noqa: ANN001
) -> SamplerPoint:
    """Creates origin sampler point (0,0,0).

    Args:
        dtype: Output dtype.

    Returns:
        Sampler point.
    """
    return np.array([0, 0, 0, 1], dtype=dtype)


def sampler_point_1d(
    value: Any,  # noqa: ANN401
    axis: int,
    dtype=np.float64,  # noqa: ANN001
) -> SamplerPoint:
    """Create sampler point on a 1D line.

    Creates sampler point at (a, b, c) so that one axis
    is set to value and the other exes are 0.

    Args:
        value: Position on the axis.
        axis: Axis.
        dtype: Output dtype.

    Returns:
        Sampler point
    """
    if axis > 2:
        raise Exception("Axis must be < 3.")
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    arr[axis] = value
    return arr


def sampler_point_2d(
    value: Union[np.ndarray, Iterable, int, float],
    zero_axis: int,
    dtype=np.float64,  # noqa: ANN001
) -> SamplerPoint:
    """Creates sampler point on a 2D plane.

    Creates sampler point at (a, b, c) so that one axis is
    set to 0 and the other exes are broadcast from value.

    Args:
        value: Position on the plane.
        zero_axis: Axis that will be set to 0.
        dtype: Output dtype.

    Returns:
        Sampler point.
    """
    if zero_axis > 2:
        raise Exception("Axis must be < 3.")
    values = np.asanyarray(value)
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    var_axes = np.full((4,), True)
    var_axes[zero_axis] = False
    var_axes[3] = False
    arr[var_axes] = values
    return arr


def sampler_point_3d(
    sampler_point_like: SamplerPointLike,
    dtype=np.float64,  # noqa: ANN001
) -> SamplerPoint:
    """Create sampler point from any array or scalar value.

    Args:
        sampler_point_like: Sampler point like.
        dtype: Output dtype.

    Returns:
        Sampler point.
    """
    values: np.ndarray = np.ravel(sampler_point_like)  # type: ignore
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    var_axes = np.full((4,), True)
    var_axes[3] = False
    arr[var_axes] = values[:3]
    return arr


# Sampler points

SamplerPoints = np.ndarray
"""Collection of sampler points."""
SamplerPointsLike = Union[np.ndarray, Iterable]
"""Objects convertible to SamplerPoints type."""


def as_sampler_points(sampler_points_like: SamplerPointsLike) -> SamplerPoints:
    """Converts SamplerPointsLike to SamplerPoints."""
    ps = np.array(sampler_points_like, dtype=np.float64).T
    if ps.shape[0] != 3:
        raise Exception("Sampler points does not have the correct format")
    return np.vstack([ps, np.ones(ps.shape[1])])


# Result

SamplerResult0D = Any
"""Return sampled point values directly."""


class SamplerResultND(NamedTuple):
    """Returned by fineslice.sample_*d(...) methods."""

    texture: np.ndarray
    """Return texture."""

    coordinates: np.ndarray
    """Coordinates of the returned texture. [(min, max), ...]"""
