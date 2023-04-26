from typing import Tuple, Union, Iterable, Protocol

import numpy as np

# Affine matrix

Affine = np.ndarray
AffineLike = Union[np.ndarray, Iterable]


def as_affine(affine_like: AffineLike, dtype=np.float64) -> Affine:
    a = np.array(affine_like, dtype=dtype)
    if a.shape == (4, 4) and np.allclose(a[3, :], np.array([0, 0, 0, 1], dtype=dtype)):
        return a
    raise Exception('Affine matrix does not have the correct format')


# Texture3D

class Texture3D(Protocol):
    ndim: int
    shape: Tuple[int, ...]

    def __getitem__(self, key):
        pass


def check_valid_texture_3d(texture_3d_like: Texture3D) -> None:
    if not texture_3d_like.ndim == 3:
        raise Exception('Texture does not have the correct format')


# Sampler point

SamplerPoint = np.ndarray
SamplerPointLike = Union[np.ndarray, Iterable, int, float]


def sampler_point_0d(dtype=np.float64) -> SamplerPoint:
    """
    Creates origin sampler point (0,0,0).

    :param dtype:
    :return:
    """
    return np.array([0, 0, 0, 1], dtype=dtype)


def sampler_point_1d(value: any, axis: int, dtype=np.float64) -> SamplerPoint:
    """
    Creates sampler point at (a, b, c) so that one axis is set to value and the other exes are 0.

    :param value:
    :param axis:
    :param dtype: Output dtype.
    :return:
    """
    if axis > 2:
        raise Exception('Axis must be < 3.')
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    arr[axis] = value
    return arr


def sampler_point_2d(value: Union[np.ndarray, Iterable, int, float], zero_axis: int,
                     dtype=np.float64) -> SamplerPoint:
    """
    Creates sampler point at (a, b, c) so that one axis is set to 0 and the other exes are broadcast from value.

    :param value:
    :param zero_axis:
    :param dtype: Output dtype.
    :return:
    """
    if zero_axis > 2:
        raise Exception('Axis must be < 3.')
    values = np.asanyarray(value)
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    var_axes = np.full((4,), True)
    var_axes[zero_axis] = False
    var_axes[3] = False
    arr[var_axes] = values
    return arr


def sampler_point_3d(sampler_point_like: SamplerPointLike, dtype=np.float64) -> SamplerPoint:
    """
    Create sampler point from any array or scalar value.

    :param sampler_point_like:
    :param dtype: Output dtype.
    :return:
    """
    values = np.asanyarray(sampler_point_like)
    arr = np.array([0, 0, 0, 1], dtype=dtype)
    var_axes = np.full((4,), True)
    var_axes[3] = False
    arr[var_axes] = values
    return arr


# Sampler points

SamplerPoints = np.ndarray
SamplerPointsLike = Union[np.ndarray, Iterable]


def as_sampler_points(sampler_points_like: SamplerPointsLike) -> SamplerPoints:
    ps = np.array(sampler_points_like, dtype=np.float64).T
    if ps.shape[0] != 3:
        raise Exception('Sampler points does not have the correct format')
    return np.vstack([
        ps, np.ones(ps.shape[1])
    ])
