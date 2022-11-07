from typing import Tuple, Union, Iterable

import numpy as np

from .filter import Filter

# Affine matrix

Affine = np.ndarray
AffineLike = np.ndarray | Iterable


def as_affine(affine_like: AffineLike, dtype=np.float64) -> Affine:
    a = np.array(affine_like, dtype=dtype)
    if a.shape == (4, 4) and np.allclose(a[3, :], np.array([0, 0, 0, 1], dtype=dtype)):
        return a
    raise Exception('Affine matrix does not have the correct format')


# Texture3D

Texture3D = np.ndarray
Texture3DLike = np.ndarray | Iterable


def as_texture_3d(texture_3d_like: Texture3DLike) -> Texture3D:
    a = np.array(texture_3d_like)
    if a.ndim == 3:
        return a
    raise Exception('Texture does not have the correct format')


# Sampler point

SamplerPoint = np.ndarray
SamplerPointLike = np.ndarray | Iterable


def as_sampler_point(sampler_point_like: SamplerPointLike, dtype=np.float64) -> Texture3D:
    try:
        return np.array([sampler_point_like[0], sampler_point_like[1], sampler_point_like[2], 1], dtype=dtype)
    except Exception as exc:
        raise Exception('Sampler point does not have the correct format') from exc


# Sampler points

SamplerPoints = np.ndarray
SamplerPointsLike = np.ndarray | Iterable


def as_sampler_points(sampler_points_like: SamplerPointsLike) -> SamplerPointsLike:
    ps = np.array(sampler_points_like, dtype=np.float64).T
    if ps.shape[0] != 3:
        raise Exception('Sampler points does not have the correct format')
    return np.vstack([
        ps, np.ones(ps.shape[1])
    ])


# Filters

Filters = Union[Filter, Tuple[Filter, Filter]]
