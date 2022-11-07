from typing import Any

import numpy as np

from .datacube import Datacube
from .filter import Filter
from .types import as_affine, AffineLike, as_texture_3d, Texture3DLike, SamplerPointLike, as_sampler_point, \
    SamplerPoint, Filters
from .utils import rep_tuple


def _sample_0d(
        data: Datacube,
        position: SamplerPoint,
        sample_default: Any,
        texture_filter_min: Filter,
        texture_filter_mag: Filter
):
    p = data.transform_inv(position).astype(int)  # todo: interpolation
    if np.any(p < 0) or np.any(p[:3] >= data.image.shape):
        return sample_default
    return data.image[p[0], p[1], p[2]]


def sample_0d(
        texture: Texture3DLike,
        affine: AffineLike,
        sample: SamplerPointLike,
        sample_default: Any = None,
        texture_filter: Filters = Filter.NEAREST
) -> Any:
    """
    Sample a single point from a texture.

    :param texture: Image texture.
    :param affine: Affine matrix.
    :param sample: Point to sample. E.g. (1, 2, 3).
    :param sample_default: Default value that will be returned when sampling outside of texture bounds.
    :param texture_filter: Filters for interpolation.
    :return: Sampled value (type equals texture array type).
    """
    filters = rep_tuple(2, texture_filter)
    return _sample_0d(
        data=Datacube(
            as_texture_3d(texture),
            as_affine(affine)),
        position=as_sampler_point(sample),
        sample_default=sample_default,
        texture_filter_min=filters[0],
        texture_filter_mag=filters[1]
    )
