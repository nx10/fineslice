from typing import Any

import numpy as np

from .datacube import Datacube
from .filter import Filter
from .types import t_spoint_like, as_slicer_point, t_filters
from .utils import rep_tuple


def _sample_0d(data: Datacube, position: np.ndarray, texture_filter_min: Filter, texture_filter_mag: Filter):
    p = data.transform_inv(position).astype(int)  # todo: interpolation
    return data.image[p[0], p[1], p[2]]


def sample_0d(
        texture: np.ndarray,
        affine: np.ndarray,
        sample: t_spoint_like,
        texture_filter: t_filters = Filter.NEAREST
) -> Any:
    filters = rep_tuple(2, texture_filter)
    return _sample_0d(Datacube(texture, affine), as_slicer_point(sample), filters[0], filters[1])
