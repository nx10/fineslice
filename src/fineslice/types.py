from typing import Tuple, List, Union

import numpy as np

from .filter import Filter

t_spoint = np.ndarray
t_spoint_like = Union[Tuple[float, float, float], List[float], t_spoint]

t_spoints = np.ndarray
t_spoints_like = Union[Tuple[t_spoint_like, ...], List[t_spoint_like], t_spoint_like]

t_filters = Union[Filter, Tuple[Filter, Filter]]


def as_slicer_point(point: t_spoint_like) -> t_spoint:
    return np.array([point[0], point[1], point[2], 1], dtype=np.float64)


def as_slicer_points(points: t_spoints_like) -> t_spoints:
    ps = np.array(points, dtype=np.float64).T
    return np.vstack([
        ps, np.ones(ps.shape[1])
    ])
