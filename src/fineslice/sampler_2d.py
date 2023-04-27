from typing import Tuple, Optional

import numpy as np

from .cuboid import cuboid, cuboid_edges
from .intersect import intersect_polygon_plane
from .types import SamplerResultND, Texture3D, AffineLike, as_affine, check_valid_texture_3d, SamplerPointLike, \
    sampler_point_3d
from .utils import eye_1d


def sample_2d(
        texture: Texture3D,
        affine: AffineLike,
        out_position: SamplerPointLike,
        out_axis: int,
        out_bounds: Optional[np.ndarray] = None,
        out_resolution_scale: float = 1,
        out_resolution: Optional[Tuple[float, float]] = None
) -> Optional[SamplerResultND]:
    """
    Sample 2d.

    :param texture:
    :param affine:
    :param out_axis:
    :param out_position:
    :param out_bounds:
    :param out_resolution:
    :return:
    """
    check_valid_texture_3d(texture)
    out_position = sampler_point_3d(out_position)

    affine = as_affine(affine)
    affine_inv = np.linalg.inv(affine)

    corners = cuboid(texture.shape)
    edges = cuboid_edges()
    corners_trans = np.dot(affine, corners)

    out_axis_offset = out_position[out_axis]

    inters = intersect_polygon_plane(
        corners_trans, edges,
        plane_origin=out_position[:3],
        plane_normal=eye_1d(3, out_axis, dtype=np.float64))

    # data cube does not intersect plane
    if inters is None or inters.shape[1] < 3:
        return None

    # select variable dimensions
    var_dims = eye_1d(4, 3, False, True, dtype=bool)
    var_dims[out_axis] = False

    var_inters = inters[var_dims]
    x_min, y_min = np.min(var_inters, axis=1)
    x_max, y_max = np.max(var_inters, axis=1)

    # constrain to bounds
    if out_bounds is not None:
        var_bounds = out_bounds[var_dims]
        x_min_bounds, y_min_bounds = np.min(var_bounds, axis=1)
        x_max_bounds, y_max_bounds = np.max(var_bounds, axis=1)

        x_min = max(x_min, x_min_bounds)
        y_min = max(y_min, y_min_bounds)
        x_max = min(x_max, x_max_bounds)
        y_max = min(y_max, y_max_bounds)

        # todo dont need to do stuff above
        x_min, y_min = np.min(var_bounds, axis=1)
        x_max, y_max = np.max(var_bounds, axis=1)

    # sampling rectangle
    rect = np.full((4, 4), fill_value=1, dtype=np.float64)
    rect[var_dims] = np.array([
        [x_min, x_max, x_max, x_min],
        [y_min, y_min, y_max, y_max]
    ])
    rect[out_axis] = out_axis_offset  # equals: inters[sample_axis, 0]

    # find minimum needed resolution (by measuring rectangle sides in data-space)
    if out_resolution is None:
        rect_trans = np.dot(affine_inv, rect)
        w1 = np.linalg.norm(rect_trans[:, 1] - rect_trans[:, 0])
        w2 = np.linalg.norm(rect_trans[:, 3] - rect_trans[:, 2])
        h1 = np.linalg.norm(rect_trans[:, 2] - rect_trans[:, 1])
        h2 = np.linalg.norm(rect_trans[:, 0] - rect_trans[:, 3])

        w = max(w1, w2)
        h = max(h1, h2)
        wn = int(np.ceil(w * out_resolution_scale)) + 1
        hn = int(np.ceil(h * out_resolution_scale)) + 1
    else:
        hn, wn = out_resolution

    # create sampling grid
    sample_grid = np.full((4, wn * hn), fill_value=1, dtype=np.float64)
    sample_grid[var_dims] = np.mgrid[
                            x_min:x_max:complex(wn),
                            y_min:y_max:complex(hn)
                            ].reshape(2, -1)
    sample_grid[out_axis] = out_axis_offset  # equals: inters[axis, 0]

    # transform sampling grid (and round for nearest neighbour TODO)
    sample_grid_trans = np.dot(affine_inv, sample_grid).astype(int)

    # clip sampling grid TODO
    for i in range(3):
        sample_grid_trans[i] = sample_grid_trans[i].clip(0, texture.shape[i] - 1)

    # raster 2D image
    x = sample_grid_trans[0:3].reshape((3, wn, hn))
    rastered = texture[x[0], x[1], x[2]]

    # select axis labels
    axis_indices = np.arange(3)[var_dims[0:3]]

    # Todo: convert rounded (for nearest neighbour) estimates
    #  back forth to get more accurate axis_lims
    axis_lims = rect[var_dims][:, (True, False, True, False)]

    return SamplerResultND(rastered, axis_lims)
