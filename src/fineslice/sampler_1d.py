from typing import Any

import numpy as np

from .debug_utils import plot_poly
from .intersect import intersect_line_plane
from .types import as_affine, AffineLike, Texture3D, check_valid_texture_3d, SamplerPointLike, sampler_point_3d, \
    sampler_point_1d, as_sampler_points


def sample_1d(
        texture: Texture3D,
        affine: AffineLike,
        out_position: SamplerPointLike,
        out_axis: int,
        # out_bounds: Optional[Iterable] = None,
        out_resolution: int = None
) -> Any:
    """


    :param texture:
    :param affine:
    :param out_position:
    :param out_axis:
    :param out_bounds:
    :param out_resolution:
    :return:
    """
    check_valid_texture_3d(texture)

    out_position = sampler_point_3d(out_position)
    out_normal = sampler_point_1d(1, axis=out_axis)

    affine = as_affine(affine)
    affine_inv = np.linalg.inv(affine)

    vecs = np.vstack([
        np.vstack((
            np.zeros(3, dtype=int),  # Origin
            texture.shape,  # Data limit
            np.eye(3, dtype=int)  # Axis directions

        )).T, np.ones(5)
    ])

    vecs_trans = np.dot(affine, vecs)
    vecs_trans_origin = vecs_trans[:, 0, np.newaxis]
    vecs_origin = vecs[:, 0, np.newaxis]
    vecs_datalim = vecs[:, 1, np.newaxis]

    unit_vecs = vecs_trans[:, 2:] - vecs_trans_origin  # Absolute directions (from new origin)

    intersects = []
    for plane_origin in vecs_trans[:, :2].T:
        for plane_normal in unit_vecs.T:
            p = intersect_line_plane(out_position[:3], out_normal[:3], plane_origin[:3], plane_normal[:3])
            if p is not None:
                intersects.append(p)
    intersects = as_sampler_points(intersects)
    intersects_trans = np.dot(affine_inv, intersects)
    intersects_in_bounds = np.all(
        ((vecs_origin - 1e-6) < intersects_trans) &
        (intersects_trans < (vecs_datalim + 1e-6)),
        axis=0
    )

    intersects_cube = intersects[:, intersects_in_bounds]
    intersects_cube_trans = intersects_trans[:, intersects_in_bounds]

    if intersects_cube.shape[0] < 2:  # No intersection
        return None

    if intersects_cube.shape[0] > 2:
        # There might be more than 2 intersections
        # (when line directly cuts through edge/corner).
        # Select first and furthest from first:
        diff_idx = [
            0,
            np.argmax(np.sum(intersects_cube[:, 1:] - intersects_cube[:, 0, np.newaxis], axis=0)) + 1
        ]
        intersects_cube = intersects_cube[:, diff_idx]
        intersects_cube_trans = intersects_cube_trans[:, diff_idx]

    line_origin = intersects_cube_trans[:, 0]
    line_target = intersects_cube_trans[:, 1]
    line_dir = line_target - line_origin

    # Find resolution in data space

    mag = np.linalg.norm(line_dir)
    res = int(np.ceil(mag)) + 1

    lmin = np.min(intersects_cube[out_axis])
    lmax = np.max(intersects_cube[out_axis])

    sample_grid = np.repeat(out_position[:,np.newaxis], repeats=res, axis=1)
    sample_grid[out_axis] = np.mgrid[lmin:lmax:complex(res)]

    # transform sampling grid (and round for nearest neighbour TODO)
    sample_grid_trans = np.dot(affine_inv, sample_grid).astype(int)

    # clip sampling grid TODO
    for i in range(3):
        sample_grid_trans[i] = sample_grid_trans[i].clip(0, texture.shape[i] - 1)

    x = sample_grid_trans
    rastered = texture[x[0], x[1], x[2]]

    return rastered, (lmin, lmax)
    """
    from matplotlib import pyplot as plt
    # mpl.use('Qt5Agg')
    plot_poly(
        np.hstack((
            vecs_trans,
            out_position[:, np.newaxis], (out_position + out_normal)[:, np.newaxis],
            intersects_cube
        )),
        edges=[[0, 2], [0, 3], [0, 4], [7, 8]]
    )
    plt.show()"""

    # Find line vector
