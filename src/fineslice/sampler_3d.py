from typing import Optional, Tuple

import numpy as np

from .cuboid import cuboid, cuboid_edges_axes, cuboid_from_bounds
from .types import Texture3D, check_valid_texture_3d, AffineLike, as_affine, SamplerPoints, SamplerResultND


def _minmax_spoints(points: SamplerPoints):
    return np.column_stack((np.min(points, axis=1), np.max(points, axis=1)))


def sample_3d(
        texture: Texture3D,
        affine: AffineLike,
        out_bounds: Optional[np.ndarray] = None,
        out_resolution_scale: float = 1,
        out_resolution: Optional[Tuple[float, float, float]] = None) -> Optional[SamplerResultND]:
    """
    Args:
        texture
        affine
        out_bounds
        out_resolution_scale
        out_resolution

    Returns:

    """
    check_valid_texture_3d(texture)

    affine = as_affine(affine)
    affine_inv = np.linalg.inv(affine)

    corners_ds = cuboid(texture.shape)
    cube_edges_axes = cuboid_edges_axes()

    # Determine sampling cube
    if out_bounds is not None:
        sampling_cube_bounds_rs = out_bounds
    else:
        # Minmax bounds after transforming to RS
        sampling_cube_bounds_rs = _minmax_spoints(np.dot(affine, corners_ds))

    sampling_cube_rs = cuboid_from_bounds(sampling_cube_bounds_rs)
    sampling_cube_ds = np.dot(affine_inv, sampling_cube_rs).astype(int)  # todo

    # Sampling grid dimensions (data space)
    if out_resolution is None:
        axis_len = np.zeros((3,))
        for v0, v1, va in cube_edges_axes:
            edge_len = np.linalg.norm(sampling_cube_ds[:, v0] - sampling_cube_ds[:, v1])
            if edge_len > axis_len[va]:
                axis_len[va] = edge_len
        axis_len = (axis_len * out_resolution_scale).astype(int)  # todo
    else:
        axis_len = out_resolution

    minmax_data = sampling_cube_bounds_rs
    minmax_data_n = axis_len
    minmax_data_n_total = np.prod(minmax_data_n)
    # print("minmax_data", minmax_data)
    # print("minmax_data_n", minmax_data_n)
    # print("minmax_data_n_total", minmax_data_n_total)

    # print(f'x = from {minmax_data[0, 0]} to {minmax_data[0, 1]} in {minmax_data_n[0]} steps')
    # print(f'y = from {minmax_data[1, 0]} to {minmax_data[1, 1]} in {minmax_data_n[1]} steps')
    # print(f'z = from {minmax_data[2, 0]} to {minmax_data[2, 1]} in {minmax_data_n[2]} steps')

    # Make sampling grid
    sample_grid = np.full((4, minmax_data_n_total), fill_value=1, dtype=np.float64)
    sample_grid[0:3] = np.mgrid[
                       minmax_data[0, 0]:minmax_data[0, 1]:complex(minmax_data_n[0]),
                       minmax_data[1, 0]:minmax_data[1, 1]:complex(minmax_data_n[1]),
                       minmax_data[2, 0]:minmax_data[2, 1]:complex(minmax_data_n[2])
                       ].reshape(3, -1)

    # print("sampling_grid", sample_grid.shape)

    # transform sampling grid (and round for nearest neighbour TODO)
    sample_grid_trans = np.dot(affine_inv, sample_grid).astype(int)

    # clip sampling grid TODO
    for i in range(3):
        sample_grid_trans[i] = sample_grid_trans[i].clip(0, texture.shape[i] - 1)

    x = sample_grid_trans[0:3].reshape((3, minmax_data_n[0], minmax_data_n[1], minmax_data_n[2]))

    rastered = texture[x[0], x[1], x[2]]

    # Todo: convert rounded (for nearest neighbour) estimates
    #  back forth to get more accurate axis_lims
    axis_lims = sampling_cube_bounds_rs[0:3]

    return SamplerResultND(rastered, axis_lims)
