from typing import Any, Optional

import numpy as np

from .types import SamplerResult0D, as_affine, AffineLike, SamplerPointLike, sampler_point_3d, Texture3D, \
    check_valid_texture_3d


def sample_0d(
        texture: Texture3D,
        affine: AffineLike,
        out_position: SamplerPointLike,
        out_default: Any = None
) -> Optional[SamplerResult0D]:
    """Sample a single point from a texture.

    Args:
        texture: Image texture.
        affine: Affine matrix.
        out_position: Point to sample. E.g. (1, 2, 3).
        out_default: Default value that will be returned when sampling
            outside of texture bounds.

    Returns:
        Sampled value (type equals texture array type).
    """
    check_valid_texture_3d(texture)

    affine = as_affine(affine)
    affine_inv = np.linalg.inv(affine)
    out_position = sampler_point_3d(out_position)

    p = np.dot(affine_inv, out_position).astype(int)
    if np.any(p < 0) or np.any(p[:3] >= texture.shape):
        return out_default
    print(p)
    return texture[p[0], p[1], p[2]]
