"""Test sampler_1d.py."""

import numpy as np

import fineslice as fine


def test_slicer_1d() -> None:
    """Test 1D slicing."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=np.float64), (3, 3, 3))
    affine = fine.affine_rotate_degrees(fine.affine_identity(), x=0, y=0)

    rastered, res = fine.sample_1d(texture, affine, out_position=1.5, out_axis=0)  # type: ignore

    # TODO
