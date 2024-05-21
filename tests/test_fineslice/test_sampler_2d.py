"""Test sampler_2d.py."""

import numpy as np

import fineslice as fine


def test_slicer_2d() -> None:
    """Test 2D slicing."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=np.float64), (3, 3, 3))
    affine = fine.affine_identity()

    rastered_0_0, _ = fine.sample_2d(texture, affine, out_axis=0, out_position=0)
    rastered_0_1, axis_lims_0_1 = fine.sample_2d(
        texture, affine, out_axis=0, out_position=1
    )
    rastered_1_0, _ = fine.sample_2d(texture, affine, out_axis=1, out_position=0)
    rastered_2_0, _ = fine.sample_2d(texture, affine, out_axis=2, out_position=0)

    assert np.allclose(
        rastered_0_0, np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    )
    assert np.allclose(axis_lims_0_1, np.array([[0, 2], [0, 2]]))

    assert np.allclose(
        rastered_0_1,
        np.array(
            [
                [
                    9.0,
                    10.0,
                    11.0,
                ],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
            ]
        ),
    )
    assert np.allclose(
        rastered_1_0, np.array([[0.0, 1.0, 2.0], [9.0, 10.0, 11.0], [18.0, 19.0, 20.0]])
    )
    assert np.allclose(
        rastered_2_0, np.array([[0.0, 3.0, 6.0], [9.0, 12.0, 15.0], [18.0, 21.0, 24.0]])
    )
