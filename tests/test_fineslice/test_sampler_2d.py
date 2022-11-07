import numpy as np

import fineslice as fine


def test_slicer_2d():
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=np.float64), (3, 3, 3))
    affine = fine.affine_identity()

    rastered_0_0, _, _ = fine.sample_2d(texture, affine, sample_axis=0, sample_axis_offset=0)
    rastered_0_1, axis_lims_0_1, _ = fine.sample_2d(texture, affine, sample_axis=0, sample_axis_offset=1)
    rastered_1_0, _, _ = fine.sample_2d(texture, affine, sample_axis=1, sample_axis_offset=0)
    rastered_2_0, _, _ = fine.sample_2d(texture, affine, sample_axis=2, sample_axis_offset=0)

    assert np.allclose(rastered_0_0, np.array(
        [[0., 1., 2.],
         [3., 4., 5.],
         [6., 7., 8.]]
    ))
    assert np.allclose(axis_lims_0_1, np.array(
        [[0, 2],
         [0, 2]]
    ))

    assert np.allclose(rastered_0_1, np.array(
        [[9., 10., 11., ],
         [12., 13., 14.],
         [15., 16., 17.]]
    ))
    assert np.allclose(rastered_1_0, np.array(
        [[0., 1., 2.],
         [9., 10., 11.],
         [18., 19., 20.]]
    ))
    assert np.allclose(rastered_2_0, np.array(
        [[0., 3., 6.],
         [9., 12., 15.],
         [18., 21., 24.]]
    ))