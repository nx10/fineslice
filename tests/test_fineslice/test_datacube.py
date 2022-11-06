import numpy as np

from fineslice.datacube import Datacube


def test_matinv_identity():
    dat = np.ones((3, 3, 3), dtype=np.float64)
    aff = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    cube = Datacube(dat, aff)
    assert np.all(cube.affine == cube.affine_inv)
