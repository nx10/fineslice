"""Tests for the 0D sampler."""

import numpy as np

import fineslice as fine


def test_sampler_0d_identity() -> None:
    """Test identity textures."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=int), (3, 3, 3))
    affine = fine.affine_identity()

    assert fine.sample_0d(texture, affine, (0, 0, 0)) == 0
    assert fine.sample_0d(texture, affine, (1, 0, 0)) == 9
    assert fine.sample_0d(texture, affine, (0, 1, 0)) == 3
    assert fine.sample_0d(texture, affine, (0, 0, 1)) == 1


def test_sampler_0d_translate() -> None:
    """Test translated textures."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=int), (3, 3, 3))
    affine = fine.affine_identity()
    tx = 3
    ty = 10
    tz = 94
    affine_translated = fine.affine_translate(affine, tx, ty, tz)

    assert fine.sample_0d(texture, affine, (0, 0, 0)) == fine.sample_0d(
        texture, affine_translated, (tx, ty, tz)
    )
    assert fine.sample_0d(texture, affine, (2, 2, 2)) == fine.sample_0d(
        texture, affine_translated, (tx + 2, ty + 2, tz + 2)
    )


def test_sampler_0d_scale() -> None:
    """Test scaled textures."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=np.float64), (3, 3, 3))
    affine = fine.affine_identity()
    sx = 0.5
    sy = 2
    sz = -1
    affine_scaled = fine.affine_scale(affine, sx, sy, sz)

    assert fine.sample_0d(texture, affine, (1, 1, 1)) == fine.sample_0d(
        texture, affine_scaled, (sx, sy, sz)
    )
    assert fine.sample_0d(texture, affine, (2, 2, 2)) == fine.sample_0d(
        texture, affine_scaled, (sx * 2, sy * 2, sz * 2)
    )


def test_sampler_0d_rotate() -> None:
    """Test rotated textures."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=np.float64), (3, 3, 3))
    affine = fine.affine_identity()
    affine_trans = fine.affine_translate(
        affine, -1, -1, -1
    )  # offset to rotate around center
    affine_rx90 = fine.affine_rotate_degrees(
        fine.affine_invert(affine_trans), x=90
    ).dot(affine_trans)

    assert fine.sample_0d(texture, affine, (0, 2, 0)) == fine.sample_0d(
        texture, affine_rx90, (0, 0, 0)
    )
    assert fine.sample_0d(texture, affine, (1, 1, 1)) == fine.sample_0d(
        texture, affine_rx90, (1, 1, 1)
    )
    assert fine.sample_0d(texture, affine, (2, 0, 2)) == fine.sample_0d(
        texture, affine_rx90, (2, 2, 2)
    )


def test_sampler_0d_shear() -> None:
    """Test sheared textures."""
    # todo
    pass


def test_sampler_0d_outside_bounds() -> None:
    """Test sampling outside of the texture bounds."""
    texture = np.reshape(np.arange(3 * 3 * 3, dtype=int), (3, 3, 3))
    affine = fine.affine_identity()

    assert fine.sample_0d(texture, affine, (-1, 0, 0)) is None
    assert fine.sample_0d(texture, affine, (0, -1, 0)) is None
    assert fine.sample_0d(texture, affine, (0, 0, -1)) is None
    assert fine.sample_0d(texture, affine, (4, 0, 0)) is None
    assert fine.sample_0d(texture, affine, (0, 4, 0)) is None
    assert fine.sample_0d(texture, affine, (0, 0, 4)) is None
    assert fine.sample_0d(texture, affine, (100, 100, 100)) is None
    assert fine.sample_0d(texture, affine, (-100, -100, -100)) is None
