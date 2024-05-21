# `fineslice`
[![Build](https://github.com/nx10/fineslice/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/nx10/fineslice/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/nx10/fineslice/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/nx10/fineslice)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD_3--Clause-blue.svg)](https://github.com/nx10/fineslice/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/template-python-repository)

`fineslice` is a lightweight sampler for 3D-affine transformed images (commonly used in neuroscience) implemented in 
pure Python + NumPy.

It does not make any assumptions about the data. Pass _any_ image texture and affine matrix directly into it.

### Features

- Precision sampling (no need to 're-sample' and loose precision)
- Automatically finds optimal dimensions
- Only depends on NumPy

### Usage with `nibabel`

For the best performance directly pass in the `nibabel` data object as a texture:

```Python
import nibabel as nib
import fineslice as fine

img = nib.load('my_image.nii.gz')

out = fine.sample_0d(
    texture=img.dataobj,
    affine=img.affine,
    out_position=(0, 0, 0)
)
```