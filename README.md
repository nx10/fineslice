# `fineslice`

`fineslice` is a lightweight sampler for 3D-affine transformed images (commonly used in neuroscience) implemented in 
pure Python + NumPy.

It does not make any assumptions about the data. Pass _any_ image texture and affine matrix directly into it.

### Features

- Precision sampling (no need to 're-sample' and loose precision)
- Automatically finds optimal dimensions
- Only depends on NumPy

## TODO

`fineslice` is an early prototype and far from ready for production use.

Sampling

- [x] 0D / point
- [ ] 1D / line
- [x] 2D / rectangle
- [x] 3D / cube

Texture filtering (interpolation)

- [x] Nearest-neighbour
- [ ] (Tri-)Linear
- [ ] (Tri-)Quadratic
- [ ] (Tri-)Cubic
- [ ] Spline
