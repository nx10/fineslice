# `fineslice`

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