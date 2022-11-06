from typing import Union, Iterable, Optional

import numpy as np


class Datacube:
    def __init__(self, image: np.ndarray, affine: np.ndarray, affine_inv: Optional[np.ndarray] = None):
        self.image = image
        self.affine = affine
        self.affine_inv = np.linalg.inv(self.affine) if affine_inv is None else affine_inv

    def copy(self):
        return Datacube(
            image=self.image.copy(),
            affine=self.image.copy(),
            affine_inv=self.affine_inv.copy()
        )

    def transform(self, p: Union[np.ndarray, Iterable]):
        """
        Data space -> reference space
        """
        return np.dot(self.affine, p)

    def transform_inv(self, p: Union[np.ndarray, Iterable]):
        """
        Reference space -> data space
        """
        return np.dot(self.affine_inv, p)
