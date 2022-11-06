from enum import Enum


class Filter(Enum):
    NEAREST = 1
    LINEAR = 2
    QUADRATIC = 3
    CUBIC = 4
    SPLINE = 5
