"""Debug utilities for the fineslice package."""

from typing import Any, Optional, Sequence

import numpy as np


def plot_poly(
    vertices: Any,  # noqa: ANN401
    edges: Optional[Any] = None,  # noqa: ANN401
    inters: Optional[Any] = None,  # noqa: ANN401
    labels: Optional[Sequence[str]] = None,
    show: bool = False,
) -> Any:  # noqa: ANN401
    """Plot polygon."""
    try:
        from matplotlib import pyplot as plt  # noqa
    except Exception as exc:
        raise RuntimeError(
            "matplotlib needs to be installed for this functionality"
        ) from exc

    labs = ("X", "Y", "Z") if labels is None else labels
    ax = plt.axes(projection="3d", xlabel=labs[0], ylabel=labs[1], zlabel=labs[2])
    ax.scatter3D(
        vertices[0],
        vertices[1],
        vertices[2],
        c=np.arange(vertices.shape[1]),
        cmap="Set1",
    )

    if edges is not None:
        for e in edges:
            ax.plot3D(vertices[0, e], vertices[1, e], vertices[2, e])

    if inters is not None:
        ax.scatter3D(inters[0], inters[1], inters[2], c="red")

    if show:
        plt.show()

    return ax
