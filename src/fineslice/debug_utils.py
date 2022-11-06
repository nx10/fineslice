import numpy as np


def plot_poly(vertices, edges=None, inters=None, labels=None):
    """
    Plot polygon
    :param vertices:
    :param edges:
    :param inters:
    :param labels:
    :return:
    """
    try:
        from matplotlib import pyplot as plt  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        raise RuntimeError('matplotlib needs to be installed for this functionality') from exc

    labs = ('X', 'Y', 'Z') if labels is None else labels
    ax = plt.axes(projection='3d', xlabel=labs[0], ylabel=labs[1], zlabel=labs[2])
    ax.scatter3D(vertices[0], vertices[1], vertices[2], c=np.arange(vertices.shape[1]), cmap='Set1')

    if edges is not None:
        for e in edges:
            ax.plot3D(vertices[0, e], vertices[1, e], vertices[2, e])

    if inters is not None:
        ax.scatter3D(inters[0], inters[1], inters[2], c="red")

    return ax
