from typing import Optional

import numpy as np


def intersect_line_plane(
        line_origin: np.ndarray,
        line_dir: np.ndarray,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        epsilon=1e-6) -> Optional[np.ndarray]:
    """Find intersection point of a line with a plane.
    Returns None if there is no intersection.

    TODO: Can this be sped up if either plane or line is axis aligned?
    """
    nu = plane_normal.dot(line_dir)

    if abs(nu) < epsilon:
        return None

    w = line_origin - plane_origin
    si = -plane_normal.dot(w) / nu
    psi = w + si * line_dir + plane_origin

    return psi


def intersect_line_segment_plane(
        point1: np.ndarray,
        point2: np.ndarray,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        epsilon=1e-6) -> Optional[np.ndarray]:
    """Find intersection point of a line segment (between 2 points) with a plane.
    Returns None if there is no intersection.
    """
    direction = point2 - point1
    orig = point1

    inter = intersect_line_plane(
        line_origin=orig,
        line_dir=direction,
        plane_origin=plane_origin,
        plane_normal=plane_normal,
        epsilon=epsilon
    )

    if inter is None:
        return None

    between_dist = np.linalg.norm(direction)
    point1_dist = np.linalg.norm(inter - point1)
    point2_dist = np.linalg.norm(inter - point2)

    if (between_dist + epsilon) < (point1_dist + point2_dist):
        return None

    return inter


def intersect_polygon_plane(
        vertices: np.ndarray,
        edges: np.ndarray,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray
) -> Optional[np.ndarray]:
    """Intersect polygon with plane. TODO

    Args:
        vertices
        edges
        plane_origin
        plane_normal

    Returns:

    """
    inters = []
    for e in edges:
        line = vertices[0:3, e]

        inter = intersect_line_segment_plane(
            point1=line[:, 0],
            point2=line[:, 1],
            plane_origin=plane_origin,
            plane_normal=plane_normal)

        if inter is not None:
            inters.append(inter)

    re = np.array(inters, dtype=np.float64).T
    if re.shape[0] == 0:
        return None
    return np.vstack([re, np.ones(re.shape[1])])
