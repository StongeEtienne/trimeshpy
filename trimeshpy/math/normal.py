# Etienne St-Onge

from __future__ import division
import numpy as np
import scipy

from trimeshpy.math.mesh_map import triangle_vertex_map
from trimeshpy.math.util import dot, normalize_vectors


# Normal Functions
# Triangle Normal right handed triangle orientation
def triangles_normal(triangles, vertices, normalize=True):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()
    e1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normal = np.cross(e1, e2)

    if normalize:
        return normalize_vectors(normal)
    return normal


def vertices_normal(triangles, vertices, normalize=True, area_weighted=True):
    tri_normal = triangles_normal(triangles, vertices, normalize=area_weighted)
    tv_matrix = triangle_vertex_map(triangles, vertices)
    vts_normal = tv_matrix.T.dot(tri_normal)

    if normalize:
        return normalize_vectors(vts_normal)
    return vts_normal


def vertices_cotan_normal(
        triangles, vertices, normalize=True, area_weighted=True):
    from trimeshpy.math.curvature import vertices_cotan_direction
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()

    cotan_normal = vertices_cotan_direction(
        triangles, vertices, normalize=False, area_weighted=area_weighted)
    vts_normal = vertices_normal(
        triangles, vertices, normalize=False, area_weighted=area_weighted)

    # inverse inverted cotan_normal direction
    cotan_normal = np.sign(
        dot(cotan_normal, vts_normal, keepdims=True)) * cotan_normal

    if normalize:
        return normalize_vectors(cotan_normal)
    return cotan_normal
