# Etienne St-Onge

from __future__ import division

import numpy as np
import scipy

from trimeshpy.math.angle import edge_trigo_angle
from trimeshpy.math.area import vertices_mix_area
from trimeshpy.math.normal import vertices_normal
from trimeshpy.math.matrix import mean_curvature_normal_matrix
from trimeshpy.math.util import dot, length, normalize_vectors, dot_angle


def vertices_cotan_normal(
        triangles, vertices, normalize=True, area_weighted=True):
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


def vertices_gaussian_curvature(triangles, vertices, area_weighted=False):
    theta_matrix = edge_trigo_angle(triangles, vertices, rot=0,
                                    angle_function=dot_angle)
    gaussian_curvature = np.squeeze(np.asarray(2.0 * np.pi - theta_matrix.sum(1)))
    if area_weighted:
        vts_mix_area = vertices_mix_area(triangles, vertices)
        gaussian_curvature = gaussian_curvature / vts_mix_area
    return np.squeeze(np.asarray(gaussian_curvature))


def vertices_cotan_curvature(triangles, vertices, area_weighted=True):
    vts_dir = vertices_cotan_direction(triangles, vertices, normalize=False,
                                       area_weighted=area_weighted)
    vts_normal = vertices_normal(triangles, vertices, normalize=False,
                                 area_weighted=area_weighted)
    curvature_sign = -np.sign(dot(vts_dir, vts_normal))
    curvature = length(vts_dir)

    return curvature * curvature_sign


def vertices_cotan_direction(triangles, vertices,
                             normalize=True, area_weighted=True):
    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices, area_weighted=area_weighted)
    cotan_normal = curvature_normal_mtx.dot(vertices)

    if normalize:
        return normalize_vectors(cotan_normal)
    return cotan_normal
