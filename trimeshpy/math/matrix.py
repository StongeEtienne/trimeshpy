# Etienne St-Onge

from __future__ import division

import logging

import numpy as np
from scipy.sparse import diags

from trimeshpy.math.util import dot, allclose_to, is_logging_in_debug
from trimeshpy.math.mesh_global import G_DTYPE

from trimeshpy.math.angle import edge_cotan_map
from trimeshpy.math.area import vertices_mix_area, triangles_area
from trimeshpy.math.mesh_map import edge_triangle_map
from trimeshpy.math.normal import vertices_normal


# Mesh operation matrix
def laplacian(adjacency_matrix, diag_of_1=True):
    # A_ij = adjacency_map ( or weighted connectivity map )
    # D_ii = weights = Diag(sum(A_ij, axis=j)) = weights
    # L_ij = D_ii - A_ij
    # => Sum(laplacian_mtx, axis=1) = 0.0
    weights = np.squeeze(np.asarray(adjacency_matrix.sum(1)))
    laplacian_mtx = adjacency_matrix - diags(weights, 0)

    if diag_of_1:
        # normalize each row
        laplacian_mtx = diags(1.0 / weights, 0).dot(laplacian_mtx)
    else:
        # normalize by the max
        laplacian_mtx = laplacian_mtx / np.abs(laplacian_mtx.data).max()

    if is_logging_in_debug() and not allclose_to(laplacian_mtx.sum(1), 0.0):
        logging.debug("WARNING laplacian_mtx.sum(1) does NOT sum to zero")
    return laplacian_mtx


#  stiffness_matrix
def mass_matrix(triangles, vertices):
    tri_area = triangles_area(triangles, vertices)
    e_area = edge_triangle_map(triangles, vertices)
    e_area.data = tri_area[e_area.data]
    # e_area = edge_mix_area(triangles, vertices)  # with vertices mix area
    # edge_voronoi_area
    e_area = (e_area + e_area.T) / 12.0
    weights = np.squeeze(np.asarray(e_area.sum(1)))
    mass_mtx = e_area + diags(weights, 0)
    return mass_mtx


def mean_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    cotan_map = edge_cotan_map(triangles, vertices)

    if area_weighted:
        # Kn(xi) = 0.5/Ai * Sum_j( ctn_ab[i,j](xj-xi) )
        # Kn(xi) = 0.5/Ai * Sum_j(ctn_ab[i,j]*xj) - xi*Sum_j(ctn_ab[i,j)
        # Kn(xi) = 0.5/Ai * L(ctn_ab[i,j])*X
        lap_ctn_ab_angle = laplacian(cotan_map, diag_of_1=False)
        vts_mix_area = vertices_mix_area(triangles, vertices)
        curv_normal_mtx = diags(
            vts_mix_area.min() / vts_mix_area, 0).dot(lap_ctn_ab_angle)
    else:
        curv_normal_mtx = laplacian(cotan_map, diag_of_1=True)

    if is_logging_in_debug() and not allclose_to(curv_normal_mtx.sum(1), 0.0):
        logging.debug("WARNING curv_normal_mtx.sum(1) does NOT sum to zero")
    return curv_normal_mtx


def positive_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    curv_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices, area_weighted)

    # get flow vector and normal vector
    direction = curv_normal_mtx.dot(vertices)
    normal_dir = vertices_normal(triangles, vertices, normalize=False)
    # positive curvature goes opposite to the normal
    pos_curv = (dot(direction, normal_dir, axis=1) < 0).astype(G_DTYPE)
    curv_normal_mtx = diags(pos_curv, 0).dot(curv_normal_mtx)
    return curv_normal_mtx

