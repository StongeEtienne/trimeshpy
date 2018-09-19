# Etienne St-Onge

from __future__ import division
import numpy as np
from scipy.sparse import diags

from trimeshpy.math.util import dot
from trimeshpy.math.mesh_global import G_DTYPE, G_ATOL


# Mesh operation matrix
def laplacian(adjacency_matrix, diag_of_1=True):
    # A_ij = adjacency_map ( or weighted connectivity map )
    # D_ii = weights = Diag(sum(A_ij, axis=j)) = weights
    # L_ij = D_ii - A_ij
    # => Sum(laplacian_mtx, axis=1) = 0.0
    weights = np.squeeze(np.array(adjacency_matrix.sum(1)))
    laplacian_mtx = adjacency_matrix - diags(weights, 0)

    if diag_of_1:
        # normalize each row
        laplacian_mtx = diags(1.0 / weights, 0).dot(laplacian_mtx)
    else:
        # normalize by the max
        laplacian_mtx = laplacian_mtx / np.abs(laplacian_mtx.data).max()

    # assert(np.allclose(laplacian_mtx.sum(1), 0.0, atol=G_ATOL))
    if not np.allclose(laplacian_mtx.sum(1), 0.0, atol=G_ATOL):
        print("WARNING :: laplacian")
    return laplacian_mtx


#  stiffness_matrix
def mean_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    from trimeshpy.math.angle import cotan_alpha_beta_angle
    from trimeshpy.math.area import vertices_mix_area
    ctn_ab_angle = cotan_alpha_beta_angle(triangles, vertices)

    if area_weighted:
        # Kn(xi) = 0.5/Ai * Sum_j( ctn_ab[i,j](xj-xi) )
        # Kn(xi) = 0.5/Ai * Sum_j(ctn_ab[i,j]*xj) - xi*Sum_j(ctn_ab[i,j)
        # Kn(xi) = 0.5/Ai * L(ctn_ab[i,j])*X
        lap_ctn_ab_angle = laplacian(ctn_ab_angle, diag_of_1=False)
        vts_mix_area = vertices_mix_area(triangles, vertices)
        curvature_normal_mtx = diags(
            vts_mix_area.min() / vts_mix_area, 0).dot(lap_ctn_ab_angle)
    else:
        curvature_normal_mtx = laplacian(ctn_ab_angle, diag_of_1=True)

    # assert(np.allclose(curvature_normal_mtx.sum(1), 0.0, atol=G_ATOL))
    if not np.allclose(curvature_normal_mtx.sum(1), 0.0, atol=G_ATOL):
        print("WARNING :: mean_curvature_normal_matrix")
    return curvature_normal_mtx


def positive_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    from trimeshpy.math.normal import vertices_normal
    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices, area_weighted)

    # get flow vector and normal vector
    direction = curvature_normal_mtx.dot(vertices)
    normal_dir = vertices_normal(triangles, vertices, normalize=False)
    # positive curvature goes opposite to the normal
    pos_curv = (dot(direction, normal_dir, axis=1) < 0).astype(G_DTYPE)
    curvature_normal_mtx = diags(pos_curv, 0).dot(curvature_normal_mtx)
    return curvature_normal_mtx


def mass_matrix(triangles, vertices):
    from trimeshpy.math.area import triangles_area
    from trimeshpy.math.mesh_map import edge_triangle_map
    tri_area = triangles_area(triangles, vertices)
    e_area = edge_triangle_map(triangles, vertices)
    e_area.data = tri_area[e_area.data]
    # e_area = edge_mix_area(triangles, vertices)  # with vertices mix area
    # edge_voronoi_area
    e_area = (e_area + e_area.T) / 12.0
    weights = np.squeeze(np.array(e_area.sum(1)))
    mass_mtx = e_area + diags(weights, 0)
    return mass_mtx
