# Etienne St-Onge

from __future__ import division
import numpy as np
import scipy

from scipy.sparse import csc_matrix

from trimeshpy.math.mesh_map import edge_triangle_map, edge_sqr_length, triangle_vertex_map, vertices_degree
from trimeshpy.math.util import length


# Area Functions
#
#        vi
#        /\
# t[a]: /  \
#      /    \
#     /      \
#    / Area_a \
#   /          \
#  vj---->-----vk
#
#  t[a] = [i, j, k]
#  t[a] formed by vertices v[i], v[j], v[k]
#  t[a] formed by edges e[i,j], e[j,k], e[k,i]
#
# Basic Triangle Area:
#  triangles_area[a] = Area_a
#  edge_area[i,j] = edge_area[j,k] = edge_area[k,i] = Area_a
#  vertices_area[i] = Sum(edge_area[i,:]) or Mean(edge_area[i,:]),
#                      if normalize=False or normalize=True
#
# Voronoi Area :
#    todo add info
#
# Mix Area :
#    an in between mix of Basic and Voronoi Area
#
def triangles_area(triangles, vertices):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()

    e1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    normal = scipy.cross(e1, e2)
    tri_area = 0.5 * length(normal)
    return tri_area


def edge_area(triangles, vertices):
    tri_area = triangles_area(triangles, vertices)
    vv_t_area = edge_triangle_map(triangles, vertices)
    vv_t_area.data = tri_area[vv_t_area.data]
    return vv_t_area


def edge_voronoi_area(triangles, vertices):
    from trimeshpy.math.angle import edge_alpha_angle, edge_gamma_angle
    vv_l2_sqr_map = edge_sqr_length(triangles, vertices)

    alpha = edge_alpha_angle(triangles, vertices)
    gamma = edge_gamma_angle(triangles, vertices)

    cot_alpha = scipy.tan(alpha)
    cot_alpha.data **= -1
    inv_sin2_alpha = scipy.sin(alpha)
    inv_sin2_alpha.data **= -2

    w_angle = scipy.sin(2.0 * gamma).multiply(inv_sin2_alpha) / 2.0
    vv_area = vv_l2_sqr_map.multiply(cot_alpha + w_angle) / 8.0
    return vv_area


def edge_mix_area(triangles, vertices):
    ########################################################################
    # if (triangle not obtuse)
    #    area = triangle voronoi area
    # else (if triangle obtuse)
    #    if current vertex angle(theta) >= 90
    #        area = (triangle area)/2
    #    else : (if current vertex angle(theta) < 90)
    #        area = (triangle area)/4
    ########################################################################
    from trimeshpy.math.angle import edge_triangle_is_obtuse
    from trimeshpy.math.angle import edge_theta_is_obtuse

    e_area = edge_area(triangles, vertices)
    e_voronoi_area = edge_voronoi_area(triangles, vertices)
    tri_is_obtuse = edge_triangle_is_obtuse(triangles, vertices)
    theta_obtuse = edge_theta_is_obtuse(triangles, vertices)

    # todo : optimize
    tri_is_not_obt = csc_matrix.copy(tri_is_obtuse)
    tri_is_not_obt.data = ~tri_is_not_obt.data
    theta_not_obtuse = csc_matrix.copy(theta_obtuse)
    theta_not_obtuse.data = ~theta_not_obtuse.data
    mix_area = (tri_is_not_obt.multiply(e_voronoi_area) +
                tri_is_obtuse.multiply(theta_obtuse).multiply(e_area) / 2 +
                tri_is_obtuse.multiply(theta_not_obtuse).multiply(e_area) / 4)

    return mix_area


def vertices_area(triangles, vertices, normalize=False):
    tri_area = triangles_area(triangles, vertices)
    tv_matrix = triangle_vertex_map(triangles, vertices)
    vts_area = tv_matrix.T.dot(tri_area)

    # normalize with the number of triangles
    if normalize:
        vts_area = vts_area / vertices_degree(triangles, vertices)
    return vts_area


def vertices_voronoi_area(triangles, vertices):
    from trimeshpy.math.angle import cotan_alpha_beta_angle
    ctn_ab_angles = cotan_alpha_beta_angle(triangles, vertices)
    vv_l2_sqr_map = edge_sqr_length(triangles, vertices)
    vts_area = ctn_ab_angles.multiply(vv_l2_sqr_map).sum(1) / 8.0
    return np.squeeze(np.array(vts_area))


def vertices_mix_area(triangles, vertices):
    return np.squeeze(np.array(edge_mix_area(triangles, vertices).sum(1)))
