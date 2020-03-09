# Etienne St-Onge

from __future__ import division

import numpy as np
import scipy
from scipy.sparse import csc_matrix

from trimeshpy.math.angle import (edge_trigo_angle, edge_angle_is_obtuse,
                                  edge_triangle_is_obtuse, edge_cotan_map)
from trimeshpy.math.mesh_map import (edge_triangle_map, edge_sqr_length,
                                     triangle_vertex_map, vertices_degree)
from trimeshpy.math.util import dot_area, dot_cos_angle


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
#    Local Voronoi area of the triangle
#
# Mix Area :
#    an in between mix of Basic and Voronoi Area
#
def triangles_area(triangles, vertices):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()

    e1 = vertices[triangles[:, 1]] - vertices[triangles[:, 0]]
    e2 = vertices[triangles[:, 2]] - vertices[triangles[:, 0]]
    tri_dot_area = dot_area(e1, e2)
    # tri_area = 0.5 * length(scipy.cross(e1, e2))
    return tri_dot_area


def edge_area(triangles, vertices):
    tri_area = triangles_area(triangles, vertices)
    vv_t_area = edge_triangle_map(triangles, vertices)
    vv_t_area.data = tri_area[vv_t_area.data]
    return vv_t_area


def edge_voronoi_area(triangles, vertices):
    vv_l2_sqr_map = edge_sqr_length(triangles, vertices)

    # fast method
    cos_alpha = edge_trigo_angle(triangles, vertices, rot=1,
                                 angle_function=dot_cos_angle).data
    cos_gamma = edge_trigo_angle(triangles, vertices, rot=2,
                                 angle_function=dot_cos_angle).data
    sin2_alpha = 1.0 - cos_alpha*cos_alpha
    sin_alpha = np.sqrt(sin2_alpha)
    sin_gamma = np.sqrt(1.0-cos_gamma*cos_gamma)
    vv_l2_sqr_map.data *= (cos_alpha/sin_alpha + cos_gamma*sin_gamma/sin2_alpha)/8.0
    return vv_l2_sqr_map


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

    e_area = edge_area(triangles, vertices)
    e_voronoi_area = edge_voronoi_area(triangles, vertices)
    tri_is_obtuse = edge_triangle_is_obtuse(triangles, vertices)
    theta_obtuse = edge_angle_is_obtuse(triangles, vertices, rot=0)

    t_not_obt = ~tri_is_obtuse.data
    e_area.data[t_not_obt] = e_voronoi_area.data[t_not_obt]
    e_area.data[np.logical_and(tri_is_obtuse.data, theta_obtuse.data)] /= 2.0
    e_area.data[np.logical_and(tri_is_obtuse.data, ~theta_obtuse.data)] /= 4.0

    return e_area


def vertices_area(triangles, vertices, normalize=False):
    tri_area = triangles_area(triangles, vertices)
    tv_matrix = triangle_vertex_map(triangles, vertices)
    vts_area = tv_matrix.T.dot(tri_area)

    # normalize with the number of triangles
    if normalize:
        vts_area = vts_area / vertices_degree(triangles, vertices)
    return vts_area


def vertices_voronoi_area(triangles, vertices):
    ctn_ab_angles = edge_cotan_map(triangles, vertices)
    vv_l2_sqr_map = edge_sqr_length(triangles, vertices)
    vts_area = ctn_ab_angles.multiply(vv_l2_sqr_map).sum(1) / 8.0
    return np.squeeze(np.asarray(vts_area))


def vertices_mix_area(triangles, vertices):
    return np.squeeze(np.asarray(edge_mix_area(triangles, vertices).sum(1)))
