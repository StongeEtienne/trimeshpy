# Etienne St-Onge

from __future__ import division

import logging

import numpy as np
import scipy
from scipy.sparse import csc_matrix

from trimeshpy.math.util import (is_logging_in_debug, dot_angle,
                                 dot_cos_angle, dot_sin_angle, dot_cotan_angle)
from trimeshpy.math.mesh_map import edge_triangle_map
from trimeshpy.math.mesh_global import G_DTYPE, G_ATOL
from trimeshpy.math.normal import triangles_normal


# Triangle Angles Functions
#
#  vi
#  |\
#  |ai
#  |  \
#  |   \
#  |    \
#  |     \
#  |aj  ak\
# vj------vk
#
#  Triangles : [[ i, j, k ], ....]
#  Angles : [[ ai, aj, ak ], ....]
#

# Generic function to compute Trigonometric function for each angle
def triangle_trigo_angle(triangles, vertices, angle_function):
    """
    :param angle_function: "dot_cos_angle","dot_sin_angle", "dot_cotan_angle", "dot_angle"
    """
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()
    # get theta angles for each points in each triangles
    edges = vertices[np.roll(triangles, 1, axis=1)] - vertices[np.roll(triangles, -1, axis=1)]
    # get the every angles of each triangles (opposite angles)
    tri_dot_angles = np.zeros_like(triangles, dtype=G_DTYPE)
    tri_dot_angles[:, 0] = angle_function(edges[:, 2], -edges[:, 1])
    tri_dot_angles[:, 1] = angle_function(edges[:, 0], -edges[:, 2])
    tri_dot_angles[:, 2] = angle_function(edges[:, 1], -edges[:, 0])
    return tri_dot_angles


def triangle_dot_angle(triangles, vertices):
    angles = triangle_trigo_angle(triangles, vertices, angle_function=dot_angle)
    if is_logging_in_debug() and not np.allclose(angles.sum(-1), np.pi, atol=G_ATOL):
        logging.debug("WARNING triangle_angle does NOT sum to 180deg")
    return angles


def triangle_cos_angle(triangles, vertices):
    return triangle_trigo_angle(triangles, vertices,
                                angle_function=dot_cos_angle)


def triangle_sin_angle(triangles, vertices):
    return triangle_trigo_angle(triangles, vertices,
                                angle_function=dot_sin_angle)


def triangle_cotan_angle(triangles, vertices):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()
    return triangle_trigo_angle(triangles, vertices,
                                angle_function=dot_cotan_angle)


def triangle_is_obtuse(triangles, vertices):
    return np.min(triangle_cos_angle(triangles, vertices), axis=1) < -G_ATOL


def triangle_is_acute(triangles, vertices):
    return np.min(triangle_cos_angle(triangles, vertices), axis=1) > G_ATOL


def triangle_is_right(triangles, vertices):
    return np.abs(np.min(triangle_cos_angle(triangles, vertices), axis=1)) < G_ATOL


# Edge Angles Functions
#
#    for the edge e[i,j] = v[i] -> v[j]
#    Theta(T):0, Alpha(A):1, Gamma(Y):2
#
#             vj
#            /|\
#           / | \
#          Yij|Tji
#         /   |   \
#        /    ^    \
#       /     |     \
#      /      |      \
#     /       |       \
#    /Aij  Tij|Yji  Aji\
# vk ---------o---------  vl
#            vi
#
def edge_trigo_angle(triangles, vertices, rot, angle_function):
    """
    :param rot: 0=Theta(T), 1=Alpha(A), 2=Gamma(Y)
    :param angle_function: "dot_cos_angle","dot_sin_angle", "dot_cotan_angle", "dot_angle"
    """
    t_angles = triangle_trigo_angle(triangles, vertices, angle_function=angle_function)
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])

    if rot == 0:  # Theta(T) angles
        e_angles = np.hstack([t_angles[:, 0], t_angles[:, 1], t_angles[:, 2]])
    elif rot == 1:  # Alpha(A) angles
        e_angles = np.hstack([t_angles[:, 2], t_angles[:, 0], t_angles[:, 1]])
    elif rot == 2:  # Gamma(Y) angles
        e_angles = np.hstack([t_angles[:, 1], t_angles[:, 2], t_angles[:, 0]])
    else:
        raise ValueError("edge_trigo_theta_angle(..., rot), {} \n"
                         "Choose triangle: 0=Theta(T), 1=Alpha(A), 2=Gamma(Y)")

    angles_map = csc_matrix((e_angles, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return angles_map


def edge_cotan_map(triangles, vertices):
    # cotan of all angle = (tan(a))^-1
    ctn_ab_angles = edge_trigo_angle(triangles, vertices, rot=1,
                                     angle_function=dot_cotan_angle)
    # matrix = cot(a) + cot(b) = cot(a_ij) + cot(b_ji)
    ctn_ab_angles = ctn_ab_angles + ctn_ab_angles.T
    return ctn_ab_angles


def edge_triangle_is_obtuse(triangles, vertices):
    tri_is_obtuse = triangle_is_obtuse(triangles, vertices)
    vv_t_is_obtuse = edge_triangle_map(triangles, vertices)
    vv_t_is_obtuse.data = tri_is_obtuse[vv_t_is_obtuse.data]
    return vv_t_is_obtuse


def edge_triangle_is_acute(triangles, vertices):
    tri_is_acute = triangle_is_acute(triangles, vertices)
    vv_t_is_acute = edge_triangle_map(triangles, vertices)
    vv_t_is_acute.data = tri_is_acute[vv_t_is_acute.data]
    return vv_t_is_acute


def edge_triangle_is_right(triangles, vertices):
    tri_is_acute = triangle_is_right(triangles, vertices)
    vv_t_is_acute = edge_triangle_map(triangles, vertices)
    vv_t_is_acute.data = tri_is_acute[vv_t_is_acute.data]
    return vv_t_is_acute


def edge_angle_is_obtuse(triangles, vertices, rot):
    """
    :param rot: 0=Theta(T), 1=Alpha(A), 2=Gamma(Y)
    """
    cos_angle = edge_trigo_angle(triangles, vertices, rot=rot,
                                 angle_function=dot_cos_angle)
    cos_angle.data = cos_angle.data < -G_ATOL
    return cos_angle


def edge_angle_is_acute(triangles, vertices, rot):
    """
    :param rot: 0=Theta(T), 1=Alpha(A), 2=Gamma(Y)
    """
    cos_angle = edge_trigo_angle(triangles, vertices, rot=rot,
                                 angle_function=dot_cos_angle)
    cos_angle.data = cos_angle.data > G_ATOL
    return cos_angle


def edge_angle_is_right(triangles, vertices, rot):
    """
    :param rot: 0=Theta(T), 1=Alpha(A), 2=Gamma(Y)
    """
    cos_angle = edge_trigo_angle(triangles, vertices, rot=rot,
                                 angle_function=dot_cos_angle)
    cos_angle.data = np.abs(cos_angle.data) < G_ATOL
    return cos_angle


# edge slope angle ( triangle's normal angle)
def edge_triangle_normal_angle(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    triangles_index = np.tile(np.arange(len(triangles)), 3)
    vv_t_map = csc_matrix((triangles_index, (vts_i, vts_j)),
                          shape=(vertices.shape[0], vertices.shape[0]))

    t_normals = triangles_normal(triangles, vertices, True)
    t_normals_ij = t_normals[vv_t_map[vts_i, vts_j]]
    t_normals_ji = t_normals[vv_t_map[vts_j, vts_i]]
    e_angles = np.sum(np.squeeze(t_normals_ij * t_normals_ji), axis=1)
    # clamp to 1 due to float precision
    e_angles = np.arccos(np.minimum(1.0, e_angles))

    vv_t_angle_map = csc_matrix((e_angles, (vts_i, vts_j)),
                                shape=(vertices.shape[0], vertices.shape[0]))
    return vv_t_angle_map
