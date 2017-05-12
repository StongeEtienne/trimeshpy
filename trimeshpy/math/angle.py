# Etienne St-Onge

import numpy as np
import scipy

from scipy.sparse import csc_matrix

from trimeshpy.math.util import square_length
from trimeshpy.math.mesh_map import edge_triangle_map

from trimeshpy.math.mesh_global import G_DTYPE, G_ATOL

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
def triangle_angle(triangles, vertices):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()
    # get theta angles for each points in each triangles
    edges_sqr_length = square_length(vertices[np.roll(triangles, 1, axis=1)]
                                     - vertices[np.roll(triangles, -1, axis=1)], axis=2)
    edges_length = np.sqrt(edges_sqr_length)

    # get the every angles of each triangles (opposite angles)
    tri_angles = np.zeros_like(triangles, dtype=G_DTYPE)
    tri_angles[:, 0] = np.arccos((edges_sqr_length[:, 1] + edges_sqr_length[:, 2] - edges_sqr_length[:, 0])
                                 / (2.0 * edges_length[:, 1] * edges_length[:, 2]))
    tri_angles[:, 1] = np.arccos((edges_sqr_length[:, 0] + edges_sqr_length[:, 2] - edges_sqr_length[:, 1])
                                 / (2.0 * edges_length[:, 0] * edges_length[:, 2]))
    tri_angles[:, 2] = (np.pi - tri_angles[:, 0] - tri_angles[:, 1])

    ########################################################################
    # print "min angle ", np.min(tri_angles)
    # print (np.arccos((edges_sqr_length[:,0]+edges_sqr_length[:,1]
    #                   - edges_sqr_length[:,2])/(2.0*edges_length[:,0]*edges_length[:,1])) - tri_angles[:,2])
    ########################################################################
    # assert (np.min(tri_angles) > G_ATOL)
    #assert np.allclose(np.arccos((edges_sqr_length[:, 0] + edges_sqr_length[:, 1] - edges_sqr_length[:, 2]) / (2.0*edges_length[:, 0]*edges_length[:, 1])), tri_angles[:, 2], atol=G_ATOL)
    if not np.allclose(np.arccos((edges_sqr_length[:, 0] + edges_sqr_length[:, 1] - edges_sqr_length[:, 2]) / (2.0*edges_length[:, 0]*edges_length[:, 1])), tri_angles[:, 2], atol=G_ATOL):
        print "WARNING :: triangle_angle"
    
    return tri_angles


def triangle_is_obtuse(triangles, vertices):
    return np.max(triangle_angle(triangles, vertices), axis=1) - G_ATOL > (np.pi / 2.0)


def triangle_is_acute(triangles, vertices):
    return np.max(triangle_angle(triangles, vertices), axis=1) + G_ATOL < (np.pi / 2.0)


def triangle_is_right(triangles, vertices):
    return np.abs(np.max(triangle_angle(triangles, vertices), axis=1) - (np.pi / 2.0)) < G_ATOL


# Edge Angles Functions
#
#    for the edge e[i,j] = v[i] -> v[j]
#    Theta(T), Alpha(A), Gamma(Y)
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
def edge_theta_angle(triangles, vertices):
    # get the every angles of each triangles
    triangles_angles = triangle_angle(triangles, vertices)

    # Get the theta of the next edge
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    theta = np.hstack([triangles_angles[:, 0], triangles_angles[:, 1], triangles_angles[:, 2]])
    # for the beta angle :  beta [j,i] = alpha[i,j]
    theta_map = csc_matrix((theta, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return theta_map


def edge_alpha_angle(triangles, vertices):
    #  alpha_map[i,j] = Aij
    #  alpha_map[i,j] = beta_map[j,i]

    # get the every angles of each triangles
    triangles_angles = triangle_angle(triangles, vertices)

    # Get the Alpha Beta angle for each edge in a connectivity matrix
    # get (directed) edge list  (row = i)(col=j) alpha
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    alpha = np.hstack([triangles_angles[:, 2], triangles_angles[:, 0], triangles_angles[:, 1]])
    # for the beta angle :  beta [j,i] = alpha[i,j]
    alpha_map = csc_matrix((alpha, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return alpha_map


def edge_gamma_angle(triangles, vertices):
    # get the every angles of each triangles
    triangles_angles = triangle_angle(triangles, vertices)

    # Get the theta of the next edge
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    gamma = np.hstack([triangles_angles[:, 1], triangles_angles[:, 2], triangles_angles[:, 0]])
    # for the beta angle :  beta [j,i] = alpha[i,j]
    gamma_map = csc_matrix((gamma, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return gamma_map


def cotan_alpha_beta_angle(triangles, vertices):
    # cotan of all angle = (tan(a))^-1
    ctn_ab_angles = scipy.tan(edge_alpha_angle(triangles, vertices))
    ctn_ab_angles.data **= -1
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


def edge_theta_is_obtuse(triangles, vertices):
    theta = edge_theta_angle(triangles, vertices)
    theta.data = (theta.data - G_ATOL > (np.pi / 2.0))
    return theta


def edge_theta_is_acute(triangles, vertices):
    theta = edge_theta_angle(triangles, vertices)
    theta.data = (theta.data + G_ATOL < (np.pi / 2.0))
    return theta


def edge_slope_angle(triangles, vertices):
    theta = edge_theta_angle(triangles, vertices)
    theta.data = (theta.data + G_ATOL < (np.pi / 2.0))
    return theta


# edge slope angle ( triangle's normal angle)
def edge_triangle_normal_angle(triangles, vertices):
    from trimeshpy.math.normal import triangles_normal
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    triangles_index = np.tile(np.arange(len(triangles)), 3)
    vv_t_map = csc_matrix((triangles_index, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    
    triangle_normals = triangles_normal(triangles, vertices, True)
    
    e_angles = np.sum(np.squeeze(triangle_normals[vv_t_map[vts_i, vts_j]]
                                 * triangle_normals[vv_t_map[vts_j, vts_i]]), axis=1)
    e_angles = np.arccos(np.minimum(1, e_angles))  # clamp to 1 due to float precision
    
    vv_t_angle_map = csc_matrix((e_angles, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return vv_t_angle_map
