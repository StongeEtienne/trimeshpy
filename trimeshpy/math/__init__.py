# Etienne.St-Onge@usherbrooke.ca

# import all MATH sub modules
# and general libraries

import numpy as np

from sys import stdout

import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, diags, identity


G_DTYPE = np.float64
G_ATOL = 1e-8

### local import
"""
from util import *
from map import *
from transfo import *
from angle import *
from area import *
from normal import *
from curvature import *
from matrix import *
from smooth import *
"""

# TODO add function I/O comments
# TODO change global to parameters
# TODO add decorator to do automatic class ?


# Mesh structure functions
# Map ( Adjacency / Connectivity ) Functions
#
#    number of vertices = n
#    number of triangles = m
#    number of edges = l = m*3  (directed edge)
#
#    vertices array : n x 3
#        v[i] = [ x, y, z ]
#
#    triangles array : m x 3
#        t[a] = [ v[i], v[j], v[k] ]
#        right handed triangles
#
#
# Example :
#
#        vj_ _ _ _ vo
#         /\      /\
#        /  \ tf /  \
#       / ta \  / te \
#    vk/_ _ _ vi _ _ _\ vn
#      \      /\      /
#       \ tb /  \ td /
#        \  / tc \  /
#         \/_ _ _ \/
#         vl       vm
#
#    Vertices = [v[i] = [x_i, y_i, z_i],
#                v[j] = [x_j, y_j, z_j],
#                v[k] = [x_k, y_k, z_k],
#                v[l] = [x_l, y_l, z_l],
#                v[m] = [x_m, y_m, z_m],
#                v[n] = [x_n, y_n, z_n],
#                v[o] = [x_o, y_o, z_o]]
#
#    Triangles = [t[a] = [i, j, k],
#                 t[b] = [i, k, l],
#                 t[c] = [i, l, m],
#                 t[d] = [i, m, n],
#                 t[e] = [i, n, o],
#                 t[f] = [i, o, j]]
#
#    triangle_vertex_map : m x n -> boolean, loss of orientation
#    t_v[] v[i] v[j] v[k] v[l] v[m] v[n] v[o]
#     t[a]   1    1    1
#     t[b]   1         1    1
#     t[c]   1              1    1
#     t[d]   1                   1    1
#     t[e]   1                        1    1
#     t[f]   1    1                        1
#
#   Edges Maps
#    edge_adjacency :  n x n -> boolean, not symmetric if mesh not closed
#    e[,] v[i] v[j] v[k] v[l] v[m] v[n] v[o]
#    v[i]        1    1    1    1    1    1
#    v[j]   1         1
#    v[k]   1              1
#    v[l]   1                   1
#    v[m]   1                        1
#    v[n]   1                             1
#    v[o]   1    1
#
#    edge_triangle_map : n x n -> triangle_index
#    e_t[,] v[i] v[j] v[k] v[l] v[m] v[n] v[o]
#      v[i]        a    b    c    d    e    f
#      v[j]   f         a
#      v[k]   a              b
#      v[l]   b                   c
#      v[m]   c                        d
#      v[n]   d                             e
#      v[o]   e    f
#
#    edge_opposing_vertex : n x n -> vertex_index
#    e_ov[] v[i] v[j] v[k] v[l] v[m] v[n] v[o]
#      v[i]        k    l    m    n    o    j
#      v[j]   o         i
#      v[k]   j              i
#      v[l]   k                   i
#      v[m]   l                        i
#      v[n]   m                             i
#      v[o]   n    i
#

#    edge_adjacency : n x n -> boolean (sparse connectivity matrix)
#    e[i,j] = v[i] -> v[j] = { 1, if connected }
def edge_adjacency(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    values = np.ones_like(vts_i, dtype=np.bool)
    vv_map = csc_matrix((values, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return vv_map


#    edge_sqr_length : n x n -> float (sparse connectivity matrix)
#    e[i,j] = v[i] -> v[j] = { || v[i] - v[j] ||^2, if connected }
def edge_sqr_length(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    values = square_length(vertices[vts_i] - vertices[vts_j])
    vv_map = csc_matrix((values, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]), dtype=G_DTYPE)
    return vv_map


#    edge_length : n x n -> float (sparse connectivity matrix)
#    e[i,j] = v[i] -> v[j] = { || v[i] - v[j] ||, if connected }
def edge_length(triangles, vertices):
    vv_map = edge_sqr_length(triangles, vertices)
    vv_map.data = np.sqrt(vv_map.data)
    return vv_map


#    edge_sqr_length : n x n -> float (sparse connectivity matrix)
#    e[i,j] = { edge_length,    if l2_weighted }
def edge_map(triangles, vertices, l2_weighted=False):
    if l2_weighted:
        return edge_length(triangles, vertices)
    else:
        return edge_adjacency(triangles, vertices)


#    edge_triangle_map : n x n -> triangle_index (sparse connectivity matrix)
#    e_t[i,j] = e[i,j] -> t[a] = { 1, if triangle[a] is compose of edge[i,j] }
def edge_triangle_map(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    triangles_index = np.tile(np.arange(len(triangles)), 3)

    vv_t_map = csc_matrix((triangles_index, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return vv_t_map


#    edge_opposing_vertex : n x n -> vertex_index (int) (sparse connectivity matrix)
#    e[i,j] = v[i],v[j] = { v[k], if v[i],v[j],v[k] triangle exist }
def edge_opposing_vertex(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    vts_k = np.hstack([triangles[:, 2], triangles[:, 0], triangles[:, 1]])

    vv_v_map = csc_matrix((vts_k, (vts_i, vts_j)), shape=(vertices.shape[0], vertices.shape[0]))
    return vv_v_map


#    triangle_vertex_map : m x n -> bool (sparse connectivity matrix)
#    t_v[i,a] = t[a] <-> v[i] = { 1, if triangle[a] is compose of vertex[i] }
def triangle_vertex_map(triangles, vertices):
    triangles_index = np.repeat(np.arange(len(triangles)), 3)
    vertices_index = np.hstack(triangles)
    values = np.ones_like(triangles_index, dtype=np.bool)

    tv_map = csc_matrix((values, (triangles_index, vertices_index)), shape=(len(triangles), vertices.shape[0]))
    return tv_map


def vertices_degree(triangles, vertices):
    tv_matrix = triangle_vertex_map(triangles, vertices)
    return np.squeeze(np.array(tv_matrix.sum(0)))


# Points Transformations Functions
#    vertices array : n x 3
#        v[i] = [ x, y, z ]
#
#    Translation: 3x1
#    Rotation: 3x3
#    General transformation: 4x4
#
def vertices_translation(triangles, vertices, translation):
    # translation = [t_x, t_y, t_z]
    return vertices + translation


def vertices_rotation(triangles, vertices, rotation):
    # rotation = [[rx1, ry1, rz1], 
    #             [rx2, ry2, rz2],
    #             [rx3, ry3, rz3]] # todo fix error
    # raise NotImplementedError()
    return np.dot(vertices, rotation)


def vertices_transformation(triangles, vertices, transfo):
    # transfo = [[rx1, ry1, rz1, s_1], 
    #            [rx2, ry2, rz2, s_2],
    #            [rx3, ry3, rz3, s_3],
    #            [t_x, t_y, t_z,  1 ]] # todo test
    raise NotImplementedError()


def flip_triangle_and_vertices(triangles, vertices, flip=[1, 1, 1]):
    # flip = [ f_x, f_y, f_y] : -1 to flip/ 1 else # todo test
    # and autorotate face if impair flip
    vertices = vertices*flip
    if np.sum(np.equal(flip, -1))%2 == 1:
        triangles_face_flip(triangles, vertices)
    return triangles, vertices

def vertices_flip(triangles, vertices, flip=[1, 1, 1]):
    # flip = [ f_x, f_y, f_y] : -1 to flip/ 1 else # todo test
    return vertices*flip


def triangles_face_flip(triangles, vertices, flip=[0, 2]):
    # flip triangle face "flip normal", 
    # flip = [i, j] to flip i,j columns to flip
    triangles[:, flip] = triangles[:, flip[::-1]]
    return triangles

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
    vv_area = edge_area(triangles, vertices)
    vv_voronoi_area = edge_voronoi_area(triangles, vertices)
    tri_is_obtuse = edge_triangle_is_obtuse(triangles, vertices)
    theta_obtuse = edge_theta_is_obtuse(triangles, vertices)

    # todo : optimize
    tri_is_not_obt = csc_matrix.copy(tri_is_obtuse)
    tri_is_not_obt.data = ~tri_is_not_obt.data
    theta_not_obtuse = csc_matrix.copy(theta_obtuse)
    theta_not_obtuse.data = ~theta_not_obtuse.data
    vv_mix_area = (tri_is_not_obt.multiply(vv_voronoi_area)
                   + tri_is_obtuse.multiply(theta_obtuse).multiply(vv_area) / 2
                   + tri_is_obtuse.multiply(theta_not_obtuse).multiply(vv_area) / 4)

    return vv_mix_area


def vertices_area(triangles, vertices, normalize=False):
    tri_area = triangles_area(triangles, vertices)
    tv_matrix = triangle_vertex_map(triangles, vertices)
    vts_area = tv_matrix.T.dot(tri_area)

    # normalize with the number of triangles
    if normalize:
        vts_area = vts_area / vertices_degree(triangles, vertices)
    return vts_area


def vertices_voronoi_area(triangles, vertices):
    ctn_ab_angles = cotan_alpha_beta_angle(triangles, vertices)
    vv_l2_sqr_map = edge_sqr_length(triangles, vertices)
    vts_area = ctn_ab_angles.multiply(vv_l2_sqr_map).sum(1) / 8.0
    return np.squeeze(np.array(vts_area))


def vertices_mix_area(triangles, vertices):
    return np.squeeze(np.array(edge_mix_area(triangles, vertices).sum(1)))

# edge slope angle ( triangle's normal angle)
def edge_triangle_normal_angle(triangles, vertices):
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


def vertices_gaussian_curvature(triangles, vertices, area_weighted=False):
    theta_matrix = edge_theta_angle(triangles, vertices)
    gaussian_curvature = np.squeeze(np.array(2.0 * np.pi - theta_matrix.sum(1)))
    if area_weighted:
        vts_mix_area = vertices_mix_area(triangles, vertices)
        gaussian_curvature = gaussian_curvature / vts_mix_area
    return np.squeeze(np.array(gaussian_curvature))


def vertices_cotan_curvature(triangles, vertices, area_weighted=True):
    vts_dir = vertices_cotan_direction(triangles, vertices, normalize=False, area_weighted=area_weighted)
    vts_normal = vertices_normal(triangles, vertices, normalize=False, area_weighted=area_weighted)
    curvature_sign = -np.sign(dot(vts_dir, vts_normal))
    curvature = length(vts_dir)
    
    return curvature*curvature_sign


def vertices_cotan_direction(triangles, vertices, normalize=True, area_weighted=True):
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
    cotan_normal = curvature_normal_mtx.dot(vertices)

    if normalize:
        return normalize_vectors(cotan_normal)
    return cotan_normal


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


def vertices_cotan_normal(triangles, vertices, normalize=True, area_weighted=True):
    if scipy.sparse.__name__ in type(vertices).__module__:
        vertices = vertices.toarray()

    cotan_normal = vertices_cotan_direction(triangles, vertices, normalize=False, area_weighted=area_weighted)
    vts_normal = vertices_normal(triangles, vertices, normalize=False, area_weighted=area_weighted)
    # inverse inverted cotan_normal direction
    cotan_normal = np.sign(dot(cotan_normal, vts_normal, keepdims=True)) * cotan_normal

    if normalize:
        return normalize_vectors(cotan_normal)
    return cotan_normal

# Mesh smoothing (flow)
# and maybe it should be in another type of mesh, test all the rest before testing this
def laplacian_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=None):

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    vertices_csc = csc_matrix(vertices)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()

        if l2_dist_weighted:
            # if l2_dist_weighted, we need to compute laplacian_matrix each iteration (because ||e_ij|| change)
            # A_ij_l2_dist_weighted = A_ij / ||e_ij||
            adjacency_matrix = edge_length(triangles, vertices_csc)
            ####################################################################
            # adjacency_matrix.data **= -1
            # laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=False)
            ####################################################################
            adjacency_matrix.data **= 1  # 1
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)
        else:
            adjacency_matrix = edge_adjacency(triangles, vertices_csc)
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)

        if area_weighted:
            vts_mix_area = vertices_mix_area(triangles, vertices_csc)
            # laplacian_matrix = diags((vts_mix_area / vts_mix_area.min()) ** -1, 0).dot(laplacian_matrix)
            laplacian_matrix = diags(vts_mix_area ** -1, 0).dot(laplacian_matrix)

        next_vertices_csc = euler_step(laplacian_matrix, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return next_vertices_csc.toarray()


def curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    vertices_csc = csc_matrix(vertices)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()

        # get curvature_normal_matrix
        curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)

        next_vertices_csc = euler_step(curvature_normal_mtx, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return vertices_csc.toarray()


# positive constrain curvature
def positive_curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))
        
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, csc_matrix(vertices), diffusion_step, backward_step).toarray()
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_normal(triangles, next_vertices, normalize=False)
        pos_curv = dot(direction, normal_dir, keepdims=True) < 0
        vertices += direction * pos_curv

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices


# positive weighted constrain curvature
def volume_curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
        
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    # curvature_normal_mtx_start = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)
    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices
        # get curvature_normal_matrix
        # todo not optimal, because operation done twice etc
        curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, csc_matrix(vertices), diffusion_step, backward_step).toarray()
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_cotan_normal(triangles, vertices, normalize=True)
        dotv = dot(normalize_vectors(direction), normal_dir, keepdims=True)
        vertices += direction * np.maximum(0.0, -dotv)

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices


def mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
    
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()
        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)

        """ # forward step: Dy = (D+d*L)x = b
        b_matrix = (mass_mtx + diags(diffusion_step,0).dot(curvature_normal_mtx)).dot(vertices_csc)
        vertices_csc = spsolve(mass_mtx, b_matrix) - vertices_csc
        """
        
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - (diags(diffusion_step,0).dot(curvature_normal_mtx))
        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)
        
        

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return vertices_csc.toarray()


# positive mass_stiffness
def positive_mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.todense()
        
        # third try
        mass_mtx = mass_matrix(triangles, vertices_csc)
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)
        
        # Dy = (D+d*L)x = b
        #b_matrix = (mass_mtx + diags(diffusion_step,0).dot(curvature_normal_mtx)).dot(vertices_csc)
        #y = spsolve(mass_mtx, b_matrix)
        #pos_curv = dot(y - vertices_csc, vertices_normal(triangles, vertices, False, False)) < 0
        
        pos_curv = vertices_cotan_curvature(triangles, vertices_csc, False) > - G_ATOL
        
        # Gaussian fix
        GAUSSIAN_TRESHOLD = 0.2 # max_gauss: PI, cube corner = PI/2 # = 0.8
        deg_vts = np.abs(vertices_gaussian_curvature(triangles, vertices_csc, False)) > GAUSSIAN_TRESHOLD
        pos_curv = np.logical_or(pos_curv, deg_vts)
        
        # Sheet fix
        ANGLE_TRESHOLD = 1.0 # max_gauss: PI, cube corner = PI/2 # = 1.7
        deg_seg = edge_triangle_normal_angle(triangles, vertices_csc).max(1).toarray().squeeze() > ANGLE_TRESHOLD
        pos_curv = np.logical_or(pos_curv, deg_seg)
        #print " ", np.count_nonzero(deg_vts), " ", np.count_nonzero(deg_seg)
        
        possitive_diffusion_step = pos_curv * diffusion_step
        
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - (diags(possitive_diffusion_step,0).dot(curvature_normal_mtx))
        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)
        
        
    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices_csc.toarray()


# positive weighted mass_stiffness
def volume_mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=False)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
    
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()
        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices)
        
        raise NotImplementedError()
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - diags(diffusion_step,0).dot(curvature_normal_mtx)
        b_matrix = mass_mtx.dot(csc_matrix(vertices_csc))
        next_vertices = spsolve(A_matrix, b_matrix)
        # test if direction is positive
        direction = next_vertices.toarray() - vertices_csc
        normal_dir = vertices_cotan_normal(triangles, next_vertices, normalize=True)
        dotv = normalize_vectors(direction).multiply(normal_dir)
        vertices_csc += direction * np.maximum(0.0, -dotv)
        # vertices_csc += direction * sigmoid(-np.arctan(dotv)*np.pi - np.pi)
        # vertices_csc += direction * softplus(-dotv)
        
    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices_csc.toarray()
    
    
def gaussian_curv_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    raise NotImplementedError()


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

    #assert(np.allclose(laplacian_mtx.sum(1), 0.0, atol=G_ATOL))
    if not np.allclose(laplacian_mtx.sum(1), 0.0, atol=G_ATOL):
        print "WARNING :: laplacian"
    return laplacian_mtx


#  stiffness_matrix
def mean_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    ctn_ab_angle = cotan_alpha_beta_angle(triangles, vertices)

    if area_weighted:
        # Kn(xi) = 0.5/Ai * Sum_j( ctn_ab[i,j](xj-xi) )
        # Kn(xi) = 0.5/Ai * Sum_j(ctn_ab[i,j]*xj) - xi*Sum_j(ctn_ab[i,j)
        # Kn(xi) = 0.5/Ai * L(ctn_ab[i,j])*X
        lap_ctn_ab_angle = laplacian(ctn_ab_angle, diag_of_1=False)
        vts_mix_area = vertices_mix_area(triangles, vertices)
        curvature_normal_mtx = diags(vts_mix_area.min() / vts_mix_area, 0).dot(lap_ctn_ab_angle)
        # curvature_normal_mtx = diags( 0.5 / vts_mix_area, 0).dot(lap_ctn_ab_angle)
    else:
        curvature_normal_mtx = laplacian(ctn_ab_angle, diag_of_1=True)

    #assert(np.allclose(curvature_normal_mtx.sum(1), 0.0, atol=G_ATOL))
    if not np.allclose(curvature_normal_mtx.sum(1), 0.0, atol=G_ATOL):
        print "WARNING :: mean_curvature_normal_matrix"
    return curvature_normal_mtx


def positive_curvature_normal_matrix(triangles, vertices, area_weighted=False):
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted)
    # get flow vector and normal vector
    direction = curvature_normal_mtx.dot(vertices)
    normal_dir = vertices_normal(triangles, vertices, normalize=False)
    # positive curvature goes opposite to the normal
    pos_curv = (dot(direction, normal_dir, axis=1) < 0).astype(G_DTYPE)
    curvature_normal_mtx = diags(pos_curv, 0).dot(curvature_normal_mtx)
    return curvature_normal_mtx


def mass_matrix(triangles, vertices):
    tri_area = triangles_area(triangles, vertices)
    e_area = edge_triangle_map(triangles, vertices) 
    e_area.data = tri_area[e_area.data]
    # e_area = edge_mix_area(triangles, vertices)  # with vertices mix area edge_voronoi_area
    e_area = (e_area + e_area.T) / 12.0
    weights = np.squeeze(np.array(e_area.sum(1)))
    mass_mtx = e_area + diags(weights, 0)
    return mass_mtx


# ###################################################################
# Generic Functions
def square_length(vectors, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    return np.sum(np.square(vectors), axis, keepdims=keepdims)


def length(vectors, axis=1, keepdims=False):
    return np.sqrt(square_length(vectors, axis, keepdims))


def normalize_vectors(vectors, axis=1):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    return vectors / length(vectors, axis, True)


def dot(vectors1, vectors2, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors1).__module__:
        vectors1 = vectors1.toarray()
    if scipy.sparse.__name__ in type(vectors2).__module__:
        vectors2 = vectors2.toarray()
    return np.sum(vectors1 * vectors2, axis, keepdims=keepdims)


# Step method
def euler_step(D_matrix, b_matrix, diffusion_step, backward_step=False):
    if backward_step:
        return backward_euler_step(D_matrix, b_matrix, diffusion_step)
    else:
        return forward_euler_step(D_matrix, b_matrix, diffusion_step)


# matrix need to be csc_matrix (not float128)
def forward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : x = ( I + d*D )b  <=> x = Af*b
    Af_matrix = csc_matrix(identity(b_matrix.shape[0])) + diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = Af_matrix.dot(b_matrix)
    return x_matrix


# matrix need to be csc_matrix (not float128)
def backward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : ( I - d*D )x = b  <=> Ab*x = b
    Ab_matrix = csc_matrix(identity(b_matrix.shape[0])) - diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = spsolve(Ab_matrix, b_matrix)
    return x_matrix

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))

def softplus(values):
    return np.log(1.0 + np.exp(values))
