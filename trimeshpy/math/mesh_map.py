# Etienne St-Onge

from __future__ import division

import numpy as np

from scipy.sparse import csc_matrix

from trimeshpy.math.util import square_length
from trimeshpy.math.mesh_global import G_DTYPE

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
#        right handed triangles1111111111111
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
    vv_map = csc_matrix((values, (vts_i, vts_j)),
                        shape=(vertices.shape[0], vertices.shape[0]))
    return vv_map


#    edge_sqr_length : n x n -> float (sparse connectivity matrix)
#    e[i,j] = v[i] -> v[j] = { || v[i] - v[j] ||^2, if connected }
def edge_sqr_length(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    values = square_length(vertices[vts_i] - vertices[vts_j])
    vv_map = csc_matrix((values, (vts_i, vts_j)), shape=(
        vertices.shape[0], vertices.shape[0]), dtype=G_DTYPE)
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

    vv_t_map = csc_matrix((triangles_index, (vts_i, vts_j)),
                          shape=(vertices.shape[0], vertices.shape[0]))
    return vv_t_map


#    edge_opposing_vertex : n x n -> vertex_index (int) (sparse co matrix)
#    e[i,j] = v[i],v[j] = { v[k], if v[i],v[j],v[k] triangle exist }
def edge_opposing_vertex(triangles, vertices):
    vts_i = np.hstack([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    vts_j = np.hstack([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    vts_k = np.hstack([triangles[:, 2], triangles[:, 0], triangles[:, 1]])

    vv_v_map = csc_matrix((vts_k, (vts_i, vts_j)),
                          shape=(vertices.shape[0], vertices.shape[0]))
    return vv_v_map


#    triangle_vertex_map : m x n -> bool (sparse connectivity matrix)
#    t_v[i,a] = t[a] <-> v[i] = { 1, if triangle[a] is compose of vertex[i] }
def triangle_vertex_map(triangles, vertices):
    triangles_index = np.repeat(np.arange(len(triangles)), 3)
    vertices_index = np.hstack(triangles)
    values = np.ones_like(triangles_index, dtype=np.bool)

    tv_map = csc_matrix((values, (triangles_index, vertices_index)),
                        shape=(len(triangles), vertices.shape[0]))
    return tv_map


def vertices_degree(triangles, vertices):
    tv_matrix = triangle_vertex_map(triangles, vertices)
    return np.squeeze(np.asarray(tv_matrix.sum(0)))
