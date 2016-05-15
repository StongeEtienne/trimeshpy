# Etienne.St-Onge@usherbrooke.ca

# import all MATH sub modules
# and general libraries

from sys import stdout

import numpy as np

import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, diags, identity


G_DTYPE = np.float64
G_ATOL = 1e-8

### local import
from util import *
from map import *
from transfo import *
from angle import *
from area import *
from normal import *
from curvature import *
from matrix import *
from smooth import *

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
