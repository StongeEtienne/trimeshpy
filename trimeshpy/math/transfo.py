# Etienne St-Onge

from __future__ import division

import numpy as np


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
    # rotation = [[rx1, rx2, rx3],
    #             [ry1, ry2, ry3],
    #             [rz1, rz2, rz3]]
    return np.dot(rotation, vertices.T).T


def vertices_transformation(triangles, vertices, transfo):
    # transfo = [[rx1, rx2, rx3, t_x],
    #            [ry1, ry2, ry3, t_y],
    #            [rz1, rz2, rz3, t_z],
    #            [s_x, s_y, s_z,  1 ]]
    vertices_4d = np.ones((len(vertices), 4))
    vertices_4d[:, :-1] = vertices
    return np.dot(transfo, vertices_4d.T).T[:, :3]


def vertices_affine(triangles, vertices, affine):
    # affine = [[rx1, rx2, rx3, t_x],
    #           [ry1, ry2, ry3, t_y],
    #           [rz1, rz2, rz3, t_z],
    #           [  0,   0,   0,  1 ]]
    return (np.dot(affine[:3, :3], vertices.T) + affine[:3, 3:4]).T


def vertices_flip(triangles, vertices, flip=(1, 1, 1)):
    # flip = [ f_x, f_y, f_y] : -1 to flip/ 1 else
    return vertices * flip


def triangles_face_flip(triangles, vertices, flip=(0, 2)):
    # flip triangle face "flip normal",
    # flip = [i, j] to flip i,j columns to flip
    triangles[:, flip] = triangles[:, flip[::-1]]
    return triangles


def flip_triangle_and_vertices(triangles, vertices, flip=(1, 1, 1)):
    # flip = [ f_x, f_y, f_y] : -1 to flip/ 1 else
    # and autorotate face if impair flip
    vertices = vertices * flip
    if np.prod(flip) < 0:
        triangles_face_flip(triangles, vertices)
    return triangles, vertices


def is_transformation_flip(transfo):
    return np.linalg.det(transfo) < 0
