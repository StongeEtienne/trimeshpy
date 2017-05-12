# Etienne St-Onge

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
    if np.sum(np.equal(flip, 1))%2 == 0:
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
