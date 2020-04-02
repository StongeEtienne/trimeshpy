#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from trimeshpy import TriMesh, TriMesh_Vtk
import trimeshpy.data
from trimeshpy.math.util import (dot, dot_area, dot_angle, dot_cos_angle,
                                 dot_sin_angle, dot_cotan_angle, square_length,
                                 tensor_dot, tensor_area, tensor_cos_angle,
                                 tensor_sin_angle, tensor_cotan_angle)


class TestTriMesh(unittest.TestCase):
    def setUp(self):
        # Cube
        triangles = np.load(trimeshpy.data.cube_triangles)
        vertices = np.load(trimeshpy.data.cube_vertices)
        self.cube = TriMesh(triangles, vertices)
        self.cube_tv_matrix = self.cube.triangle_vertex_map()

        # Sphere
        self.sphere = TriMesh_Vtk(trimeshpy.data.sphere, None)
        self.sphere_tv_matrix = self.sphere.triangle_vertex_map()

        # Grid_mesh
        triangles = np.load(trimeshpy.data.planar_triangles)
        vertices = np.load(trimeshpy.data.planar_vertices2d)
        self.grid_mesh_2d = TriMesh(triangles, vertices, assert_args=False)

    # Connectivity matrix
    def test_vertex_connectivity(self):
        # Cube
        vv_matrix = self.cube.edge_map(False)
        self.is_equal(np.abs((vv_matrix - vv_matrix.T).nnz), 0)

        # Sphere
        vv_matrix = self.sphere.edge_map(False)
        self.is_equal(np.abs((vv_matrix - vv_matrix.T).nnz), 0)

    def test_triangles_vertex_connectivity(self):
        # Cube
        self.is_true(self.cube_tv_matrix.sum(1) == 3)

        # Sphere
        self.is_true(self.sphere_tv_matrix.sum(1) == 3)

    def test_weighted_vertex_connectivity(self):
        # Cube
        w_vv_matrix = self.cube.edge_map(True)
        self.is_equal(np.abs((w_vv_matrix - w_vv_matrix.T).nnz), 0)

    # Area
    def test_area(self):
        # Cube
        t_area = self.cube.triangles_area()
        vts_area = self.cube.vertices_area()
        self.is_equal(t_area, 0.5)
        self.is_equal(vts_area, self.cube.nb_triangles_per_vertex() * 0.5)
        self.is_equal(vts_area, self.cube_tv_matrix.T.dot(t_area.T))

        # Sphere
        t_area = self.sphere.triangles_area()
        vts_area = self.sphere.vertices_area()
        self.is_equal(vts_area, self.sphere_tv_matrix.T.dot(t_area.T))

    def test_edge_area(self):
        # Cube
        w_vts_area = self.cube.vertices_area(True)
        v_vts_area = self.cube.vertices_voronoi_area()
        self.is_equal(w_vts_area, 0.5)
        self.is_equal(v_vts_area, 0.75)

        vv_vor_area = self.cube.edge_voronoi_area()
        vv_mix_area = self.cube.edge_mix_area()
        self.is_equal((vv_mix_area - vv_vor_area).data, 0.0)

        # Sphere
        v_vts_area = self.sphere.vertices_voronoi_area()
        vv_vor_area = self.sphere.edge_voronoi_area()
        self.is_equal(vv_vor_area.T.sum(0) - v_vts_area, 0.0)

    # Angle
    def test_angles(self):
        # Cube
        e_theta = self.cube.edge_trigo_angle(rot=0, angle_function=dot_angle)
        self.is_equal(e_theta.sum(1), 3 * np.pi / 2)

        # Sphere
        t_angles = self.sphere.triangle_trigo_angle(angle_function=dot_angle)
        nb_tri_per_vts = self.sphere.nb_triangles_per_vertex()
        angles_to_vts = self.sphere_tv_matrix.T.dot(t_angles)
        sum_ang = angles_to_vts.sum(1) / nb_tri_per_vts

        self.is_equal(t_angles.sum(1), np.pi)
        self.is_equal(sum_ang, np.pi)

        e_theta = self.sphere.edge_trigo_angle(rot=0, angle_function=dot_angle)
        e_alpha = self.sphere.edge_trigo_angle(rot=1, angle_function=dot_angle)
        e_gamma = self.sphere.edge_trigo_angle(rot=2, angle_function=dot_angle)

        self.is_equal((e_theta + e_alpha + e_gamma).data, np.pi)
        self.is_equal(e_theta.T.sum(0) + (e_alpha + e_alpha.T).sum(0),
                           nb_tri_per_vts * np.pi)
        # angle from dot
        triangle_dot_theta = self.sphere.triangle_dot_angle()
        self.is_equal(t_angles, triangle_dot_theta)

        # Sin, cos, cotan
        t_cos = self.sphere.triangle_trigo_angle(angle_function=dot_cos_angle)
        t_sin = self.sphere.triangle_trigo_angle(angle_function=dot_sin_angle)
        t_cot = self.sphere.triangle_trigo_angle(angle_function=dot_cotan_angle)
        self.is_equal(t_cos, np.cos(t_angles))
        self.is_equal(t_sin, np.sin(t_angles))
        self.is_equal(t_cot, np.tan(t_angles)**-1)

        # cos**2 + sin**2 = 1
        self.is_equal(t_cos**2 + t_sin**2, 1.0)
        self.is_equal(t_cos/t_sin, t_cot)

        # cotan matrix
        cotan_ab_map2 = self.sphere.edge_cotan_map()

    def test_triangle_type(self):
        # Cube
        is_right = self.cube.triangle_is_right()
        self.is_true(is_right)

        # Sphere
        is_obtuse = self.sphere.triangle_is_obtuse()
        is_acute = self.sphere.triangle_is_acute()
        is_right = self.sphere.triangle_is_right()

        sum_type = (is_right.astype(np.int) + is_acute.astype(np.int)
                    + is_obtuse.astype(np.int))
        # Should have only one type
        self.is_equal(sum_type, 1)

    # Curvature
    def test_curvature(self):
        # Cube
        triangles_normal = self.cube.triangles_normal()
        vertices_normal = self.cube.vertices_normal(True)

        gaussian_curvature = self.cube.vertices_gaussian_curvature()
        mean_curvature = self.cube.mean_curvature_normal_matrix(True)

        self.is_equal(gaussian_curvature, np.pi / 2)

    # Planar
    def test_plane_structure(self):
        is_right = self.grid_mesh_2d.triangle_is_right()
        self.is_true(is_right)
        self.is_equal(self.grid_mesh_2d.triangles_area(), 0.5)

        self.grid_mesh_2d.edge_cotan_map()
        self.grid_mesh_2d.vertices_cotan_direction()

    # Planar
    def test_dist(self):
        id_3d = np.eye(3)
        u = self.sphere.get_vertices()
        v = u + 1.0
        self.is_equal(dot(u, u), square_length(u))
        self.is_equal(dot(u, v), tensor_dot(u, v, id_3d))
        self.is_equal(dot_area(u, v), tensor_area(u, v, id_3d))
        self.is_equal(dot_cos_angle(u, v), tensor_cos_angle(u, v, id_3d))
        self.is_equal(dot_sin_angle(u, v), tensor_sin_angle(u, v, id_3d))
        self.is_equal(dot_cotan_angle(u, v), tensor_cotan_angle(u, v, id_3d))

        d_factor = 5.0
        d_3d = id_3d * d_factor
        self.is_equal(square_length(u) * d_factor, tensor_dot(u, u, d_3d))
        self.is_equal(dot(u, v) * d_factor, tensor_dot(u, v, d_3d))

    # Util
    def is_equal(self, value0, value1):
        if isinstance(value0, (bool, int,)):
            self.assertTrue(value0 == value1)
        elif isinstance(value0, float):
            self.assertTrue(value0 == value1)
        elif (value0.dtype is np.bool or
              np.issubdtype(value0.dtype, np.integer)):
            self.assertTrue(np.alltrue(np.equal(value0, value1)))
        elif np.issubdtype(value0.dtype, np.floating):
            self.assertTrue(np.allclose(value0, value1))

    def is_true(self, booleans):
        self.assertTrue(np.alltrue(booleans))


if __name__ == '__main__':
    unittest.main()
