# by Etienne St-Onge

import numpy as np

import trimeshpy
from trimeshpy.trimesh_class import TriMesh
from trimeshpy.trimesh_vtk import TriMesh_Vtk

file_name = trimeshpy.data.cube
#file_name = trimeshpy.data.sphere
#file_name = trimeshpy.data.torus

mesh = TriMesh_Vtk(file_name, None)
triangles = mesh.get_triangles()
vertices = mesh.get_vertices()

tri_mesh = TriMesh(triangles, vertices)
print tri_mesh.get_nb_triangles(), tri_mesh.get_nb_vertices()

bool_tests = True
current_test = True

"""CONNECTIVITY MATRIX"""
# test vertex connectivity
vv_matrix = tri_mesh.edge_map(False)
current_test = np.alltrue(vv_matrix.todense() == vv_matrix.T.todense())
print current_test, ": connectivity is symmetric"
bool_tests = bool_tests and current_test

# test triangles vertex connectivity
tv_matrix = tri_mesh.triangle_vertex_map()
nb_triangles_per_points = tri_mesh.nb_triangles_per_vertex()
current_test = np.alltrue(tv_matrix.sum(1) == 3)
print current_test, ": triangles have 3 vertices"
bool_tests = bool_tests and current_test

# test weighted vertex connectivity
w_vv_matrix = tri_mesh.edge_map(True)
current_test = np.alltrue(w_vv_matrix.todense() == w_vv_matrix.T.todense())
print current_test, ": w_connectivity is symmetric"
bool_tests = bool_tests and current_test

vv_t_matrix = tri_mesh.edge_triangle_map()

"""AREA"""
# test triangles area
t_area = tri_mesh.triangles_area()
print np.allclose(t_area, 0.5), ": cube triangle area are all 0.5"
bool_tests = bool_tests and current_test

# test points area
vts_area = tri_mesh.vertices_area()
current_test = np.allclose(vts_area, nb_triangles_per_points * 0.5)
print current_test, ": cube vertices area are all 0.5*nb_triangles"
bool_tests = bool_tests and current_test
current_test = np.allclose(vts_area, tv_matrix.T.dot(t_area.T))
print current_test, ": area with both method test"
bool_tests = bool_tests and current_test
w_vts_area = tri_mesh.vertices_area(True)
current_test = np.allclose(w_vts_area, 0.5)
print current_test, ": cube vertices area average are all 0.5"
bool_tests = bool_tests and current_test
v_vts_area = tri_mesh.vertices_voronoi_area()
current_test = np.allclose(v_vts_area, 0.75)
print current_test, ": cube vertices voronoi area average are all 0.75 (3*0.5^2) "
bool_tests = bool_tests and current_test

"""ANGLE"""

is_obtuse = tri_mesh.triangle_is_obtuse()
# test triangles angles
triangle_theta = tri_mesh.triangle_angle()
current_test = np.allclose(triangle_theta.sum(1), np.pi)
print current_test, ": triangle angles sum to 180"
bool_tests = bool_tests and current_test
test2 = tv_matrix.T.dot(triangle_theta)
current_test = np.allclose(test2.sum(1) / nb_triangles_per_points, np.pi)
print current_test, ": triangle theta sum to 180"
bool_tests = bool_tests and current_test

# test theta, alpha,gamma angles
edge_theta_angle = tri_mesh.edge_theta_angle()
edge_alpha_angle = tri_mesh.edge_alpha_angle()
edge_gamma_angle = tri_mesh.edge_gamma_angle()

# todo test gamma and all possibilities

current_test = np.allclose(edge_theta_angle.sum(1), 3 * np.pi / 2)
print current_test, ": cube theta angles sum to 3*90"
bool_tests = bool_tests and current_test
current_test = np.allclose((edge_theta_angle + edge_alpha_angle + edge_gamma_angle).data, np.pi)
print current_test, ": triangles sum( theta + alpha + gamma) = 180"
bool_tests = bool_tests and current_test
current_test = np.allclose(edge_theta_angle.T.sum(0) + (edge_alpha_angle + edge_alpha_angle.T).sum(0), nb_triangles_per_points * np.pi)
print current_test, ": vertices sum( alpha + beta(alpha.T) + theta) = nb_triangles*180"
bool_tests = bool_tests and current_test

##
vv_area = tri_mesh.edge_area()
vv_vor_area = tri_mesh.edge_voronoi_area()
vv_mix_area = tri_mesh.edge_mix_area()
current_test = np.allclose((vv_mix_area - vv_vor_area).data, 0.0)
print current_test, ": cube mix_area == voronoi_area "
bool_tests = bool_tests and current_test

vertices_vor_area = tri_mesh.vertices_voronoi_area()
current_test = np.allclose(vv_vor_area.T.sum(0) - vertices_vor_area, 0.0)
print current_test, ": sum( alpha + beta + theta) = nb_triangles*180"
bool_tests = bool_tests and current_test


"""NORMAL"""
# test normal direction
triangles_normal = tri_mesh.triangles_normal()
vertices_normal = tri_mesh.vertices_normal(True)

# Curvature Matrix test
gaussian_curvature = tri_mesh.vertices_gaussian_curvature()
current_test = np.allclose(gaussian_curvature, np.pi / 2)
print current_test, ": cube gaussian_curvature on corner is pi/2"
test = tri_mesh.mean_curvature_normal_matrix(True)


print ""
if bool_tests:
    print " ALL TEST PASSED !"
else:
    print " there's some error, fix it !"
