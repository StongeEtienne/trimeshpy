# by Etienne.St-Onge@usherbrooke.ca

import numpy as np
from dipy.viz import fvtk
import time
import scipy.sparse
from scipy.sparse import csc_matrix, diags, identity


from trimeshpy.trimesh_vtk import TriMesh_Vtk, save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk

import trimeshpy.math as processing

# Test files

file_name = "test_mesh/cube_simple.obj"
#file_name = "test_mesh/sphere.obj"
#file_name = "test_mesh/torus.obj"
#file_name = "test_mesh/spot.obj"
#file_name = "test_mesh/bunny.obj"
#ile_name = "brain_mesh/lh_sim_tri.obj"
#file_name = "brain_mesh/lh_sim_tri_tau20_l3l10.ply"
#file_name = "Raihaan/civet_obj_stl/stosurf_S1-A1_T1_white_surface_rsl_right_calibrated_81920.stl"
#file_name = "brain_mesh/white_rsl_smooth_test.ply"
#file_name = "surf_vtk/lhwhite_fsT.vtk"
#file_name = "brain_mesh/rhwhiteRAS_LPS_smooth.vtk"
#file_name = "brain_mesh/lh_white.vtk"
#file_name = "s1a1/lh_white.vtk"
#file_name = "brain_mesh/prefix_100307_white_surface.obj"
#file_name = "brain_mesh/prefix_100307_mid_surface.obj"
#file_name = "hcp/t1000_01/hcp_test_smooth_rh.vtk"

mesh = TriMesh_Vtk(file_name, None)
triangles = mesh.get_triangles()
vertices = mesh.get_vertices()
mesh.display()

"""
from scipy.spatial import Delaunay

# Triangulate parameter space to determine the triangles
#tri = mtri.Triangulation(u, v)
tri = Delaunay(vertices)
 

delaunay_triangles = np.vstack((tri.simplices[:,0:3], np.roll(tri.simplices,2,axis=1)[:,0:3]))
mesh.set_triangles(tri.simplices[:,0:3]) 
mesh.display()
"""

"""
#pre-smooth
vertices = mesh.laplacian_smooth(2, 5.0, l2_dist_weighted=False, area_weighted=False, backward_step=True, flow_file=None)
mesh.set_vertices(vertices)
#mesh.display()

# test colors curvature
test_curv = mesh.vertices_cotan_curvature(False)
color_curv = np.zeros([len(test_curv),3])
max_curv_color = 10000
color_curv[:,0] = np.maximum(-test_curv,0) * max_curv_color / np.abs(test_curv).max()
color_curv[:,2] = np.maximum(test_curv,0) * max_curv_color / np.abs(test_curv).max()
mesh.set_colors(color_curv)
mesh.display()
exit()
"""

tri_mesh_flow = TriMeshFlow_Vtk(triangles, vertices)
print tri_mesh_flow.get_nb_vertices(), tri_mesh_flow.get_nb_triangles()

# Test parameters
nb_step = 10
diffusion_step = 1#
saved_flow = "testflow.dat"  # None
saved_fib = "testflow.fib"  # None

# Test functions
start = time.time()
#points = tri_mesh_flow.laplacian_smooth(nb_step, diffusion_step, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=saved_flow)
#points = tri_mesh_flow.curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
#points = tri_mesh_flow.positive_curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
#points = tri_mesh_flow.mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.positive_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.volume_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
stop = time.time()
print (stop - start)


"""
tofix = (np.abs(test) > np.pi/4).astype(np.float)
laplacian_matrix = diags(tofix, 0).dot(mesh.laplacian(mesh.edge_map(True), diag_of_1=True))
next_vertice = processing.euler_step(laplacian_matrix, points, 1, False)
mesh.set_vertices(next_vertice)
mesh.set_colors(color)
mesh.display()
"""
"""
#saved_flow = "hcp/t1000_01/hcp_test_rh.dat"  # None
#nb_step = 1000
lines = np.memmap(saved_flow, dtype=np.float64, mode='r', shape=(nb_step, vertices.shape[0], vertices.shape[1]))[::10]#[:51]
print lines.shape
tri_mesh_flow.set_vertices_flow(np.array(lines))
#tri_mesh_flow.display()
#tri_mesh_flow.display_vertices_flow()
mesh.set_vertices(lines[99])
#test = mesh.edge_triangle_normal_angle().max(1).toarray().squeeze()
#test = mesh.vertices_gaussian_curvature(False)
test = mesh.vertices_cotan_curvature(False)
print "min =", test.min(), "max =", test.max()
print test
color = np.zeros_like(lines[0])
tmax = test.max()
tmin = -test.min()
color[:,0] = np.maximum(test,0).reshape((1,-1))*255/tmax
color[:,2] = np.maximum(-test,0).reshape((1,-1))*255/tmin
mesh.set_colors(color)
mesh.display()
"""
"""
line_to_save = np.swapaxes(lines, 0, 1)
rend = fvtk.ren()
fvtk.add(rend, mesh.get_vtk_actor())
fvtk.add(rend, fvtk.line(line_to_save))
fvtk.show(rend)
"""
"""
# save fibers in .fib normal
#line_to_save = streamline.compress_streamlines(np.swapaxes(lines, 0, 1))
line_to_save = np.swapaxes(lines, 0, 1)
lines_polydata = lines_to_vtk_polydata(line_to_save, None, np.float32)
save_polydata(lines_polydata, saved_fib, True)


"""