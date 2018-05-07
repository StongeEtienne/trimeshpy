# by Etienne St-Onge 

import numpy as np
import time

import trimeshpy
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk
from trimeshpy.vtk_util import lines_to_vtk_polydata, save_polydata

# Test files
file_name = trimeshpy.data.brain_lh

mesh = TriMesh_Vtk(file_name, None)
triangles = mesh.get_triangles()
vertices = mesh.get_vertices()
mesh.display(display_name="Trimeshpy: Initial Mesh")

# pre-smooth
vertices = mesh.laplacian_smooth(2, 10.0, l2_dist_weighted=False, area_weighted=False, backward_step=True, flow_file=None)
mesh.set_vertices(vertices)
mesh.display(display_name="Trimeshpy: Smoothed Mesh")

tri_mesh_flow = TriMeshFlow_Vtk(triangles, vertices)

# Test parameters
nb_step = 10
diffusion_step = 10
saved_flow = trimeshpy.data.output_test_flow
saved_fib = trimeshpy.data.output_test_fib

# Test functions
start = time.time()
#points = tri_mesh_flow.laplacian_smooth(nb_step, diffusion_step, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=saved_flow)
#points = tri_mesh_flow.curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
#points = tri_mesh_flow.positive_curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
points = tri_mesh_flow.mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.positive_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.volume_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
stop = time.time()
print (stop - start)

lines = np.memmap(saved_flow, dtype=np.float64, mode='r', shape=(nb_step, vertices.shape[0], vertices.shape[1]))
tri_mesh_flow.set_vertices_flow(np.array(lines))
tri_mesh_flow.display(display_name="Trimeshpy: Flow resulting surface")
tri_mesh_flow.display_vertices_flow(display_name="Trimeshpy: Flow visualization")

"""
### save fibers in .fib normal
#line_to_save = streamline.compress_streamlines(np.swapaxes(lines, 0, 1))
line_to_save = np.swapaxes(lines, 0, 1)
lines_polydata = lines_to_vtk_polydata(line_to_save, None, np.float32)
save_polydata(lines_polydata, saved_fib, True)
"""
