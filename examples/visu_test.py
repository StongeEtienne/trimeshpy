# by Etienne St-Onge 

import logging
import trimeshpy
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk

# Test files
file_name = trimeshpy.data.brain_lh
logging.basicConfig(level=logging.DEBUG)

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
#points = tri_mesh_flow.laplacian_smooth(nb_step, diffusion_step, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=saved_flow)
#points = tri_mesh_flow.curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
#points = tri_mesh_flow.positive_curvature_normal_smooth(nb_step, diffusion_step, area_weighted=True, backward_step=True, flow_file=saved_flow)
points = tri_mesh_flow.mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.positive_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)
#points = tri_mesh_flow.volume_mass_stiffness_smooth(nb_step, diffusion_step, flow_file=saved_flow)

tri_mesh_flow.set_vertices_flow_from_hdf5(saved_flow)
tri_mesh_flow.display(display_name="Trimeshpy: Flow resulting surface")
tri_mesh_flow.display_vertices_flow(display_name="Trimeshpy: Flow visualization")

