# by Etienne St-Onge 

import numpy as np
from dipy.viz import fvtk
import time
import scipy.sparse
from scipy.sparse import csc_matrix, diags, identity

import trimeshpy
import trimeshpy.math as tmath

from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk
from trimeshpy.vtk_util import lines_to_vtk_polydata, save_polydata, generate_colormap



# Test files
file_name = trimeshpy.data.spot

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
#mesh.set_colors(color_curv)
mesh.set_scalars(test_curv)
mesh.display()
exit()
"""

tri_mesh_flow = TriMeshFlow_Vtk(triangles, vertices)
print tri_mesh_flow.get_nb_vertices(), tri_mesh_flow.get_nb_triangles()

# Test parameters
nb_step = 10
diffusion_step = 0.001
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


#saved_flow = "hcp/t1000_01/hcp_test_rh.dat"  # None
#nb_step = 1000
lines = np.memmap(saved_flow, dtype=np.float64, mode='r', shape=(nb_step, vertices.shape[0], vertices.shape[1]))#[::10][:51]
print lines.shape
tri_mesh_flow.set_vertices_flow(np.array(lines))
tri_mesh_flow.display()
tri_mesh_flow.display_vertices_flow()


"""
### Render both, mesh and flow
mesh.set_vertices(points)
line_to_save = np.swapaxes(lines, 0, 1)
rend = fvtk.ren()
fvtk.add(rend, mesh.get_vtk_actor())
fvtk.add(rend, fvtk.line(line_to_save))
fvtk.show(rend)
"""

"""
### save fibers in .fib normal
#line_to_save = streamline.compress_streamlines(np.swapaxes(lines, 0, 1))
line_to_save = np.swapaxes(lines, 0, 1)
lines_polydata = lines_to_vtk_polydata(line_to_save, None, np.float32)
save_polydata(lines_polydata, saved_fib, True)
"""
