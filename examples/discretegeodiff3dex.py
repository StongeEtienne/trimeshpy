# by Etienne.St-Onge@usherbrooke.ca

from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk

# Init
saved_flow = "testflow.dat"

#files names
sphere_file_name = "../data/test_mesh/sphere.obj"
cube_file_name = "../data/test_mesh/cube_simple.obj"
torus_file_name = "../data/test_mesh/torus.obj"
spot_file_name = "../data/test_mesh/spot.obj"
brain_file_name = "../data/brain_mesh/100307_smooth_lh.vtk"


# Init Sphere
s_mesh = TriMesh_Vtk(sphere_file_name, None)
s_vshape0 = s_mesh.get_nb_vertices()

# Display sphere
sphere_tmf = TriMeshFlow_Vtk(s_mesh.get_triangles(), s_mesh.get_vertices())
sphere_tmf.display()

# Umbrella sphere
sphere_tmf.laplacian_smooth(100, 1, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()

# L2 weighted
sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
sphere_tmf.laplacian_smooth(100, 1, l2_dist_weighted=True, area_weighted=False, backward_step=False, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()

# L2 weighted implicit step
sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
sphere_tmf.laplacian_smooth(100, 1, l2_dist_weighted=True, area_weighted=False, backward_step=True, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()

# Umbrella area weighted implicit stp
sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
sphere_tmf.laplacian_smooth(100, 125, l2_dist_weighted=False, area_weighted=True, backward_step=True, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()

# Cotan
sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
sphere_tmf.curvature_normal_smooth(100, 1, area_weighted=False, backward_step=True, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()

# Cotan area weighted
sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
sphere_tmf.curvature_normal_smooth(100, 20, area_weighted=True, backward_step=True, flow_file=saved_flow)
sphere_tmf.set_vertices_flow_from_memmap(saved_flow, 100, s_vshape0)
sphere_tmf.display()
sphere_tmf.display_vertices_flow()


# Init Cube
c_mesh = TriMesh_Vtk(cube_file_name, None)
cube_tmf = TriMeshFlow_Vtk(c_mesh.get_triangles(), c_mesh.get_vertices())
cube_tmf.display()

cube_tmf.curvature_normal_smooth(10, 1, area_weighted=True, backward_step=True, flow_file=saved_flow)
cube_tmf.set_vertices_flow_from_memmap(saved_flow, 10, c_mesh.get_nb_vertices())
cube_tmf.display()
cube_tmf.display_vertices_flow()


# Init Torus
t_mesh = TriMesh_Vtk(torus_file_name, None)
torus_tmf = TriMeshFlow_Vtk(t_mesh.get_triangles(), t_mesh.get_vertices())
torus_tmf.display()

torus_tmf.curvature_normal_smooth(50, 10, area_weighted=True, backward_step=True, flow_file=saved_flow)
torus_tmf.set_vertices_flow_from_memmap(saved_flow, 50, t_mesh.get_nb_vertices())
torus_tmf.display()
torus_tmf.display_vertices_flow()


# Init Spot
sp_mesh = TriMesh_Vtk(spot_file_name, None)
spot_tmf = TriMeshFlow_Vtk(sp_mesh.get_triangles(), sp_mesh.get_vertices())
spot_tmf.display()

# spot laplacian
spot_tmf.laplacian_smooth(100, 0.001, area_weighted=True, backward_step=True, flow_file=saved_flow)
spot_tmf.set_vertices_flow_from_memmap(saved_flow, 100, sp_mesh.get_nb_vertices())
spot_tmf.display()
spot_tmf.display_vertices_flow()

# spot curv_flow
spot_tmf.set_vertices_flow(sp_mesh.get_vertices())
spot_tmf.curvature_normal_smooth(20, 1000, area_weighted=True, backward_step=True, flow_file=saved_flow)
spot_tmf.set_vertices_flow_from_memmap(saved_flow, 20, sp_mesh.get_nb_vertices())
spot_tmf.display()
spot_tmf.display_vertices_flow()

# spot mass_stiffness
spot_tmf.set_vertices_flow(sp_mesh.get_vertices())
spot_tmf.mass_stiffness_smooth(30, 0.003, flow_file=saved_flow)
spot_tmf.set_vertices_flow_from_memmap(saved_flow, 30, sp_mesh.get_nb_vertices())
spot_tmf.display()
spot_tmf.display_vertices_flow()


# Init Brain
b_mesh = TriMesh_Vtk(brain_file_name, None)
brain_tmf = TriMeshFlow_Vtk(b_mesh.get_triangles(), b_mesh.get_vertices())
brain_tmf.display()

brain_tmf.mass_stiffness_smooth(10, 30, flow_file=saved_flow)
brain_tmf.set_vertices_flow_from_memmap(saved_flow, 10, b_mesh.get_nb_vertices())
brain_tmf.display()
brain_tmf.display_vertices_flow()
