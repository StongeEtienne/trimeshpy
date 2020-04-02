#!/usr/bin/env python
# -*- coding: utf-8 -*-

import trimeshpy
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk


def main():
    # Init Sphere
    s_mesh = TriMesh_Vtk(trimeshpy.data.sphere, None)

    # Display sphere
    sphere_tmf = TriMeshFlow_Vtk(s_mesh.get_triangles(), s_mesh.get_vertices())
    sphere_tmf.display()
    NB_STEP_SPHERE = 100

    # Umbrella sphere
    sphere_tmf.laplacian_smooth(NB_STEP_SPHERE, 1, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()

    # L2 weighted
    sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
    sphere_tmf.laplacian_smooth(NB_STEP_SPHERE, 1, l2_dist_weighted=True, area_weighted=False, backward_step=False, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()

    # L2 weighted implicit step
    sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
    sphere_tmf.laplacian_smooth(NB_STEP_SPHERE, 1, l2_dist_weighted=True, area_weighted=False, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()

    # Umbrella area weighted implicit step
    sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
    sphere_tmf.laplacian_smooth(NB_STEP_SPHERE, 100, l2_dist_weighted=False, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()

    # Cotan
    sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
    sphere_tmf.curvature_normal_smooth(NB_STEP_SPHERE, 1, area_weighted=False, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()

    # Cotan area weighted
    sphere_tmf.set_vertices_flow(s_mesh.get_vertices())
    sphere_tmf.curvature_normal_smooth(NB_STEP_SPHERE, 20, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    sphere_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    sphere_tmf.display()
    sphere_tmf.display_vertices_flow()
    del s_mesh, sphere_tmf

    # Init Cube
    c_mesh = TriMesh_Vtk(trimeshpy.data.cube, None)
    cube_tmf = TriMeshFlow_Vtk(c_mesh.get_triangles(), c_mesh.get_vertices())
    cube_tmf.display()

    # Cotan area weighted
    NB_STEP_CUBE = 10
    cube_tmf.curvature_normal_smooth(NB_STEP_CUBE, 1, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    cube_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    cube_tmf.display()
    cube_tmf.display_vertices_flow()
    del c_mesh, cube_tmf

    # Init Torus
    t_mesh = TriMesh_Vtk(trimeshpy.data.torus, None)
    torus_tmf = TriMeshFlow_Vtk(t_mesh.get_triangles(), t_mesh.get_vertices())
    torus_tmf.display()

    # Cotan area weighted
    NB_STEP_TORUS = 50
    torus_tmf.curvature_normal_smooth(NB_STEP_TORUS, 10, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    torus_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    torus_tmf.display()
    torus_tmf.display_vertices_flow()
    del t_mesh, torus_tmf

    # Init Spot
    sp_mesh = TriMesh_Vtk(trimeshpy.data.spot, None)
    spot_tmf = TriMeshFlow_Vtk(sp_mesh.get_triangles(), sp_mesh.get_vertices())
    spot_tmf.display()

    # spot laplacian
    spot_tmf.laplacian_smooth(100, 0.001, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    spot_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    spot_tmf.display()
    spot_tmf.display_vertices_flow()

    # spot curv_flow
    spot_tmf.set_vertices_flow(sp_mesh.get_vertices())
    spot_tmf.curvature_normal_smooth(20, 1000, area_weighted=True, backward_step=True, flow_file=trimeshpy.data.output_test_flow)
    spot_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    spot_tmf.display()
    spot_tmf.display_vertices_flow()

    # spot mass_stiffness
    spot_tmf.set_vertices_flow(sp_mesh.get_vertices())
    spot_tmf.mass_stiffness_smooth(30, 0.003, flow_file=trimeshpy.data.output_test_flow)
    spot_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    spot_tmf.display()
    spot_tmf.display_vertices_flow()
    del sp_mesh, spot_tmf

    # Init Brain
    b_mesh = TriMesh_Vtk(trimeshpy.data.brain_lh, None)
    brain_tmf = TriMeshFlow_Vtk(b_mesh.get_triangles(), b_mesh.get_vertices())
    brain_tmf.display()

    brain_tmf.mass_stiffness_smooth(10, 30, flow_file=trimeshpy.data.output_test_flow)
    brain_tmf.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)
    brain_tmf.display()
    brain_tmf.display_vertices_flow()
    del b_mesh, brain_tmf


if __name__ == "__main__":
    main()
