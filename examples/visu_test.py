#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import trimeshpy
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk


def main():
    # Test files
    file_name = trimeshpy.data.brain_lh
    logging.basicConfig(level=logging.DEBUG)

    # Load VTK surface
    mesh = TriMesh_Vtk(file_name, None)

    # Get triangles and vertices
    triangles = mesh.get_triangles()
    vertices = mesh.get_vertices()

    # Display initial surface
    mesh.display(display_name="Trimeshpy: Initial Mesh")

    # pre-smooth
    vertices = mesh.laplacian_smooth(2, 10.0, l2_dist_weighted=False, area_weighted=False, backward_step=True, flow_file=None)

    # Update vertices from smoothed result
    mesh.set_vertices(vertices)

    # Display smoothed surface
    mesh.display(display_name="Trimeshpy: Smoothed Mesh")

    # Execute Surface Flow (mass_stiffness_smooth)
    vertices = mesh.mass_stiffness_smooth(5, 10.0, flow_file=trimeshpy.data.output_test_flow)

    # Update flow vertices from output file
    tri_mesh_flow = TriMeshFlow_Vtk(triangles, vertices)
    tri_mesh_flow.set_vertices_flow_from_hdf5(trimeshpy.data.output_test_flow)

    # Display resulting surface and flow lines
    tri_mesh_flow.display(display_name="Trimeshpy: Flow resulting surface")
    tri_mesh_flow.display_vertices_flow(display_name="Trimeshpy: Flow visualization")


if __name__ == "__main__":
    main()
