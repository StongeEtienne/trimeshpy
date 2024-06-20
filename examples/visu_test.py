#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import trimeshpy
from trimeshpy.data import spot, output_test_flow
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_vtk import TriMeshFlow_Vtk


def main():
    # Load VTK surface
    logging.basicConfig(level=logging.DEBUG)
    mesh = TriMesh_Vtk(spot, None)

    # Get triangles and vertices
    triangles = mesh.get_triangles()
    vertices = mesh.get_vertices()
    print(len(triangles))
    print(len(vertices))

    # Display initial surface
    mesh.display(display_name="Trimeshpy: Initial Mesh")

    # pre-smooth
    vertices = mesh.laplacian_smooth(2, 1, l2_dist_weighted=False, area_weighted=False, backward_step=True, flow_file=None)

    # Update vertices from smoothed result
    mesh.set_vertices(vertices)

    # Display smoothed surface
    mesh.display(display_name="Trimeshpy: Smoothed Mesh")

    # Execute Surface Flow (mass_stiffness_smooth)
    vertices = mesh.mass_stiffness_smooth(20, 0.01, flow_file=output_test_flow)

    # Update flow vertices from output file
    tri_mesh_flow = TriMeshFlow_Vtk(triangles, vertices)
    tri_mesh_flow.set_vertices_flow_from_hdf5(output_test_flow)

    # Display resulting surface and flow lines
    tri_mesh_flow.display(display_name="Trimeshpy: Flow resulting surface")
    tri_mesh_flow.display_vertices_flow(display_name="Trimeshpy: Flow visualization")


if __name__ == "__main__":
    main()
