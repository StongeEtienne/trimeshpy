# -*- coding: utf-8 -*-

from os.path import splitext

from nibabel.freesurfer.io import read_geometry

from trimeshpy.trimesh_vtk import TriMesh_Vtk


SUPPORTED_VTK_EXTENSIONS = [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]


def load_mesh_from_file(mesh_file, mesh_assert=False):
    file_extension = splitext(mesh_file)[-1].lower()
    if file_extension in SUPPORTED_VTK_EXTENSIONS:
        return load_mesh_with_vtk(mesh_file, mesh_assert=mesh_assert)
    else:
        return load_mesh_with_nibabel(mesh_file, mesh_assert=mesh_assert)


def load_mesh_with_vtk(mesh_file, mesh_assert=False):
    # Load surface with TriMeshPy, VTK supported formats
    return TriMesh_Vtk(mesh_file, None, assert_args=mesh_assert)


def load_mesh_with_nibabel(mesh_file, mesh_assert=False):
    # Load surface with Nibabel (mainly Freesurfer surface)
    [vts, tris] = read_geometry(mesh_file)
    return TriMesh_Vtk(tris, vts, assert_args=mesh_assert)


def generate_mesh(triangles, vertices, mesh_assert=False):
    return TriMesh_Vtk(triangles, vertices, assert_args=mesh_assert)
