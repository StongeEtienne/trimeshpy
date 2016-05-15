#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import os, argparse, tempfile
import numpy as np
import nibabel as nib

from trimeshpy.trimesh_vtk import TriMesh_Vtk

parser = argparse.ArgumentParser(description='Surface transformation from RAS to LPS')
parser.add_argument('nii', type=str, default=None, help='input nii transform file')
parser.add_argument('input', type=str, default=None, help='input RAS surface file name')
parser.add_argument('output', nargs='?', type=str, default=None, help='output LPS surface file name')

parser.add_argument('--no_lps', action='store_true', default=False, help='no LPS transformation')
parser.add_argument('--no_xras_translation', action='store_true', default=False, help='no xras translation')


# flip
parser.add_argument('--fx', action='store_true', default=False, help='flip x')
parser.add_argument('--fy', action='store_true', default=False, help='flip y')
parser.add_argument('--fz', action='store_true', default=False, help='flip z')

"""
parser.add_argument('-tx', type=float, default=None, help='x translation')
parser.add_argument('-ty', type=float, default=None, help='y translation')
parser.add_argument('-tz', type=float, default=None, help='z translation')
"""

# Parse input arguments
args = parser.parse_args()
surface_file_in = args.input
nii_name = args.nii

# if we got the output file name, else generate a "_lps" at the end
if args.output:
    surface_file_out = args.output
else:
    dot_index = surface_file_in.rfind(".")
    surface_file_out = surface_file_in[:dot_index] + "_lps" + surface_file_in[dot_index:]

# Apply transformation
mesh = TriMesh_Vtk(surface_file_in, None)
if not args.no_xras_translation:
    temp_file_name = "se_lps_temp_cras.txt"
    temp_dir = tempfile.mkdtemp()
    temp_file_dir = temp_dir + "/" +  temp_file_name
    
    # Find freesurfer transformation
    transform_command_line = "mri_info --cras " + nii_name + " --o " +  temp_file_dir
    print "run :", transform_command_line
    os.system(transform_command_line)
    
    print "read :", temp_file_dir
    open_file = open(temp_file_dir, 'r')
    center_ras = open_file.read()
    open_file.close()
    os.remove(temp_file_dir)
    os.rmdir(temp_dir)
    
    print center_ras
    translate = np.array([float(x) for x in center_ras.split()])# temp HACK
    mesh.set_vertices(mesh.vertices_translation(translate))
    print "cras", translate
    
    
if args.fx or args.fy or args.fz:
    flip = [-1 if args.fx else 1, 
            -1 if args.fy else 1,
            -1 if args.fz else 1]
    print "flip:", flip
    f_triangles, f_vertices = mesh.flip_triangle_and_vertices(flip)
    mesh.set_vertices(f_vertices)
    mesh.set_triangles(f_triangles)
    
    
volume_nib = nib.load(nii_name)
if not args.no_lps:
    voxel_space = nib.aff2axcodes(volume_nib.get_affine())
    new_vertice = mesh.get_vertices()
    # voxel_space -> LPS
    print str(voxel_space), "-to-> LPS"
    if voxel_space[0] != 'L':
        new_vertice[:,0] = -new_vertice[:,0]
    if voxel_space[1] != 'P':
        new_vertice[:,1] = -new_vertice[:,1]
    if voxel_space[2] != 'S':
        new_vertice[:,2] = -new_vertice[:,2]
    mesh.set_vertices(new_vertice)
    
"""
if args.tx is not None or args.ty is not None or args.tz is not None :
    translate2 = [0.0, 0.0, 0.0]
    if args.tx is not None:
        translate2[0] = args.tx
    if args.ty is not None:
        translate2[1] = args.ty
    if args.tz is not None:
        translate2[2] = args.tz
"""

mesh.save(surface_file_out)
#print "\n!!! LPS surface Saved !!!\n"
