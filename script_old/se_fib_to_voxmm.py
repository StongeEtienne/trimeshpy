#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca

import argparse
import nibabel as nib
import numpy as np

from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.trimesh_vtk import save_polydata, load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='Transform .fib file to voxel space mm')
parser.add_argument('tract', type=str, default=None, help='tractography input (.fib)')
parser.add_argument('mask', type=str, default=None, help='anatomy or mask for trasform .nii')

parser.add_argument('-o', type=str, default=None, help='output (.fib)')

parser.add_argument('--lps_ras', action='store_true', default=False, help='force ras')
parser.add_argument('--no_transfo', action='store_true', default=False, help='force ras')

parser.add_argument('--fx1', action='store_true', default=False, help='flip x, from voxel_space')
parser.add_argument('--fy1', action='store_true', default=False, help='flip x, from voxel_space')
parser.add_argument('--fz1', action='store_true', default=False, help='flip x, from voxel_space')
parser.add_argument('--fx2', action='store_true', default=False, help='flip x, from volume shape')
parser.add_argument('--fy2', action='store_true', default=False, help='flip x, from volume shape')
parser.add_argument('--fz2', action='store_true', default=False, help='flip x, from volume shape')


parser.add_argument('-tx', type=float, default=None, help='x translation')
parser.add_argument('-ty', type=float, default=None, help='y translation')
parser.add_argument('-tz', type=float, default=None, help='z translation')

args = parser.parse_args()

# get transform
nib_mask = nib.load(args.mask)
lines = get_streamlines(load_streamlines_poyldata(args.tract))
rotation = nib_mask.get_affine()[:3,:3]
inv_rotation = np.linalg.inv(rotation)
translation = nib_mask.get_affine()[:3,3]
scale = np.array(nib_mask.get_header().get_zooms())
voxel_space = nib.aff2axcodes(nib_mask.get_affine())

shape = nib_mask.get_data().shape
print shape

# transform 
if not args.no_transfo:
    if args.lps_ras:
        print "LPS -> RAS"
        print "Not implemented"
        raise NotImplementedError()
    else:
        print "LPS ->", voxel_space, " mm"
        for i in range(len(lines)):
            if voxel_space[0] != 'L':
                lines[i][:,0] = -lines[i][:,0]
            if voxel_space[1] != 'P':
                lines[i][:,1] = -lines[i][:,1]
            if voxel_space[2] != 'S':
                lines[i][:,2] = -lines[i][:,2]
                
            if args.fx1:
                lines[i][:,0] = -lines[i][:,0]
            if args.fy1:
                lines[i][:,1] = -lines[i][:,1]
            if args.fz1:
                lines[i][:,2] = -lines[i][:,2]
    
            lines[i] = lines[i] - translation
            lines[i] = lines[i].dot(inv_rotation)
            lines[i] = lines[i] * scale

if args.fx2:
    shift = shape[0] * scale[0]
    print shift, shape[0], scale[0]
    for i in range(len(lines)):
        lines[i][:,0] = shift - lines[i][:,0]
        
if args.fy2:
    shift = shape[1] * scale[1]
    print shift, shape[1], scale[1]
    for i in range(len(lines)):
        lines[i][:,1] = shift - lines[i][:,1]
        
if args.fz2:
    shift = shape[2] * scale[2]
    print shift, shape[2], scale[2]
    for i in range(len(lines)):
        lines[i][:,2] = shift - lines[i][:,2]
        
        
if args.tx is not None:
    for i in range(len(lines)):
        lines[i][:,0] = lines[i][:,0] + args.tx
        
if args.ty is not None:
    for i in range(len(lines)):
        lines[i][:,1] = lines[i][:,1] + args.ty
        
if args.tz is not None:
    for i in range(len(lines)):
        lines[i][:,2] = lines[i][:,2] + args.tz
            
print "save as .fib"
lines_polydata = lines_to_vtk_polydata(lines, None, np.float32)
save_polydata(lines_polydata, args.o , True)
