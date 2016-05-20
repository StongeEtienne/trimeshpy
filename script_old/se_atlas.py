#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
import numpy as np
from nibabel.freesurfer.io import read_annot

from trimeshpy.trimesh_vtk import TriMesh_Vtk

parser = argparse.ArgumentParser(description='Surface Enhanced tractography')
parser.add_argument('surface', type=str, default=None, help='input surface')
parser.add_argument('annot', type=str, default=None, help='input annot')

parser.add_argument('-index', type=int, nargs='+', default=None, help='color only the selected label')

parser.add_argument('-out_surface', type=str, default=None, help='output surface (.vtk)')
parser.add_argument('-out_vts_mask', type=str, default=None, help='output mask (npy array) from vts index')
parser.add_argument('--inverse_mask', action='store_true', default=False, help='inverse output mask')

parser.add_argument('--v', action='store_true', default=False, help='view surface')
parser.add_argument('--white', action='store_true', default=False, help='color white all label')
parser.add_argument('--info', action='store_true', default=False, help='view surface')

args = parser.parse_args()
print args.annot

mesh = TriMesh_Vtk(args.surface, None)
[vts_label, label_color, label_name] = read_annot(args.annot)

colors = label_color[:,:3]
vts_color =  colors[vts_label]

if args.info:
    for index in range(len(label_name)):
        print index, ": ", label_color[index], label_name[index]
    
#filter
if args.index is not None:
    print args.index
    mask = np.zeros([len(vts_label)], dtype=np.bool )
    for index in args.index:
        if index == -1:
            print "selected region :", index, "None" 
        else:
            print "selected region :", index, label_name[index]
            
        mask = np.logical_or(mask, (vts_label == index))
    if args.inverse_mask:
        mask = ~mask
    vts_color[~mask] = [0,0,0]
    
    if args.white:
        vts_color[mask] = [255,255,255]

mesh.update_polydata()
mesh.set_colors(vts_color)

if args.v:
    mesh.display()

if args.out_surface is not None:
    mesh.save(args.o)
    
if args.out_vts_mask is not None:
    np.save(args.out_vts_mask, mask)
