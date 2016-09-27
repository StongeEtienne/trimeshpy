#!/usr/bin/env python

import argparse
import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk

parser = argparse.ArgumentParser(description='Surface transformation from RAS to LPS')
parser.add_argument('input', type=str, default=None, help='input RAS surface file name')
parser.add_argument('output', type=str, default=None, help='output LPS surface file name')


#args = parser.parse_args("brain_mesh/rhwhiteRAS.vtk brain_mesh/t1_PA000002.nii.gz".split())
args = parser.parse_args()
surface_file_in = args.input
surface_file_out = args.output

mesh = TriMesh_Vtk(surface_file_in, None)
    
mesh.update_polydata()
mesh.save(surface_file_out)

