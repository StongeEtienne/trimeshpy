#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca

import argparse
import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk, save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata

from dipy.tracking.benchmarks.bench_streamline import compress_streamlines

parser = argparse.ArgumentParser(description='SE flow to streamlines .fib')
parser.add_argument('rh_surface', type=str, default=None, help='right surface')
parser.add_argument('lh_surface', type=str, default=None, help='left surface')
parser.add_argument('rh_flow_file', type=str, default=None, help='right surface flow input .dat')
parser.add_argument('lh_flow_file', type=str, default=None, help='left surface flow input .dat')
parser.add_argument('rh_flow_info', type=str, default=None, help='right surface flow info .npy')
parser.add_argument('lh_flow_info', type=str, default=None, help='left surface flow info .npy')
parser.add_argument('output', type=str, default=None, help='output .fib')

parser.add_argument('-max_nb_step', type=int, default=100, help='nb step for surface tracking interpolation')
args = parser.parse_args()

print " loading files!"
rh_mesh = TriMesh_Vtk(args.rh_surface, None)
rh_vts = rh_mesh.get_vertices()
rh_tris = rh_mesh.get_triangles()

lh_mesh = TriMesh_Vtk(args.lh_surface, None)
lh_vts = lh_mesh.get_vertices()
lh_tris = lh_mesh.get_triangles()

rh_info = np.load(args.rh_flow_info)
lh_info = np.load(args.lh_flow_info)

rh_flow = np.memmap(args.rh_flow_file, dtype=np.float64, mode='r', 
                 shape=(rh_info[0], rh_vts.shape[0], rh_vts.shape[1]))

lh_flow = np.memmap(args.lh_flow_file, dtype=np.float64, mode='r', 
                 shape=(lh_info[0], lh_vts.shape[0], lh_vts.shape[1]))


if args.max_nb_step > rh_info[0]:
    flow_step = 1
else:
    flow_step = rh_info[0]//args.max_nb_step


print "compression of ", rh_vts.shape[0] + lh_vts.shape[0], "/ ", rh_info, "-", flow_step
rh_lines = compress_streamlines(np.swapaxes(rh_flow[::flow_step], 0, 1))
lh_lines = compress_streamlines(np.swapaxes(lh_flow[::flow_step], 0, 1))
rh_lines.extend(lh_lines)

print "save ", len(rh_lines), "lines as .fib" 
lines_polydata = lines_to_vtk_polydata(rh_lines, None, np.float32)
save_polydata(lines_polydata, args.output, True)
