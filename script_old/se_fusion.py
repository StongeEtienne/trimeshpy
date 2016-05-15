#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca

import argparse
from sys import stdout
import numpy as np
from dipy.tracking.benchmarks.bench_streamline import compress_streamlines

from trimeshpy.trimesh_vtk import TriMesh_Vtk, save_polydata
from trimeshpy.trimesh_vtk import load_streamlines_poyldata, get_streamlines
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.math import length

parser = argparse.ArgumentParser(description='SE Surface tractography')
# need a better description
parser.add_argument('tracto_cut', type=str, default=None, help='tracto input .fib')
parser.add_argument('rh_surface', type=str, default=None, help='right surface')
parser.add_argument('lh_surface', type=str, default=None, help='left surface')
parser.add_argument('rh_flow_file', type=str, default=None, help='right surface flow input .dat')
parser.add_argument('lh_flow_file', type=str, default=None, help='left surface flow input .dat')
parser.add_argument('rh_flow_info', type=str, default=None, help='right surface flow info .npy')
parser.add_argument('lh_flow_info', type=str, default=None, help='left surface flow info .npy')
parser.add_argument('intersection', type=str, default=None, help='input intersection file')
parser.add_argument('surf_idx', type=str, default=None, help='surface index intersection file')
parser.add_argument('output', type=str, default=None, help='tracking result .fib format')


parser.add_argument('-max_nb_step', type=int, default=100, help='nb step for surface tracking interpolation')
parser.add_argument('-compression', type=float, default=None, help='compression toll (no-compression by default, but 0.01 is dipy default)')

args = parser.parse_args()

### weight correctly the ending triangle !

print " loading files!"
intersect = np.load(args.intersection)
surf_id = np.load(args.surf_idx)

tracto_polydata = load_streamlines_poyldata(args.tracto_cut)
tracto = get_streamlines(tracto_polydata)

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
                 shape=(rh_info[0], lh_vts.shape[0], lh_vts.shape[1]))

vts = (rh_vts, lh_vts)
tris = (rh_tris, lh_tris)
flow = (rh_flow, lh_flow)



if args.max_nb_step > rh_info[0]:
    flow_step = 1
else:
    flow_step = rh_info[0]//args.max_nb_step

# MASK
"""
masking = False
if args.rh_mask is not None and args.lh_mask is not None:
    masking = True
    rh_mask = np.load(args.rh_mask)
    lh_mask = np.load(args.lh_mask)
    mask = (rh_mask, lh_mask)"""


print rh_flow.shape, lh_flow.shape, flow_step

loop_len = len(intersect)
print "Fusion of", loop_len ,"lines !"
streamlines = []
for i in range(loop_len):
    stdout.write("\r %d%%" % (i*101//loop_len))
    stdout.flush()
    line_idx = i
    vts_idx = intersect[i,0]
    tri_idx = intersect[i,1]
    
    start_surf_id = surf_id[i, 0]
    end_surf_id= surf_id[i, 1]
    if vts_idx != -1 and tri_idx != -1:
        # tract_a = surface tracking initial streamline
        if start_surf_id > 1:
            tract_a_b = tracto[line_idx]
        else:
            tract_a_b = np.concatenate((flow[start_surf_id][::flow_step,vts_idx], tracto[line_idx]))
        
        if end_surf_id > 1:
            tract_a_b_c = tract_a_b
        else:
            pt_end = tract_a_b[-1]
            
            tri_pt_idx = tris[end_surf_id][tri_idx]
            tri_pt = vts[end_surf_id][tri_pt_idx]
            
            # triangle barycenter weight 
            dists = length(tri_pt - pt_end)
            weights = dists**-1 / np.sum(dists**-1)
            
            flow_tri = flow[end_surf_id][::flow_step,tri_pt_idx]
            
            # tract_c = surface tracking ending streamline
            #   interpolation of each points flow of the triangle vertices
            tract_c = np.sum(flow_tri*np.reshape(weights,(-1,1)),axis=1)
            tract_c = tract_c[::-1]
            
            #merge to make the full streamline
            tract_a_b_c = np.concatenate((tract_a_b, tract_c))
        streamlines.append(tract_a_b_c)
stdout.write("\n done")
stdout.flush()

if args.compression is not None:
    streamlines = compress_streamlines(streamlines, tol_error=args.compression)
   
print "save as .fib" 
lines_polydata = lines_to_vtk_polydata(streamlines, None, np.float32)
save_polydata(lines_polydata, args.output, True)
        
    
