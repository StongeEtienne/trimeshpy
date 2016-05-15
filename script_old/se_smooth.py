#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk

parser = argparse.ArgumentParser(description='Smooth surface before SE tractography')
parser.add_argument('surface', type=str, default=None, help='input surface')
parser.add_argument('smooth_surface', type=str, default=None, help='output smoothed surface file name')

#visu
parser.add_argument('--vi', action='store_true', default=False, help='view input surface')
parser.add_argument('--vo', action='store_true', default=False, help='view output surface')
#parser.add_argument('--vf', action='store_true', default=False, help='view surface flow')

#smooth option
parser.add_argument('-nb_step', type=int, default=2, help='number of step for the smoothing')
parser.add_argument('-step_size', type=float, default=5.0, help='smoothing step size')
parser.add_argument('--dist_weighted', action='store_true', default=False, help='use edge distance to weight the smoothing')
parser.add_argument('--area_weighted', action='store_true', default=False, help='use triangle area to weight the smoothing')
parser.add_argument('--forward_step', action='store_true', default=False, help='use forward_step, faster but might diverge if step>1, or if any weighting is used')

# mask
parser.add_argument('-mask' , type=str, default=None, help='mask smooth')

# Parse input arguments
args = parser.parse_args()

surface_file_in = args.surface

if args.smooth_surface is None:
    dot_index = surface_file_in.rfind(".")
    surface_file_out = surface_file_in[:dot_index] + "_smooth" + surface_file_in[dot_index:]
else:
    surface_file_out = args.smooth_surface


# Option output
param_str = "parameters: "
if args.dist_weighted: param_str += "dist_weighted, "
if args.area_weighted: param_str += "area_weighted, "
if args.forward_step: param_str += "forward_step, "

print args.nb_step, "step of size ", args.step_size
print param_str

# mask step size
if args.mask is not None:
    mask = np.load(args.mask)
    step_size = args.step_size * mask
else:
    step_size = args.step_size

mesh = TriMesh_Vtk(surface_file_in, None)
smooth_vertices = mesh.laplacian_smooth(
                    nb_iter=args.nb_step, 
                    diffusion_step=step_size, 
                    l2_dist_weighted=args.dist_weighted, 
                    area_weighted=args.area_weighted, 
                    backward_step=not(args.forward_step), 
                    flow_file=None)


smoothed_mesh = TriMesh_Vtk(mesh.get_triangles(), smooth_vertices)


# Save
smoothed_mesh.save(surface_file_out)

# Display
if args.vi:
    mesh.display('input surface')
    
if args.vo:
    smoothed_mesh.display('output surface')

#print "\n!!! Smooth surface Saved  !!!\n"
