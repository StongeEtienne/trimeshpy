#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import numpy as np
import argparse
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.vtk_util import lines_to_vtk_polydata,  save_polydata
#from dipy.tracking.benchmarks.bench_streamline import compress_streamlines
from dipy.viz import fvtk

parser = argparse.ArgumentParser(description='FS Surface tractography')
parser.add_argument('surface', type=str, default=None, help='input surface')
parser.add_argument('flow_file', type=str, default=None, help='tracking flow result')

# output option
parser.add_argument('-end_points', type=str, default=None, help='output points where flow stopped')
parser.add_argument('-end_normal', type=str, default=None, help='output points direction where flow stopped')
parser.add_argument('-end_surf', type=str, default=None, help='output smoothed surface file name')
parser.add_argument('-tracking', type=str, default=None, help='tracking result (VTK FORMAT)')
parser.add_argument('-info', type=str, default=None, help='info, shape step size and (npy array save)')

#flow option
parser.add_argument('-nb_step', type=int, default=5, help='number of step for the smoothing')
parser.add_argument('-step_size', type=float, default=10, help='smoothing step size')
#parser.add_argument('-add_normal_step', type=float, default=None, help='add a step (input step length) at the end in the normal direction to last point')

# mask
parser.add_argument('-mask' , type=str, default=None, help='mask smooth')

#end dir option option
parser.add_argument('--ed_not_normed', action='store_false', default=True, help='view input surface')
parser.add_argument('--ed_not_weighted', action='store_false', default=True, help='view output surface')

# visu option
parser.add_argument('--vi', action='store_true', default=False, help='view input surface')
parser.add_argument('--vo', action='store_true', default=False, help='view output surface')
parser.add_argument('--vf', action='store_true', default=False, help='view surface flow')

# Parse input arguments
args = parser.parse_args()

# option output
print args.nb_step, "step of size ", args.step_size

# mask step size
if args.mask is not None:
    mask = np.load(args.mask)
    step_size = args.step_size * mask
else:
    step_size = args.step_size

mesh = TriMesh_Vtk(args.surface, None)
end_vertices = mesh.positive_mass_stiffness_smooth(
                    nb_iter=args.nb_step,
                    diffusion_step=step_size,
                    flow_file=args.flow_file)


end_mesh = TriMesh_Vtk(mesh.get_triangles(), end_vertices)

print "saving surface tracking ..."
flow = np.memmap(args.flow_file, dtype=np.float64, mode='r', shape=(args.nb_step, end_vertices.shape[0], end_vertices.shape[1]))

"""
if args.tracking is not None:
    lines = compress_streamlines(np.swapaxes(flow, 0, 1))
    lines_polydata = lines_to_vtk_polydata(lines, None, np.float32)
    save_polydata(lines_polydata, args.tracking, True)
"""
    
# save
if args.end_points is not None:
    # save only not masked points
    if args.mask is not None:
        np.save(args.end_points, end_vertices[mask])
    else:
        np.save(args.end_points, end_vertices)
    
if args.end_normal is not None:
    end_normal = end_mesh.vertices_normal(args.ed_not_normed, args.ed_not_weighted)
    
    # save only not masked points
    if args.mask is not None:
        np.save(args.end_normal, end_normal[mask])
    else:
        np.save(args.end_normal, end_normal)
    
if args.end_surf is not None:
    end_mesh.save(args.end_surf)
    
if args.info is not None:
    info = np.array([args.nb_step, args.step_size], np.object)
    np.save(args.info, info)
    
# display 
if args.vi:
    mesh.display('input surface')

if args.vo:
    end_mesh.display('output surface')
    
if args.vf:
    lines = np.swapaxes(flow, 0, 1)
    actor = fvtk.line(lines, colors=[0.5,0,0.1])
    ren = fvtk.ren()
    fvtk.add(ren, actor)
    fvtk.show(ren)
    
#print "\n !!! tracking Saved !!!\n"
