#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
import numpy as np

from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.trimesh_vtk import save_polydata, load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='lenghts stats')
parser.add_argument('fibers', type=str, nargs='+', default=None, help='tractography fibers (.fib)')
parser.add_argument('-o', type=str, default=None, help='merged tractography (.fib)')


args = parser.parse_args()

streamlines_list = []
for filename in args.fibers: 
    streamlines_list.append(get_streamlines(load_streamlines_poyldata(filename)))
    print filename, len(streamlines_list[-1])

final_streamlines = []
for streamlines in streamlines_list:
    for line in streamlines:
        final_streamlines.append(line)
         
print args.out, len(final_streamlines)
         
lines_polydata = lines_to_vtk_polydata(final_streamlines, None, np.float32)
save_polydata(lines_polydata, args.o , True)
