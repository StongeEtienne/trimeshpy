#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
import numpy as np

from trimeshpy.trimesh_vtk import load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='length to numpy array')
parser.add_argument('tract', type=str, default=None, help='tractography input (.fib)')
parser.add_argument('out_array', type=str, default=None, help='tractography length (.npyw)')

args = parser.parse_args()

filename = args.tract
lines = get_streamlines(load_streamlines_poyldata(args.tract))

lines_length = np.zeros([len(lines)], dtype=np.float)
for i in range(len(lines)):
    dist = lines[i][:-1] - lines[i][1:]
    lines_length[i] = np.sum(np.sqrt(np.sum(np.square(dist), axis=1)))

np.save(args.out_array, lines_length)
