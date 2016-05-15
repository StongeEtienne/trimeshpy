# Etienne.St-Onge@usherbrooke.ca
from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from trimeshpy.trimesh_vtk import load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='lenghts stats')
parser.add_argument('-lengths', type=str, nargs='+', default=None, help='tractography length (.npy)')
parser.add_argument('-lengths', type=str, nargs='+', default=None, help='tractography surface connection idx (.npy)')

args = parser.parse_args()

streamlines_list = []
for filename in args.tracts: 
    streamlines_list.append(get_streamlines(load_streamlines_poyldata(filename)))
    
lines_length_list = []
for lines in streamlines_list:
    lines_length = np.zeros([len(lines)], dtype=np.float)
    for i in range(len(lines)):
        dist = lines[i][:-1] - lines[i][1:]
        lines_length[i] = np.sum(np.sqrt(np.sum(np.square(dist), axis=1)))
    lines_length_list.append(lines_length)
    #print np.average(lines_length)
    #print np.var(lines_length)
    
histogram_list = []
color_list =  ['lightcoral', 'darkred', 'lightgreen', 'darkgreen', 'skyblue', 'darkblue', 'khaki', 'y']
patches =  []
fontsize=20
for i in range(len(lines_length_list)):
    lines_length = lines_length_list[i]
    min_v = 0
    max_v = 100
    step_v = 1
    group = 2
    hist, bin_edges = np.histogram(lines_length, np.arange(min_v,max_v,step_v))
    
    # normalize ?
    hist = hist.astype(np.float)/len(streamlines_list[i])
    
    bar_size = float(step_v)/(len(lines_length_list)/group+1)
    #plt.bar(bin_edges[:-1]+i//group*bar_size, hist, width=bar_size, color=color_list[i])
    plt.plot(bin_edges, np.concatenate(([0], hist)), color=color_list[i], linewidth=2.0)
    plt.xlim(min_v, max_v)
    patches.append(mpatches.Patch(color=color_list[i], label=args.tracts[i]))
    histogram_list.append(hist)
    
plt.xlabel("Streamline length (mm)", fontsize=fontsize)
plt.ylabel("Frequency % (streamlines histogram)", fontsize=fontsize)
plt.axis(fontsize=fontsize)
plt.legend(handles=patches, fontsize=fontsize)
plt.show()


