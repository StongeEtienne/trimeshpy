#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import numpy as np
import argparse
import nibabel as nib
from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.vtk_util import lines_to_vtk_polydata,  save_polydata, load_streamlines_poyldata, get_streamlines


from dipy.viz import fvtk

parser = argparse.ArgumentParser(description='FS Surface tractography')
parser.add_argument('volume', type=str, default=None, help='input volume or mask')
parser.add_argument('fibers', type=str, nargs='+', default=None, help='tractography fibers (.fib)')

parser.add_argument('-o', type=str, default=None, help='output volume dir')

args = parser.parse_args()


# load volume
mask_file = args.volume
volume_nib = nib.load(mask_file)
    
# load tracto
init_streamlines_list = []
for filename in args.fibers: 
    init_streamlines_list.append(get_streamlines(load_streamlines_poyldata(filename)))
    print filename, len(init_streamlines_list[-1])

# Transform tracto to Voxel space
rotation = volume_nib.get_affine()[:3,:3]
inv_rotation = np.linalg.inv(rotation)
translation = volume_nib.get_affine()[:3,3]
scale = np.array(volume_nib.get_header().get_zooms())
voxel_space = nib.aff2axcodes(volume_nib.get_affine())

print voxel_space
# seed points transfo
# LPS -> voxel_space
vertices_list = []
dirs_list = []
for streamlines in init_streamlines_list:
    for streamline in streamlines:
        if voxel_space[0] != 'L':
            #print "flip X"
            streamline[:,0] = -streamline[:,0]
        if voxel_space[1] != 'P':
            #print "flip Y"
            streamline[:,1] = -streamline[:,1]
        if voxel_space[2] != 'S':
            #print "flip Z"
            streamline[:,2] = -streamline[:,2]

        # other transfo
        streamline = streamline - translation
        streamline = streamline.dot(inv_rotation)
        streamline = streamline * scale #(if voxmm)
        
        vertices = 0.5*(streamline[1:] + streamline[:-1])
        dirs = streamline[:-1] - streamline[1:]
        
        vertices_list.append(vertices)
        dirs_list.append(dirs)


vertices_array = np.vstack(vertices_list)
dirs_array = np.vstack(dirs_list)
print vertices_array.shape, dirs_array.shape
print vertices_array.min(0), vertices_array.max(0)


# Change tracto to list of Vertices with directions
#for streamline in init_streamlines_list:


# Create a VTK -KdTree


# For each voxel find a list of vertices


# average each direction ( average / other method )


# save volume with "RGB

# Save volume with Peaks

