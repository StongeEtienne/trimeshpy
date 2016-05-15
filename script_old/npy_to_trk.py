# Etienne.St-Onge@usherbrooke.ca

import numpy as np
import nibabel as nib
from nibabel import trackvis
from trimesh_class import TriMesh
from trimesh_vtk import TriMesh_Vtk



# add normal direction
#file_name = "brain_mesh/lh_sim_tri_tau20_l3l10.ply"
file_name = "brain_mesh/lhwhitet.vtk"
mesh = TriMesh_Vtk(file_name, None)
vertices = mesh.get_vertices()

nb_step = 10
saved_flow = "testflow.dat"
trk_filename = "testtract_lhwhite2.trk"
lines = np.memmap(saved_flow, dtype=np.float64, mode='r', shape=(nb_step, vertices.shape[0], vertices.shape[1]))

# mesh.update()
mesh.update_normals()
normals = mesh.get_normals()


# load nifti for trk info
#subject_nii_file = "brain_mesh/s31_T1.nii.gz"
subject_nii_file = "brain_mesh/S1-A2_fa.nii.gz"
fa_file = nib.load(subject_nii_file)
img = fa_file.get_data()
img_affine = fa_file.get_affine()
img_zoom = fa_file.get_header().get_zooms()[:3]


# transform lines
#transfo = (np.array(img.shape[:3]) / 2.0).reshape((1, 1, -1))
#lines = lines + transfo

# fix lines format
#normal_length = 0.1
#epsilon = 0.05

new_lines = []
for i in range(lines.shape[1]):
    #print i
    line = lines[:, i]
    """
    normal = normals[i]
    if not(np.allclose(line[0], line[-1])):
        # new_line = [line[0]] # default line
        new_line = [line[0] + normal * normal_length] # added normal step outside
        for index in range(len(line)):
            if not(np.allclose(line[index], new_line[-1], atol=epsilon)):
                new_line += [line[index]]
        new_line = np.array(new_line)
        new_lines += [new_line]
    """
    new_lines += [np.array(line)]

del lines


hdr = nib.trackvis.empty_header()
trackvis.aff_to_hdr(img_affine, hdr, True, True)
"""
pixels_shape = [img.shape[0], img.shape[1], img.shape[2]]
hdr['voxel_size'] = img_zoom
hdr['voxel_order'] = 'LAS'
hdr['dim'] = pixels_shape
hdr['n_count'] = len(new_lines)
"""



streamlines = []
for fibre in list(new_lines):
    streamlines += [(fibre, None, None)]

nib.trackvis.write(trk_filename, streamlines, hdr, points_space='rasmm')

print "done and saved in :", trk_filename