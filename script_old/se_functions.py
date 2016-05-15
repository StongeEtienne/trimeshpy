# Etienne.St-Onge@usherbrooke.ca

import numpy as np
import nibabel as nib
import os


### surface filter
def get_nifti_voxel_space(nii):
    return  nib.aff2axcodes(nii.get_affine())

def get_nifti_world_space(mesh):
    return 

def vox_to_world(mesh, vox_space, world_space):
    flip = [1,1,1]
    for i in range(3):
        if vox_space[i].lower() != world_space[i].lower():
            flip[i] = -1
            
    [triangles, vertices] = mesh.flip_triangle_and_vertices(flip)
    mesh.set_triangles(triangles)
    mesh.set_vertices(vertices)
    
    
def get_xras_translation(nii_file):
    # get FreeSurfer "mri_info --cras"
    center_ras = os.popen("mri_info --cras " + nii_file ).read()
    return np.array([float(x) for x in center_ras.split()])


