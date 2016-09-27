# Etienne.St-Onge@usherbrooke.ca
import argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import morphology

import mcubes # https://github.com/pmneila/PyMCubes

from trimeshpy.trimesh_vtk import TriMesh_Vtk
from scipy import ndimage as ndi

# example : use wmparc.a2009s.nii.gz with some aseg.stats indices
# >>> python se_vol2surf.py s1a1/mask/S1-A1_wmparc.a2009s.nii.gz --v -index 16 5001 5002  --world_lps -opening 2 -smooth 2

parser = argparse.ArgumentParser(description='FS segmented volume to surface with marching cube')
parser.add_argument('volume', type=str, default=None, help='input volume or mask')
parser.add_argument('-index', type=int, nargs='+', default=None, help='color only the selected label')
parser.add_argument('-value', type=float, default=0.5, help='threshold value')

parser.add_argument('-out_surface', type=str, default=None, help='output surface (.vtk)')

parser.add_argument('-smooth', type=float, default=None, help='smoothing size (1 implicit step)')
parser.add_argument('-erosion', type=int, default=None, help='opening iterations')
parser.add_argument('-dilation', type=int, default=None, help='closing iterations')
parser.add_argument('-opening', type=int, default=None, help='opening iterations')
parser.add_argument('-closing', type=int, default=None, help='closing iterations')

parser.add_argument('--max_label',action='store_true', default=False, help='label all group of voxel and take the biggest')

parser.add_argument('--v', action='store_true', default=False, help='view surface')
parser.add_argument('--world_lps', action='store_true', default=False, help='transfo to world_lps')
parser.add_argument('--fx', action='store_true', default=False, help='flip x')
parser.add_argument('--fy', action='store_true', default=False, help='flip y')
parser.add_argument('--fz', action='store_true', default=False, help='flip z')

# Parse input arguments
args = parser.parse_args()

# load volume
mask_file = args.volume
volume_nib = nib.load(mask_file)

# get data mask from index and volume
if args.index is not None:
    volume = volume_nib.get_data()
    mask = np.zeros_like(volume)
    for index in args.index:
        mask = np.logical_or(mask, volume == index)
else:
    mask = volume_nib.get_data()

# Basic morphology
if args.erosion is not None:
    mask = morphology.binary_erosion(mask, iterations=args.erosion)
if args.dilation is not None:
    mask = morphology.binary_dilation(mask, iterations=args.dilation)
if args.opening is not None:
    mask = morphology.binary_opening(mask, iterations=args.opening)
if args.closing is not None:
    mask = morphology.binary_closing(mask, iterations=args.closing)
    

# Label fill
if args.max_label:
    label_objects, nb_labels = ndi.label(mask)
    sizes = np.bincount(label_objects.ravel())
    sizes[0] = 0 # ingnore zero voxel
    max_label = np.argmax(sizes)
    max_mask = (label_objects == max_label)
    mask = max_mask

# Extract marching cube surface from mask
vertices, triangles = mcubes.marching_cubes(mask, args.value)

# Generate mesh
mesh = TriMesh_Vtk(triangles.astype(np.int), vertices)

# transformation
if args.world_lps:
    rotation = volume_nib.get_affine()[:3,:3]
    translation = volume_nib.get_affine()[:3,3]
    voxel_space = nib.aff2axcodes(volume_nib.get_affine())
    
    new_vertice = vertices.dot(rotation)
    new_vertice = new_vertice + translation
    # voxel_space -> LPS
    print str(voxel_space), "-to-> LPS"
    if voxel_space[0] != 'L':
        new_vertice[:,0] = -new_vertice[:,0]
    if voxel_space[1] != 'P':
        new_vertice[:,1] = -new_vertice[:,1]
    if voxel_space[2] != 'S':
        new_vertice[:,2] = -new_vertice[:,2]
    mesh.set_vertices(new_vertice)
    
if args.fx or args.fy or args.fz:
    flip = [-1 if args.fx else 1, 
            -1 if args.fy else 1,
            -1 if args.fz else 1]
    print "flip:", flip
    f_triangles, f_vertices = mesh.flip_triangle_and_vertices(flip)
    mesh.set_vertices(f_vertices)
    mesh.set_triangles(f_triangles)

# smooth
if args.smooth is not None:
    new_vertice = mesh.laplacian_smooth(1, args.smooth, l2_dist_weighted=False, area_weighted=False, backward_step=True)
    mesh.set_vertices(new_vertice)

if args.out_surface is not None:
    mesh.save(args.out_surface)
    
# view 
if args.v:
    mesh.update_polydata()
    mesh.update_normals()
    mesh.display()


