#!/usr/bin/env python

import nibabel as nib
import argparse
import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk

parser = argparse.ArgumentParser(description='Surface transformation from RAS to LPS')
parser.add_argument('nii', type=str, default=None, help='input nii transform file')
parser.add_argument('input', type=str, default=None, help='input RAS surface file name')
parser.add_argument('output', nargs='?', type=str, default=None, help='output LPS surface file name')

# nii transformation option
parser.add_argument('--t', action='store_true', default=False, help='use nii translation')
parser.add_argument('--r', action='store_true', default=False, help='use nii rotation')
parser.add_argument('--m', action='store_true', default=False, help='use nii midpoint translation')
parser.add_argument('--mv', action='store_true', default=False, help='use nii midpoint translation with voxel zoom')
parser.add_argument('--it', action='store_true', default=False, help='use nii inverse translation')
parser.add_argument('--ir', action='store_true', default=False, help='use nii inverse rotation')
parser.add_argument('--tr', action='store_true', default=False, help='use nii rotated translation')
parser.add_argument('--tir', action='store_true', default=False, help='use nii inverse rotated translation')
parser.add_argument('--itr', action='store_true', default=False, help='use nii rotated translation')
parser.add_argument('--itir', action='store_true', default=False, help='use nii inverse rotated inverse translation')
parser.add_argument('--im', action='store_true', default=False, help='use nii inverse midpoint translation')
parser.add_argument('--imv', action='store_true', default=False, help='use nii inverse midpoint translation with voxel zoom')
parser.add_argument('--test', action='store_true', default=False, help=' test')

# other transformation option
parser.add_argument('--RAS2LPS', action='store_true', default=False, help='output LPS surface file name')

# manual transformation option
#todo ?!

#args = parser.parse_args("brain_mesh/rhwhiteRAS.vtk brain_mesh/t1_PA000002.nii.gz".split())
args = parser.parse_args()
surface_file_in = args.input

# if we got the output file name, else generate a "_fsT" at the end
if args.output:
    surface_file_out = args.output
else:
    dot_index = surface_file_in.rfind(".")
    surface_file_out = surface_file_in[:dot_index] + "_fsT" + surface_file_in[dot_index:]

img = nib.load(args.nii)
R = img.affine[:3,:3]
T = img.affine[:3,3]
z = img.header.get_zooms()
mid_point = np.array(img.shape, dtype=np.float)/2.0

# Input assert
assert not(args.ir and args.r), "--ir or --r option"
assert not(args.it and args.t), "--it or --t option"
assert not(args.im and args.m and args.imz and args.mz ), "--m,im,mz or imz option"
assert not(args.tr and args.itr and args.tir and args.itir), "--tr,itr,tir or itir option"


mesh = TriMesh_Vtk(surface_file_in, None)
    
if args.test :
    tranfo = [-2,33,12]
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "test : " + str(tranfo)
    
#todo nii rotation before or after translation?!?!?!
if args.r :
    tranfo = R
    mesh.set_vertices(mesh.vertices_rotation(tranfo))
    print "r : " + str(tranfo)
elif args.ir :
    tranfo = R.T
    mesh.set_vertices(mesh.vertices_rotation(R))
    print "ir : " + str(tranfo)
    
# nii translation
if args.t:
    tranfo = T
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "t : " + str(tranfo)
elif args.t :
    tranfo = -T
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "it : " + str(tranfo)
elif args.tr:
    tranfo = T.dot(R)
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "tr : " + str(tranfo)
elif args.itr :
    tranfo = (-T).dot(R)
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "itr : " + str(tranfo)
elif args.tir :
    tranfo = T.dot(R.T)
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "tir : " + str(tranfo)
elif args.itir :
    tranfo = (-T).dot(R.T)
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "itir : " + str(tranfo)
    
# nii midpoint translation
if args.m :
    tranfo = mid_point
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "m : " + str(tranfo)
elif args.im :
    tranfo = -mid_point
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "imz : " + str(tranfo)
elif args.mv :
    tranfo = mid_point*z
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "mz : " + str(tranfo)
elif args.imv :
    tranfo = -mid_point*z
    mesh.set_vertices(mesh.vertices_translation(tranfo))
    print "imz : " + str(tranfo)
    

if args.RAS2LPS :
    mesh.set_vertices(mesh.vertices_flip([-1, -1, 1]))

mesh.update_polydata()
mesh.save(surface_file_out)


"""
### FreeSurfer info
out1 = os.popen("mri_info --ras2vox-tkr " + file_name ).read()
out2 = os.popen("mri_info --vox2ras " + file_name ).read()

r2v = np.array([float(x) for x in out1.split()]).reshape(4,4)
v2r = np.array([float(x) for x in out2.split()]).reshape(4,4)

transfo = np.dot(r2v, v2r)
print transfo

R = transfo[0:3,0:3]
T = transfo[0:3,3]  # - [ -0.25, -0.25, -0.25]


print "### Nii info from nibabel"
import nibabel as nib
img = nib.load(file_name)
R = img.affine[:3,:3]
T = img.affine[:3,3]
zooms = img.header.get_zooms()
mid_point = np.array(img.shape, dtype=np.float)/2.0
shape = np.array(img.shape, dtype=np.float)
size = shape*zooms
shape_mid = shape/2.0
size_mid = size/2.0

print "Affine", img.affine
print "shape", shape
print "size", size
print "shape_mid", shape_mid
print "size_mid", size_mid
"""

"""
# http://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferTrackVisTransforms
tkr1_tfm = fsbt.mri_info( op.join( tpms_dir, 'wm_volume_fraction.nii.gz' ), 'vox2ras-tkr' )
r2v_tfm = fsbt.mri_info( op.join( tpms_dir, 'wm_volume_fraction.nii.gz' ), 'ras2vox' )

tkr2_tfm = fsbt.mri_info( op.join( tpms_dir, 'wm_volume_fraction.nii.gz' ), 'ras2vox-tkr' )
v2r_tfm = fsbt.mri_info( op.join( tpms_dir, 'wm_volume_fraction.nii.gz' ), 'vox2ras' )

tfm = np.dot( tkr2_tfm, v2r_tfm )
M = tfm[0:3,0:3]
O = tfm[0:3,3]

print O
O = [ -0.25, -0.25, -0.25]

surf = op.join( working_dir, 'test.vtk' )

r = tvtk.PolyDataReader( file_name=surf )
vtk = r.output
r.update()

for i,point in enumerate( vtk.points ):
    vtk.points[i] = np.dot( M, point ) - O
    
    
w = tvtk.PolyDataWriter( file_name=op.join( working_dir, 'wm_corrected2.vtk' ), input=vtk )
w.update()
"""


