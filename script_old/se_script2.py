# Etienne.St-Onge@usherbrooke.ca

import se_functions as fsf
import numpy as np

import json, argparse

from trimeshpy.trimesh_vtk import TriMesh_Vtk

# python se_script2.py pft/rhwhite.vtk pft/lhwhite.vtk pft/mask.nii.gz pft/fodf.nii.gz pft/testtrack.fib

parser = argparse.ArgumentParser(description='FS Surface tracking script,\n vtk surface and .fib are in world coordinate.\n\n ***Freesurfer script need to be in current root***')

parser.add_argument('rh_surface', type=str, default=None, help='input right surface')
parser.add_argument('lh_surface', type=str, default=None, help='input left surface')
parser.add_argument('mask', type=str, default=None, help='tractography mask (with nii transfo)')
parser.add_argument('fodf', type=str, default=None, help='tractography fodf')
parser.add_argument('tracking', type=str, default=None, help='tracking output (.fib)')

parser.add_argument('-json', type=str, default=None, help='tracking output (.fib)')

args = parser.parse_args()

if args.json is not None:
    json_file = args.json
else:
    json_file = "se_script_config.json"
    
with open(json_file) as data_file:    
    config = json.load(data_file)

rh_mesh = TriMesh_Vtk(args.rh_surface, None)
lh_mesh = TriMesh_Vtk(args.lh_surface, None)

run = config["run"]
if run["translate"]:
    t = fsf.get_xras_translation(args.mask)
    rh_mesh.set_vertices(rh_mesh.vertices_translation(t))
    lh_mesh.set_vertices(lh_mesh.vertices_translation(t))
    print "xras translation", t

if run["space"]:
    if config["space"]["auto"]:
        vox = fsf.get_nifti_voxel_space(args.mask)
        world = fsf.get_nifti_voxel_space(args.mask)
    else: 
        vox = config["space"]["vox"]
        world = config["space"]["world"]
        
    rh_mesh = fsf.vox_to_world(rh_mesh, vox, world)
    lh_mesh = fsf.vox_to_world(lh_mesh, vox, world)
    
    
if run["space"]:
    if config["space"]["auto"]:
        vox = fsf.get_nifti_voxel_space(args.mask)
        world = fsf.get_nifti_voxel_space(args.mask)
    else: 
        vox = config["space"]["vox"]
        world = config["space"]["world"]
        
    rh_mesh = fsf.vox_to_world(rh_mesh, vox, world)
    lh_mesh = fsf.vox_to_world(lh_mesh, vox, world)
    
if run["smooth"]:
    #todo
    rh_mesh = rh_mesh.laplacian_smooth(
                nb_iter=args.nb_step, 
                diffusion_step=args.step_size, 
                l2_dist_weighted=args.dist_weighted, 
                area_weighted=args.area_weighted, 
                backward_step=not(args.forward_step), 
                flow_file=None)
    
    lh_mesh = lh_mesh.laplacian_smooth(
                nb_iter=args.nb_step, 
                diffusion_step=args.step_size, 
                l2_dist_weighted=args.dist_weighted, 
                area_weighted=args.area_weighted, 
                backward_step=not(args.forward_step), 
                flow_file=None)
    

    