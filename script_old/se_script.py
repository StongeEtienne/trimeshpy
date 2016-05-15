#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
# General surface-enhanced script tested on freesurfer 

# python se_script.py pft/rhwhite.vtk pft/lhwhite.vtk pft/mask.nii.gz pft/fodf.nii.gz pft/output/testtrack.fib
# python se_script.py pft/rhwhite.vtk pft/lhwhite.vtk pft/mask.nii.gz pft/fodf.nii.gz pft/output/testtrack.fib -include pft/other/include.nii.gz -exclude pft/other/exclude.nii.gz
# python se_script.py pft/rhwhite.vtk pft/lhwhite.vtk pft/mask.nii.gz pft/fodf.nii.gz pft/output/testtrack.fib -include pft/other/include.nii.gz -exclude pft/other/exclude.nii.gz -rh_atlas pft/label/rh.aparc.a2009s.annot  -lh_atlas pft/label/lh.aparc.a2009s.annot 
# python se_script.py pft/rhwhite.vtk pft/lhwhite.vtk pft/mask.nii.gz pft/fodf.nii.gz pft/output/testtrack.fib -include pft/other/include.nii.gz -exclude pft/other/exclude.nii.gz -rh_atlas pft/label/rh.aparc.a2009s.annot  -lh_atlas pft/label/lh.aparc.a2009s.annot -rh_outer_surface pft/rhpial.vtk -lh_outer_surface pft/lhpial.vtk
import os
import argparse
import threading

parser = argparse.ArgumentParser(description='Surface Enhanced (SE) tracking script,\n vtk surface and .fib are in world coordinate.\n\n ***Freesurfer script needs to be in current root***')

parser.add_argument('rh_surface', type=str, default=None, help='input right surface (.vtk)')
parser.add_argument('lh_surface', type=str, default=None, help='input left surface (.vtk)')
parser.add_argument('mask', type=str, default=None, help='tractography mask (with nii transfo)')
parser.add_argument('fodf', type=str, default=None, help='tractography fodf')
parser.add_argument('tracking', type=str, default=None, help='tracking output (.fib)')

# remove step : todo input list of step to run ( ex: -run 4, ex: -run 1-4 )
parser.add_argument('--no_lps', action='store_true', default=False, help='no LPS transformation')
parser.add_argument('--no_smooth', action='store_true', default=False, help='no surface smoothing before surface tracking')

# pft tracking
parser.add_argument('-include', type=str, default=None, help='include map for pft tracking')
parser.add_argument('-exclude', type=str, default=None, help='exclude map for pft tracking')

# double interface filtering
parser.add_argument('-rh_outer_surface', type=str, default=None, help='input right outer surface (.vtk)')
parser.add_argument('-lh_outer_surface', type=str, default=None, help='input left outer surface (.vtk)')

# atlas mask
parser.add_argument('-rh_atlas', type=str, default=None, help='right surface atlas annot label')
parser.add_argument('-lh_atlas', type=str, default=None, help='left surface atlas annot label')

# parse arguments
args = parser.parse_args()

from se_script_config import params_dict

# tracking naming 
result_path, result_filename = os.path.split(args.tracking)

tracking = args.tracking
report = tracking.replace(".fib", ".txt")

rh_flow = tracking.replace(".fib", "_rh.dat")
lh_flow = tracking.replace(".fib", "_lh.dat")
rh_flow_info = tracking.replace(".fib", "_info_rh.npy")
lh_flow_info = tracking.replace(".fib", "_info_lh.npy")

rh_tracto = rh_flow.replace(".dat", ".fib")
lh_tracto = lh_flow.replace(".dat", ".fib")

tracto_cut = tracking.replace(".fib", "_cut.fib")

intersections = tracking.replace(".fib", ".npy")
surf_idx_inter = intersections.replace(".npy", "_sidx.npy")

rh_surf = args.rh_surface
lh_surf = args.lh_surface

outer_surface = False
if args.rh_outer_surface is not None:
    outer_surface = True

if outer_surface:
    if args.no_lps:
        rh_out_surf = args.rh_outer_surface
        lh_out_surf = args.lh_outer_surface
    else:
        rh_out_surf = args.tracking.replace(".fib", "_out_lps_rh.vtk")
        lh_out_surf = args.tracking.replace(".fib", "_out_lps_lh.vtk")

# surface naming 
if args.no_lps:
    rh_surf_t = rh_surf
    lh_surf_t = lh_surf
else:
    rh_surf_t = args.tracking.replace(".fib", "_lps_rh.vtk")
    lh_surf_t = args.tracking.replace(".fib", "_lps_lh.vtk")

if args.no_smooth:
    rh_surf_t_s = rh_surf_t
    lh_surf_t_s = lh_surf_t
else:
    rh_surf_t_s = args.tracking.replace(".fib", "_smooth_rh.vtk")
    lh_surf_t_s = args.tracking.replace(".fib", "_smooth_lh.vtk")

rh_trk_surf = args.tracking.replace(".fib", "_st_rh.vtk")
lh_trk_surf = args.tracking.replace(".fib", "_st_lh.vtk")

# other naming 
rh_pts = tracking.replace(".fib", "_pts_rh.npy")
lh_pts = tracking.replace(".fib", "_pts_lh.npy")
rh_normals = tracking.replace(".fib", "_nls_rh.npy")
lh_normals = tracking.replace(".fib", "_nls_lh.npy")

# masking
mask = False
if args.rh_atlas is not None and args.lh_atlas is not None:
    mask = True
    rh_mask = tracking.replace(".fib", "_mask_rh.npy")
    lh_mask = tracking.replace(".fib", "_mask_lh.npy")

if mask != outer_surface:
    raise IOError("ATLAS AND OUTER_SURFACES NEED TO BE PROVIDED TOGETHER !!!")

def launch_job(string):
    os.system(string)
    
def launch_jobs(string1, string2):
    prcs_1 = threading.Thread(target=launch_job, args=(string1,))
    prcs_2 = threading.Thread(target=launch_job, args=(string2,))
    prcs_1.start()
    prcs_2.start()
    prcs_1.join()
    prcs_2.join()

print "Anatomy :", args.mask
print "Tractography :", tracking

### Generate path
if not os.path.exists(result_path):
    os.makedirs(result_path)

### transformation
if not(args.no_lps):
    print "\n --- RUNNING se_lps.py --- \n"
    param_rh = ("python se_lps.py "
                + args.mask + " "
                + rh_surf + " "
                + rh_surf_t + " "
                + params_dict["lps"])
    param_lh = ("python se_lps.py "
                + args.mask + " "
                + lh_surf + " "
                + lh_surf_t + " "
                + params_dict["lps"])
    launch_jobs(param_rh, param_lh)
    if outer_surface:
        param_rh = ("python se_lps.py "
                    + args.mask + " "
                    + args.rh_outer_surface + " "
                    + rh_out_surf + " "
                    + params_dict["lps"])
        param_lh = ("python se_lps.py "
                    + args.mask + " "
                    + args.lh_outer_surface + " "
                    + lh_out_surf + " "
                    + params_dict["lps"])
        launch_jobs(param_rh, param_lh)

### se_atlas mask
if mask:
    print "\n --- RUNNING se_atlas.py --- \n"
    param_rh = ("python se_atlas.py "
                + rh_surf_t + " "
                + args.rh_atlas + " "
                + "-out_vts_mask " + rh_mask + " "
                + params_dict["mask"])
    param_lh = ("python se_atlas.py "
                + lh_surf_t + " "
                + args.lh_atlas + " "
                + "-out_vts_mask " + lh_mask + " "
                + params_dict["mask"])
    launch_jobs(param_rh, param_lh)
    
    
### smooth the surface
if not(args.no_smooth):
    print "\n --- RUNNING se_smooth.py --- \n"
    param_rh = ("python se_smooth.py "
                + rh_surf_t + " "
                + rh_surf_t_s + " "
                + params_dict["smooth"])
    param_lh = ("python se_smooth.py "
                + lh_surf_t + " "
                + lh_surf_t_s + " "
                + params_dict["smooth"])
    if mask:
        param_rh += " -mask " + rh_mask
        param_lh += " -mask " + lh_mask
    
    launch_jobs(param_rh, param_lh)

### Surface tracking
print "\n --- RUNNING se_tracking.py --- \n"
param_rh = ("python se_tracking.py "
          + rh_surf_t_s + " "
          + rh_flow + " "
          + "-end_points " + rh_pts + " "
          + "-end_normal " + rh_normals  + " "
          + "-end_surf " + rh_trk_surf  + " "
          + "-info " + rh_flow_info + " "
          + params_dict["st"])
param_lh = ("python se_tracking.py "
          + lh_surf_t_s + " "
          + lh_flow + " "
          + "-end_points " + lh_pts + " "
          + "-end_normal " + lh_normals  + " "
          + "-end_surf " + lh_trk_surf  + " "
          + "-info " + lh_flow_info + " "
          + params_dict["st"])
if mask:
    param_rh += " -mask " + rh_mask
    param_lh += " -mask " + lh_mask

launch_jobs(param_rh, param_lh)

# TODO run multiple with the same seed but different "--random"

if args.include is None:
    ### dMRI tractography
    print "\n --- RUNNING compute_local_tracking_mesh.py --- \n"
    param_rh = ("python compute_local_tracking_mesh.py "
              + args.fodf + " "
              + rh_pts + " "
              + rh_normals + " "
              + args.mask  + " "
              + rh_tracto + " "
              + params_dict["tracto"])
    param_lh = ("python compute_local_tracking_mesh.py " 
              + args.fodf + " "
              + lh_pts + " "
              + lh_normals + " "
              + args.mask  + " "
              + lh_tracto + " "
              + params_dict["tracto"])
    launch_jobs(param_rh, param_lh)
else:
    ### dMRI PFT tractography
    print "\n --- RUNNING compute_pft_tracking_mesh.py --- \n"
    param_rh = ("python compute_pft_tracking_mesh.py "
              + args.fodf + " "
              + rh_pts + " "
              + rh_normals + " "
              + args.include  + " "
              + args.exclude  + " "
              + rh_tracto + " "
              + params_dict["pft_tracto"])
    
    param_lh = ("python compute_pft_tracking_mesh.py " 
              + args.fodf + " "
              + lh_pts + " "
              + lh_normals + " "
              + args.include  + " "
              + args.exclude  + " "
              + lh_tracto + " "
              + params_dict["pft_tracto"])
    launch_jobs(param_rh, param_lh)

### tractography intersection
print "\n --- RUNNING se_mesh_intersect.py --- \n"
if not(outer_surface):
    param = ("python se_mesh_intersect.py "
              + rh_tracto + " "
              + lh_tracto + " "
              + rh_trk_surf + " "
              + lh_trk_surf + " "
              + tracto_cut + " "
              + intersections + " "
              + surf_idx_inter + " "
              + "-report " + report + " "
              + params_dict["intersect"])
    if mask:
        param += " -rh_mask " + rh_mask + " -lh_mask " + lh_mask
    launch_job(param)
else:
    param = ("python se_mesh_intersect2.py "
              + rh_tracto + " "
              + lh_tracto + " "
              + rh_trk_surf + " "
              + lh_trk_surf + " "
              + rh_out_surf + " "
              + lh_out_surf + " "
              + rh_mask + " "
              + lh_mask + " "
              + tracto_cut + " "
              + intersections + " "
              + surf_idx_inter + " "
              + "-report " + report + " "
              + params_dict["intersect"])
    launch_job(param)

### tractography fusion with surface tracking
print "\n --- RUNNING se_fusion.py --- \n"
param = ("python se_fusion.py "
          + tracto_cut + " "
          + rh_trk_surf + " "
          + lh_trk_surf + " "
          + rh_flow + " "
          + lh_flow + " "
          + rh_flow_info + " "
          + lh_flow_info + " "
          + intersections + " "
          + surf_idx_inter + " "
          + tracking + " "
          + params_dict["fusion"])
launch_job(param)
print "\n --- Done --- \n"


