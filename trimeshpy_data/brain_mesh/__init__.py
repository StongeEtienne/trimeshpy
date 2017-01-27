
import os
local_path = os.path.dirname(os.path.abspath(__file__)) + "/"

brain_lh = local_path + "100307_lh.vtk"
brain_rh = local_path + "100307_rh.vtk"

brain_nuclei = local_path + "100307_nuclei.vtk"
brain_stem = local_path + "100307_stem.vtk"

brain_lh_smoothed = local_path + "100307_smooth_lh.vtk"
brain_rh_smoothed = local_path + "100307_smooth_rh.vtk"
brain_lh_smoothed_set = local_path + "100307_set_lh.vtk"
brain_rh_smoothed_set = local_path + "100307_set_rh.vtk"

brain_t1 = local_path + "100307_t1.nii.gz"
