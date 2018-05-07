
import os
local_path = os.path.dirname(os.path.abspath(__file__)) + "/"

brain_lh = local_path + "100307_white_lh.vtk"
brain_rh = local_path + "100307_white_rh.vtk"

brain_nuclei = local_path + "100307_nuclei.vtk"
brain_stem = local_path + "100307_stem.vtk"

brain_t1 = local_path + "100307_t1.nii.gz"
brain_wmparc = local_path + "100307_wmparc.nii.gz"

brain_lh_aparc = local_path + "100307_aparc_lh.annot"
brain_rh_aparc = local_path + "100307_aparc_rh.annot"
brain_lh_a2009s = local_path + "100307_a2009s_lh.annot"
brain_rh_a2009s = local_path + "100307_a2009s_rh.annot"
