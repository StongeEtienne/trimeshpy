
from trimeshpy.trimesh_vtk import load_polydata, save_polydata, load_streamlines_poyldata, get_streamlines
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
import numpy as np

fib_file_name = "../data/tract.fib"
save_file = "../data/tract.xml"
#save_file = "../data/tract.stl"


polydata = load_polydata(fib_file_name)
save_polydata(polydata, save_file)


### load streamlines liste and save a a new smaller file
"""lines = get_streamlines(load_streamlines_poyldata(fib_file_name))
new_lines = lines[0:1000]
lines_polydata = lines_to_vtk_polydata(new_lines, None, np.float32)
save_polydata(lines_polydata, "../data/tract2.fib")"""