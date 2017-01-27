
import trimeshpy
from trimeshpy.vtk_util import load_polydata, save_polydata

fib_file = "tract.fib"
surface_file = "tract.vtk"
folder_path = trimeshpy.data.output_path

fib_file = trimeshpy.data.output_path + fib_file
surface_file = trimeshpy.data.output_path + surface_file
#save_file = "../data/tract.stl"

polydata = load_polydata(fib_file)
save_polydata(polydata, surface_file)