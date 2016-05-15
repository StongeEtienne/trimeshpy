
from trimeshpy.trimesh_vtk import load_polydata, save_polydata, load_streamlines_poyldata, get_streamlines
from trimeshpy.trimesh_vtk import set_input
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
import numpy as np
import vtk
from dipy.viz.actor import streamtube
from dipy.viz import window

fib_file_name = "/home/eti/home_mint/final_smooth_2_5_flow_100_1_prob.fib"
save_file = "/home/eti/home_mint/Nic/s100_s100_tube_flow.obj"
#save_file = "../data/tract.stl"

min_length = 10
spline_subdiv = 100
tube_sides = 9
linewidth = 0.1

polydata_in = load_polydata(fib_file_name)
streamlines_in = get_streamlines(polydata_in)
streamlines = []

for line in streamlines_in:
    dist = line[:-1] - line[1:]
    line_length = np.sum(np.sqrt(np.sum(np.square(dist), axis=1)))
        
    if line_length > min_length:
        streamlines.append(line)

#streamlines_sub = streamlines
streamlines_sub = streamlines[::100]

print len(streamlines_in), len(streamlines_sub)

actor = streamtube(streamlines_sub, linewidth=linewidth, tube_sides=tube_sides, spline_subdiv=spline_subdiv)

renderer = window.Renderer()
renderer.add(actor)
my_window = window.show(renderer)

#objexporter = vtk.vtkOBJExporter()
#objexporter.SetInput(my_window)
#objexporter.SetFileName(save_file)
#objexporter.SetFileName("/home/eti/home_mint/Nic/s100_s100_tube_flow.obj")
#objexporter.Write()

#save_polydata(actor.GetMapper().GetInput(), save_file)
