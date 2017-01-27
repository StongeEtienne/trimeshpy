
import numpy as np

import trimeshpy
from trimeshpy.vtk_util import  load_polydata, get_streamlines

from dipy.viz.actor import streamtube
from dipy.viz import window

fib_file = "tract.fib"
surface_file = "tract.vtk"
folder_path = trimeshpy.data.output_path

fib_file = trimeshpy.data.output_path + fib_file
surface_file = trimeshpy.data.output_path + surface_file


min_length = 10
spline_subdiv = 100
tube_sides = 9
linewidth = 0.1

polydata_in = load_polydata(fib_file)
streamlines_in = get_streamlines(polydata_in)
streamlines = []

for line in streamlines_in:
    dist = line[:-1] - line[1:]
    line_length = np.sum(np.sqrt(np.sum(np.square(dist), axis=1)))
        
    if line_length > min_length:
        streamlines.append(line)

### Reduce the amount of streamlines to display
#streamlines_sub = streamlines
streamlines_sub = streamlines[::100]

print len(streamlines_in), len(streamlines_sub)

actor = streamtube(streamlines_sub, linewidth=linewidth, tube_sides=tube_sides, spline_subdiv=spline_subdiv)

renderer = window.Renderer()
renderer.add(actor)
my_window = window.show(renderer)


### Save geometry (tubes streamlines) if needed
# save_polydata(actor.GetMapper().GetInput(), surface_file)


# display with vtk
# import vtk
# objexporter = vtk.vtkOBJExporter()
# objexporter.SetInput(my_window)
# objexporter.SetFileName(surface_file)
# objexporter.SetFileName("/home/eti/home_mint/Nic/s100_s100_tube_flow.obj")
# objexporter.Write()

