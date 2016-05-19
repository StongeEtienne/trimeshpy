import vtk
import vtk.util.numpy_support as ns
import numpy as np

# Utility functions
def set_input(vtk_object, current_input):
    """ Generic input for vtk data, """
    if isinstance(current_input, vtk.vtkPolyData):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(current_input)
        else:
            vtk_object.SetInputData(current_input)
    elif isinstance(input, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(current_input)

    vtk_object.Update()
    return vtk_object


def load_polydata(file_name):
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    # todo better generic load
    if file_extension == "vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "vtp":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == "ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == "stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == "xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == "obj":
        reader = vtk.vtkOBJReader()
        #try:  # try to read as a normal obj
        #    reader = vtk.vtkOBJReader()
        #except:  # than try load a MNI obj format
        #    reader = vtk.vtkMNIObjectReader()
    else:
        raise "polydata " + file_extension + " is not suported"

    reader.SetFileName(file_name)
    reader.Update()
    print file_name, " Mesh ", file_extension, "Loaded"
    return reader.GetOutput()

def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

    # todo better generic load
    # todo test all
    if file_extension == "vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "vtp":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "fib":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == "ply":
        writer = vtk.vtkPLYWriter()
    elif file_extension == "stl":
        writer = vtk.vtkSTLWriter()
    elif file_extension == "xml":
        writer = vtk.vtkXMLPolyDataWriter()
    elif file_extension == "obj":
        raise "mni obj or Wavefront obj ?"
    #    writer = set_input(vtk.vtkMNIObjectWriter(), polydata)

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None:
        writer.SetArrayName(color_array_name);
    
    if binary :
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
    
    
def load_streamlines_poyldata(file_name):
    # get file extension (type)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    line_data = reader.GetOutput()
    return line_data
    
def get_streamlines(line_polydata):
    lines_vertices = ns.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(line_polydata.GetLines().GetData())
    
    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]
        #print line_len
        next_idx = current_idx + line_len + 1 
        line_range = lines_idx[current_idx + 1: next_idx]
        #print line_range
        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines

def lines_to_vtk_polydata(lines, colors="RGB", dtype=None):#"RGB"
    # Get the 3d points_array
    points_array = np.vstack(lines)

    nb_lines = len(lines)
    nb_points = len(points_array)

    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    # todo put from array
    lines_array = []
    points_per_line = np.zeros([nb_lines], np.int64)
    current_position = 0
    for i in lines_range:
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position
    
    if dtype is None:
        lines_array = np.array(lines_array)
    else:
        lines_array = np.array(lines_array)

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array.astype(dtype))

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.GetData().DeepCopy(ns.numpy_to_vtk(lines_array))
    vtk_lines.SetNumberOfCells(len(lines))

    # colors test, todo improve
    if colors is not None:
        if colors == "RGB":  # set automatic rgb colors
            print "RGB to improve"
            colors = np.abs(lines[:,-1] - lines[:,0])
            colors = 0.8 * colors / np.sqrt(np.sum(colors ** 2, axis=1, keepdims=True))
            colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
            vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])
        else:
            colors = np.array(colors)
            if colors.dtype == np.object:  # colors is a list of colors
                vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
            else:
                if len(colors) == nb_points:  # one colors per points / colormap way
                    vtk_colors = ns.numpy_to_vtk(colors, deep=True)
                    raise NotImplementedError()
    
                elif colors.ndim == 1:  # the same colors for all points
                    vtk_colors = numpy_to_vtk_colors(np.tile(255 * colors, (nb_points, 1)))
    
                elif colors.ndim == 2:  # map color to each line
                    colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                    vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])

        vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)
    
    if colors is not None:
        poly_data.GetPointData().SetScalars(vtk_colors)

    return poly_data

# todo improve
def vtkIdList_to_array(vtkIdList):
    array = np.zeros(vtkIdList.GetNumberOfIds(), dtype=np.uint64)
    for i in range(vtkIdList.GetNumberOfIds()):
        array[i] = vtkIdList.GetId(i)
    return  array

def numpy_to_vtk_colors(colors):
    vtk_colors = ns.numpy_to_vtk(np.asarray(colors), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors

def numpy_to_vtk_points(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True))
    return vtk_points
