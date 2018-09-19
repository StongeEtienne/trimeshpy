# Etienne St-Onge
from __future__ import division
import importlib
import numpy as np
from scipy.ndimage import map_coordinates

# Import vtk functions
def import_vtk():
    try:
        vtk = importlib.import_module('vtk')
    except ImportError:
        print("Unable to import VTK")
        vtk = None
        
    return vtk

def import_vtk_numpy_support():
    try:
        ns = importlib.import_module('vtk.util.numpy_support')
    except ImportError:
        print("Unable to import VTK numpy support")
        ns = None
        
    return ns

# Find a better way to self use
vtk = import_vtk()
ns = import_vtk_numpy_support()

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


# Load
def load_streamlines(file_name):
    return get_streamlines(load_polydata(file_name))


def get_streamlines(line_polydata):
    lines_vertices = ns.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(line_polydata.GetLines().GetData())

    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]
        next_idx = current_idx + line_len + 1
        line_range = lines_idx[current_idx + 1: next_idx]
        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines


def load_polydata(file_name):
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

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
        # special case, since there is two obj format
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        if reader.GetOutput().GetNumberOfCells() == 0:
            reader = vtk.vtkMNIObjectReader()
    else:
        raise "polydata " + file_extension + " is not suported"

    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


# Save
def save_polydata(polydata, file_name, binary=False, color_array_name=None):
    # get file extension (type)
    file_extension = file_name.split(".")[-1].lower()

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
        # writer = set_input(vtk.vtkMNIObjectWriter(), polydata)
        raise "mni obj or Wavefront obj ?"

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None:
        writer.SetArrayName(color_array_name)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def lines_to_vtk_polydata(lines, colors="RGB"):
    # Get the 3d points_array
    points_array = np.vstack(lines).astype(np.float32)

    nb_lines = len(lines)
    nb_points = len(points_array)
    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    # todo put from array
    lines_array = []
    points_per_line = np.zeros([nb_lines], dtype=np.int32)
    current_position = 0
    for i in xrange(nb_lines):
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)
    lines_array = ns.numpy_to_vtk(lines_array, array_type=vtk.VTK_INT)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines.SetNumberOfCells(nb_lines)
    vtk_lines.GetData().DeepCopy(lines_array)

    # colors test, todo improve
    if colors is not None:
        if colors == "RGB":  # set automatic rgb colors
            colors = np.abs(lines[:, -1] - lines[:, 0])
            colors = 0.8 * colors / \
                np.sqrt(np.sum(colors ** 2, axis=1, keepdims=True))
            colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
            vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])
        else:
            colors = np.array(colors)
            if colors.dtype == np.object:  # colors is a list of colors
                vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))
            else:
                # one colors per points / colormap way
                if len(colors) == nb_points:
                    vtk_colors = ns.numpy_to_vtk(colors, deep=True)
                    raise NotImplementedError()

                elif colors.ndim == 1:  # the same colors for all points
                    vtk_colors = numpy_to_vtk_colors(
                        np.tile(255 * colors, (nb_points, 1)))

                elif colors.ndim == 2:  # map color to each line
                    colors_mapper = np.repeat(
                        lines_range, points_per_line, axis=0)
                    vtk_colors = numpy_to_vtk_colors(
                        255 * colors[colors_mapper])

        vtk_colors.SetName("Colors")

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    if colors is not None:
        poly_data.GetPointData().SetScalars(vtk_colors)

    return poly_data


def vtkIdList_to_array(vtkIdList):
    array = np.zeros(vtkIdList.GetNumberOfIds(), dtype=np.uint64)
    for i in range(vtkIdList.GetNumberOfIds()):
        array[i] = vtkIdList.GetId(i)
    return array


def numpy_to_vtk_colors(colors):
    vtk_colors = ns.numpy_to_vtk(
        np.asarray(colors), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    return vtk_colors


def numpy_to_vtk_points(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True, array_type=vtk.VTK_FLOAT))
    return vtk_points


def get_polydata_triangles(polydata):
    vtk_polys = ns.vtk_to_numpy(polydata.GetPolys().GetData())
    assert((vtk_polys[::4] == 3).all())  # test if really triangle
    return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T


def get_polydata_vertices(polydata):
    return ns.vtk_to_numpy(polydata.GetPoints().GetData())


def map_coordinates_3d_4d(input_array, indices):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation

    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray

    Returns
    -------
    output : ndarray
        1D or 2D array
    """

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=1)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=1)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def generate_colormap(scale_range=(0.0, 1.0), hue_range=(0.8, 0.0),
                      saturation_range=(1.0, 1.0), value_range=(0.8, 0.8),
                      nan_color=(0.2, 0.2, 0.2, 1.0)):
    """ Generate colormap's lookup table

    Parameters
    ----------
    scale_range : tuple
        It can be anything e.g. (0, 1) or (0, 255). Usually it is the mininum
        and maximum value of your data. Default is (0, 1).
    hue_range : tuple of floats
        HSV values (min 0 and max 1). Default is (0.8, 0).
    saturation_range : tuple of floats
        HSV values (min 0 and max 1). Default is (1, 1).
    value_range : tuple of floats
        HSV value (min 0 and max 1). Default is (0.8, 0.8).

    Returns
    -------
    lookup_table : vtkLookupTable

    """
    lookup_table = vtk.vtkLookupTable()
    lookup_table.SetRange(scale_range)

    lookup_table.SetHueRange(hue_range)
    lookup_table.SetSaturationRange(saturation_range)
    lookup_table.SetValueRange(value_range)
    lookup_table.SetNanColor(nan_color)
    lookup_table.Build()
    return lookup_table


# Streamlines generic processing
def streamlines_to_endpoints(streamlines):
    endpoints = np.zeros((2, len(streamlines), 3))
    for i, streamline in enumerate(streamlines):
        endpoints[0, i] = streamline[0]
        endpoints[1, i] = streamline[-1]
    return endpoints

