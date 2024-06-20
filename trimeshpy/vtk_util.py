# Etienne St-Onge

import h5py
import importlib
import logging

import numpy as np
from scipy.ndimage import map_coordinates
from nibabel.affines import apply_affine


# Import vtk functions
def import_vtk():
    try:
        vtk_lib = importlib.import_module('vtk')
    except ImportError:
        logging.warning("Unable to import VTK")
        vtk_lib = None

    return vtk_lib


def import_vtk_numpy_support():
    try:
        ns_lib = importlib.import_module('vtk.util.numpy_support')
    except ImportError:
        logging.warning("Unable to import VTK numpy support")
        ns_lib = None

    return ns_lib


# Find a better way to self use
vtk = import_vtk()
ns = import_vtk_numpy_support()


# Utility functions
def set_input(vtk_object, current_input):
    """ Generic input for vtk data, """
    if isinstance(current_input, vtk.vtkPolyData):
        vtk_object.SetInputData(current_input)
    elif isinstance(input, vtk.vtkAlgorithmOutput):
        vtk_object.SetInputConnection(current_input)

    vtk_object.Update()
    return vtk_object


# Load
def load_streamlines(file_name):
    file_extension = file_name.split(".")[-1].lower()
    if file_extension == "hdf5":
        vertices_flow = load_hdf5_vertices(file_name)
        nb_lines = vertices_flow.shape[1]
        lines = [vertices_flow[:, i] for i in range(nb_lines)]
        return lines
    else:
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


def load_hdf5_vertices(file_name):
    f = h5py.File(file_name, mode='r')
    return f["vertices"]


def load_hdf5_triangles(file_name):
    f = h5py.File(file_name, mode='r')
    return f["triangles"]


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
        raise IOError("." + file_extension + " is not supported by TriMeshPy")

    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


# Save
def save_polydata(polydata, file_name, binary=False, color_array_name=None, legacy_vtk_format=False):
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
        find_keyword = file_name.lower().split(".")
        if "mni" in find_keyword or "mnc" in find_keyword:
            writer = vtk.vtkMNIObjectWriter()
        else:
            raise IOError("mni obj or Wavefront obj (require a scene)")
    else:
        raise IOError("." + file_extension + " is not supported by TriMeshPy")

    writer.SetFileName(file_name)
    writer = set_input(writer, polydata)
    if color_array_name is not None:
        writer.SetArrayName(color_array_name)
    if legacy_vtk_format and file_extension in ["vtk", "vtp", "fib"]:
        writer.SetFileVersion(42)
    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()


def lines_to_vtk_polydata(lines, colors=None):
    # Get the 3d points_array
    points_array = np.vstack(lines).astype(np.float32)

    nb_lines = len(lines)
    nb_points = len(points_array)
    lines_range = range(nb_lines)

    # Get lines_array in vtk input format
    lines_array = []
    points_per_line = np.zeros([nb_lines], dtype=np.int32)
    current_position = 0
    for i in range(nb_lines):
        current_len = len(lines[i])
        points_per_line[i] = current_len

        end_position = current_position + current_len
        lines_array += [current_len]
        lines_array += range(current_position, end_position)
        current_position = end_position

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_lines = vtk.vtkCellArray()
    vtk_lines_array = ns.numpy_to_vtk(np.asarray(lines_array), deep=True, array_type=vtk.VTK_ID_TYPE)
    vtk_lines.SetCells(nb_lines, vtk_lines_array)

    # Create the poly_data
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(vtk_lines)

    if colors is not None:
        vtk_colors = None
        if (colors is True) or (isinstance(colors, str) and colors == "RGB"):
            cols_arr = np.zeros_like(points_array)
            cols_arr[1:] = np.diff(points_array, axis=0)
            cols_arr[0] = cols_arr[1]
            offsets = np.cumsum(points_per_line)
            cols_arr[offsets[:-1]] = cols_arr[offsets[:-1]+1]
            cols_arr[offsets-1] = cols_arr[offsets-2]
            cols_arr = np.abs(cols_arr) / np.sqrt(np.sum(np.square(cols_arr), axis=1, keepdims=True))
            vtk_colors = numpy_to_vtk_colors(255 * cols_arr)
        else:
            try:
                colors = np.asarray(colors)
                if colors.ndim == 1:  # the same colors for all points
                    if len(colors) == 1 or len(colors) == 3 or len(colors) == 4:
                        vtk_colors = numpy_to_vtk_colors(np.tile(255 * colors, (nb_points, 1)))
                    elif len(colors) == nb_lines:
                        colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                        vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])
                    elif len(colors) == nb_points:
                        vtk_colors = numpy_to_vtk_colors(255 * colors)
                if colors.ndim == 2:   # map color to each line
                    colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
                    vtk_colors = numpy_to_vtk_colors(255 * colors[colors_mapper])

            except ValueError:
                if len(colors) == nb_lines:
                    # assume one color per points in a list of list
                    vtk_colors = numpy_to_vtk_colors(255 * np.vstack(colors))

        if vtk_colors is not None:
            vtk_colors.SetName("Colors")
            poly_data.GetPointData().SetScalars(vtk_colors)
        else:
            raise NotImplementedError()

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


# transformation
def vtk_to_vox(vts, nibabel_img):
    inv_affine = np.linalg.inv(nibabel_img.affine)
    flip = np.diag([-1, -1, 1, 1])
    vts = apply_affine(np.dot(inv_affine, flip), vts)
    return vts


def vtk_to_voxmm(vts, nibabel_img):
    scale = np.array(nibabel_img.get_header().get_zooms())
    return vtk_to_vox(vts, nibabel_img) * scale


def vox_to_vtk(vts, nibabel_img):
    flip = np.diag([-1., -1., 1., 1.])
    vts = apply_affine(np.dot(flip, nibabel_img.affine), vts)
    return vts


def voxmm_to_vtk(vts, nibabel_img):
    scale = np.array(nibabel_img.get_header().get_zooms(), dtype=float)
    return vox_to_vtk(vts / scale, nibabel_img)


# Streamlines generic processing
def streamlines_to_endpoints(streamlines):
    endpoints = np.zeros((2, len(streamlines), 3))
    for i, streamline in enumerate(streamlines):
        endpoints[0, i] = streamline[0]
        endpoints[1, i] = streamline[-1]
    return endpoints
