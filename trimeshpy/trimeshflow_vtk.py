# Etienne St-Onge

import logging

import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.trimeshflow_class import TriMeshFlow
import trimeshpy.vtk_util as vtk_u

vtk = vtk_u.import_vtk()


# "TriMeshFlow_Vtk" is Based on "TriMeshFlow",
#  but contain display functions from "TriMesh_Vtk"
class TriMeshFlow_Vtk(TriMeshFlow, TriMesh_Vtk):

    def __init__(self, triangles, vertices_flow,
                 dtype=np.float64, atol=1e-8, assert_args=True):
        self.__polydata__ = None
        self.__polydata_is_up_to_date__ = False
        self.__polydata_color_is_scalars__ = None
        TriMeshFlow.__init__(self, triangles=triangles,
                             vertices_flow=vertices_flow,
                             dtype=dtype, atol=atol, assert_args=assert_args)

    # set and get, add an to update bool
    def set_triangles(self, triangles):
        TriMeshFlow.set_triangles(self, triangles)
        self.__polydata_is_up_to_date__ = False

    def set_vertices(self, vertices):
        TriMeshFlow.set_vertices(self, vertices)
        self.__polydata_is_up_to_date__ = False

    def set_vertices_flow(self, vertices_flow):
        TriMeshFlow.set_vertices_flow(self, vertices_flow)
        self.__polydata_is_up_to_date__ = False

    def set_vertices_flow_from_memmap(self, vertices_flow_memmap,
                                      flow_length, nb_vertices):
        TriMeshFlow.set_vertices_flow_from_memmap(
            self, vertices_flow_memmap=vertices_flow_memmap,
            flow_length=flow_length, nb_vertices=nb_vertices)
        self.__polydata_is_up_to_date__ = False

    def set_vertices_flow_from_hdf5(self, vertices_flow_hdf5):
        TriMeshFlow.set_vertices_flow_from_hdf5(
            self, vertices_flow_hdf5)
        self.__polydata_is_up_to_date__ = False

    # vtk polydata function
    def get_polydata(self):
        if self.__polydata_is_up_to_date__ is False:
            self.__polydata__ = vtk.vtkPolyData()
            self.__polydata_is_up_to_date__ = True
            self.update_polydata()
        return self.__polydata__

    def update_polydata(self, vertices_flow_index=-1):
        TriMesh_Vtk.set_polydata_triangles(
            self, TriMeshFlow.get_triangles(self))
        TriMesh_Vtk.set_polydata_vertices(
            self, TriMeshFlow.get_vertices(self, vertices_flow_index=vertices_flow_index))

    def display_vertices_flow(self, display_name="trimesh", size=(400, 400),
                              png_magnify=1):
        renderer = vtk.vtkRenderer()
        renderer.AddActor(self.get_vertices_flow_actor())
        renderer.ResetCamera()
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetWindowName(display_name)
        window.SetSize(size[0], size[1])
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(window)
        picker = vtk.vtkCellPicker()
        iren.SetPicker(picker)

        def key_press(obj, event):
            key = obj.GetKeySym()
            if key == 's' or key == 'S':
                logging.info('Saving image...')
                renderLarge = vtk.vtkRenderLargeImage()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    renderLarge.SetInput(renderer)
                else:
                    renderLarge.SetInputData(renderer)

                renderLarge.SetMagnification(png_magnify)
                renderLarge.Update()
                writer = vtk.vtkPNGWriter()
                writer.SetInputConnection(renderLarge.GetOutputPort())
                writer.SetFileName('trimesh_save.png')
                writer.Write()
                logging.info('Look for trimesh_save.png in your current directory.')

        iren.AddObserver('KeyPressEvent', key_press)
        iren.SetInteractorStyle(style)
        iren.Initialize()
        picker.Pick(85, 126, 0, renderer)
        window.Render()
        iren.Start()
        window.RemoveRenderer(renderer)
        renderer.SetRenderWindow(None)

    def get_vertices_flow_actor(self, colors="RGB", opacity=1, linewidth=1,
                                spline_subdiv=None):
        lines = np.swapaxes(self.get_vertices_flow(), 0, 1)
        poly_data = vtk_u.lines_to_vtk_polydata(lines, colors)
        next_input = poly_data

        # use spline interpolation
        if (spline_subdiv is not None) and (spline_subdiv > 0):
            spline_filter = vtk_u.set_input(vtk.vtkSplineFilter(), next_input)
            spline_filter.SetSubdivideToSpecified()
            spline_filter.SetNumberOfSubdivisions(spline_subdiv)
            spline_filter.Update()
            next_input = spline_filter.GetOutputPort()

        poly_mapper = vtk_u.set_input(vtk.vtkPolyDataMapper(), next_input)
        poly_mapper.ScalarVisibilityOn()
        poly_mapper.SetScalarModeToUsePointFieldData()
        poly_mapper.SelectColorArray("Colors")
        poly_mapper.Update()

        # Set Actor
        actor = vtk.vtkActor()
        actor.SetMapper(poly_mapper)
        actor.GetProperty().SetLineWidth(linewidth)
        actor.GetProperty().SetOpacity(opacity)
        return actor
