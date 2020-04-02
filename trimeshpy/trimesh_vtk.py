# Etienne St-Onge

import logging
from six import string_types

import numpy as np

from trimeshpy.trimesh_class import TriMesh
import trimeshpy.vtk_util as vtk_u


vtk = vtk_u.import_vtk()
ns = vtk_u.import_vtk_numpy_support()


class TriMesh_Vtk(TriMesh):
    """
     Triangle Mesh VTK class
        fast python mesh processing
        with numpy, scipy, and VTK

    Triangle list
    Vertex list
    vtkPolyData generated when needed
    """

    def __init__(self, triangles, vertices, dtype=np.float64,
                 atol=1e-8, assert_args=True):

        if isinstance(triangles, string_types):
            mesh_filename = triangles
        elif isinstance(vertices, string_types):
            mesh_filename = vertices
        else:
            mesh_filename = None

        if mesh_filename is not None:
            self.__polydata__ = vtk_u.load_polydata(mesh_filename)
            self.__polydata_is_up_to_date__ = True
            self.__polydata_color_is_scalars__ = None
            TriMesh.__init__(self, self.get_polydata_triangles(),
                             self.get_polydata_vertices(),
                             dtype=dtype, atol=atol, assert_args=assert_args)
        else:
            self.__polydata__ = None
            self.__polydata_is_up_to_date__ = False
            self.__polydata_color_is_scalars__ = None
            TriMesh.__init__(self, triangles, vertices,
                             dtype=dtype, atol=atol, assert_args=assert_args)

    # set and get, add an to update bool
    def set_triangles(self, triangles):
        TriMesh.set_triangles(self, triangles)
        self.__polydata_is_up_to_date__ = False

    def set_vertices(self, vertices):
        TriMesh.set_vertices(self, vertices)
        self.__polydata_is_up_to_date__ = False

    # vtk polydata function
    def get_polydata(self, update_normal=False):
        if self.__polydata_is_up_to_date__ is False:
            self.update_polydata()
        if update_normal:
            self.update_normals()
        return self.__polydata__

    def set_polydata(self, new_polydata):
        self.__polydata__ = new_polydata
        self.__polydata_is_up_to_date__ = True
        self.__polydata_color_is_scalars__ = None
        TriMesh.__init__(self, self.get_polydata_triangles(),
                         self.get_polydata_vertices(),
                         dtype=self.__dtype__, atol=self.__atol__)

    def update_polydata(self):
        self.__polydata__ = vtk.vtkPolyData()
        self.__polydata_is_up_to_date__ = True
        self.set_polydata_triangles(self.get_triangles())
        self.set_polydata_vertices(self.get_vertices())

    def get_polydata_triangles(self):
        return vtk_u.get_polydata_triangles(self.get_polydata())

    def get_polydata_vertices(self):
        return vtk_u.get_polydata_vertices(self.get_polydata())

    def set_polydata_triangles(self, triangles):
        vtk_triangles = np.hstack(
            np.c_[np.ones(len(triangles)).astype(np.int) * 3, triangles])
        vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(triangles), vtk_triangles)
        self.get_polydata().SetPolys(vtk_cells)

    def set_polydata_vertices(self, vertices):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
        self.get_polydata().SetPoints(vtk_points)

    # Normals
    def get_normals(self):
        vtk_normals = self.get_polydata().GetPointData().GetNormals()
        if vtk_normals is None:
            return None
        else:
            return ns.vtk_to_numpy(vtk_normals)

    def set_normals(self, normals):
        vtk_normals = ns.numpy_to_vtk(normals, deep=True)
        self.get_polydata().GetPointData().SetNormals(vtk_normals)

    # Colors
    def get_colors(self):
        vtk_colors = self.get_polydata().GetPointData().GetScalars()
        if vtk_colors is None:
            return None
        else:
            return ns.vtk_to_numpy(vtk_colors)

    def set_colors(self, colors):
        # Colors are [0,255] RGB for each points
        vtk_colors = ns.numpy_to_vtk(
            colors, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_colors.SetNumberOfComponents(colors.shape[1])
        vtk_colors.SetName("RGB")
        self.get_polydata().GetPointData().SetScalars(vtk_colors)
        self.__polydata_color_is_scalars__ = False

    def set_scalars(self, scalars, colormap=None):
        scalars = scalars.astype(float)
        vtk_scalars = ns.numpy_to_vtk(scalars, deep=True)
        vtk_scalars.SetNumberOfComponents(1)
        vtk_scalars.SetName("Scalars")
        self.get_polydata().GetPointData().SetScalars(vtk_scalars)

        self.__polydata_color_is_scalars__ = True
        if colormap is not None:
            self.set_colormap(colormap)
        else:
            self.set_colormap(vtk_u.generate_colormap(
                scale_range=(np.nanmin(scalars), np.nanmax(scalars))))

    def get_scalars(self):
        vtk_scalars = self.get_polydata().GetPointData().GetScalars()
        if vtk_scalars is None:
            return None
        else:
            return ns.vtk_to_numpy(vtk_scalars)

    def set_colormap(self, colormap):
        if self.__polydata_color_is_scalars__:
            self.__colormap__ = colormap
        else:
            logging.warning("WARNING: call 'set_colormap()' after 'set_scalars'")

    def get_colormap(self):
        return self.__colormap__

    # Updates :
    def update_normals(self):
        normals_gen = self.polydata_input(vtk.vtkPolyDataNormals())
        normals_gen.ComputePointNormalsOn()
        normals_gen.ComputeCellNormalsOn()
        normals_gen.SplittingOff()
        # normals_gen.FlipNormalsOn()
        # normals_gen.ConsistencyOn()
        # normals_gen.AutoOrientNormalsOn()
        normals_gen.Update()

        # memory leak if we use :  self.polydata = normals_gen.GetOutput()
        # dont copy the polydata because memory leak
        vtk_normals = normals_gen.GetOutput().GetPointData().GetNormals()
        self.get_polydata().GetPointData().SetNormals(vtk_normals)

    def update(self):
        if vtk.VTK_MAJOR_VERSION <= 5:
            self.get_polydata().Update()
        else:
            self.update_polydata()

    # Display :
    def get_vtk_polymapper(self):
        poly_mapper = self.polydata_input(vtk.vtkPolyDataMapper())
        poly_mapper.ScalarVisibilityOn()
        poly_mapper.InterpolateScalarsBeforeMappingOn()
        poly_mapper.StaticOn()
        poly_mapper.Update()

        if self.__polydata_color_is_scalars__:
            poly_mapper.SetLookupTable(self.get_colormap())
            poly_mapper.SetScalarModeToUsePointData()
            poly_mapper.UseLookupTableScalarRangeOn()

        return poly_mapper

    def get_vtk_actor(self, light=(0.2, 0.15, 0.05)):
        poly_mapper = self.get_vtk_polymapper()
        actor = vtk.vtkActor()
        actor.SetMapper(poly_mapper)
        actor.GetProperty().BackfaceCullingOn()
        actor.GetProperty().SetInterpolationToPhong()

        actor.GetProperty().SetAmbient(light[0])  # .3
        actor.GetProperty().SetDiffuse(light[1])  # .3
        actor.GetProperty().SetSpecular(light[2])  # .3
        return actor

    def display(self, display_name="trimesh", size=(1000, 800),
                light=(0.2, 0.15, 0.05), background=(0.0, 0.0, 0.0),
                png_magnify=1, display_colormap="Range",
                camera_rot=(0.0, 0.0, 0.0), zoom=1.0):
        # from dipy.fvtk
        renderer = vtk.vtkRenderer()
        actor = self.get_vtk_actor(light=light)
        renderer.AddActor(actor)
        renderer.ResetCamera()
        renderer.SetBackground(background)

        camera = renderer.GetActiveCamera()
        camera.Roll(camera_rot[0])
        camera.Elevation(camera_rot[1])
        camera.Azimuth(camera_rot[2])
        camera.Zoom(zoom)

        if self.__polydata_color_is_scalars__ and display_colormap:
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetTitle(display_colormap)
            scalar_bar.SetLookupTable(self.get_colormap())
            scalar_bar.SetNumberOfLabels(7)
            renderer.AddActor(scalar_bar)

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

                render_large = vtk.vtkRenderLargeImage()
                if vtk.VTK_MAJOR_VERSION <= 5:
                    render_large.SetInput(renderer)
                else:
                    render_large.SetInputData(renderer)

                render_large.SetMagnification(png_magnify)
                render_large.Update()
                writer = vtk.vtkPNGWriter()
                writer.SetInputConnection(render_large.GetOutputPort())
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

    def save_png(self, file_name, size=(1000, 800),
                 light=(0.2, 0.15, 0.05), background=(0.0, 0.0, 0.0),
                 display_colormap="Range", camera_rot=(0.0, 0.0, 0.0),
                 zoom=1.0, offscreen=True):
        self.update()
        renderer = vtk.vtkRenderer()
        actor = self.get_vtk_actor(light=light)
        renderer.AddActor(actor)
        renderer.ResetCamera()
        renderer.SetBackground(background)

        camera = renderer.GetActiveCamera()
        camera.Roll(camera_rot[0])
        camera.Elevation(camera_rot[1])
        camera.Azimuth(camera_rot[2])
        camera.Zoom(zoom)

        if self.__polydata_color_is_scalars__ and display_colormap:
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetTitle(display_colormap)
            scalar_bar.SetLookupTable(self.get_colormap())
            scalar_bar.SetNumberOfLabels(7)
            renderer.AddActor(scalar_bar)

        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(size[0], size[1])

        if offscreen:
            graphics_factory = vtk.vtkGraphicsFactory()
            graphics_factory.SetOffScreenOnlyMode(1)
            graphics_factory.SetUseMesaClasses(1)
            window.SetOffScreenRendering(1)

        window.Render()

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(window)
        window_to_image_filter.Update()

        if file_name is None:
            vtk_image = window_to_image_filter.GetOutput()
            h, w, _ = vtk_image.GetDimensions()
            vtk_array = vtk_image.GetPointData().GetScalars()
            components = vtk_array.GetNumberOfComponents()
            img_array = ns.vtk_to_numpy(vtk_array).reshape(h, w, components)
            return img_array

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(file_name)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    # Input for mapper or polydata
    def polydata_input(self, vtk_object):
        if vtk.VTK_MAJOR_VERSION <= 5:
            vtk_object.SetInput(self.get_polydata())
        else:
            vtk_object.SetInputData(self.get_polydata())
        vtk_object.Update()
        return vtk_object

    def save(self, file_name):
        self.update()
        vtk_u.save_polydata(self.__polydata__, file_name)
