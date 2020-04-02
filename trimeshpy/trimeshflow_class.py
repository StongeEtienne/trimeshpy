# Etienne St-Onge

import logging
from six import string_types

import numpy as np

from trimeshpy.trimesh_class import TriMesh
import trimeshpy.vtk_util as vtk_u


# Flow triangles mesh
# composed of :
#   - triangles
#   - flow_vertices / displacement of vertices over time
#        -> initial_vertices = flow_vertices[0]
#        -> current_vertices = flow_vertices[-1]
class TriMeshFlow(TriMesh):

    # Init and test arguments
    def __init__(self, triangles, vertices_flow,
                 dtype=np.float64, atol=1e-8, assert_args=True):

        self.__dtype__ = dtype
        self.__atol__ = atol
        self.set_triangles(triangles)

        if isinstance(vertices_flow, string_types):
            self.set_vertices_flow_from_hdf5(vertices_flow)
        else:
            self.set_vertices_flow(vertices_flow)

        if assert_args:
            self._assert_init_args_()

    # Redefinition
    def _assert_vertices_(self):
        if type(self.__vertices_flow__).__module__ != np.__name__:
            logging.error("vertices_flow should be a numpy array, not: %r" % type(self.__vertices_flow__))
        if not (np.issubdtype(self.__vertices_flow__.dtype, np.floating)
                or np.issubdtype(self.__vertices_flow__.dtype, np.integer)):
            logging.error("vertices_flow should be number(float or integer), not: %r" % type(self.__vertices_flow__))
        if self.__vertices_flow__.shape[-1] != 3:
            logging.error("each vertex should be 3 dimensional, not: %r" % self.__vertices_flow__.shape[1])
        if self.__vertices_flow__.ndim != 2 and self.__vertices_flow__.ndim != 3:
            logging.error("vertices_flow array should only have 2 or 3 dimensions, not: %r" % self.__vertices_flow__.ndim)

    def _assert_edges_(self):
        return

# Get class variable
    def get_triangles(self):
        return self.__triangles__

    def get_nb_triangles(self):
        return self.__nb_triangles__

    def get_flow_length(self):
        return self.__flow_length__

    def get_nb_vertices(self):
        return self.__nb_vertices__

    def get_vertices_flow(self):
        return self.__vertices_flow__

    def get_initial_vertices(self):
        return self.__vertices_flow__[0]

    def get_current_vertices(self):
        return self.__vertices_flow__[-1]

    def get_vertices(self, vertices_flow_index=-1):
        # interface with TriMesh
        return self.__vertices_flow__[vertices_flow_index]

    def get_atol(self):
        return self.__atol__

    def get_dtype(self):
        return self.__dtype__

    # Set class variable
    def set_triangles(self, triangles):
        self.__triangles__ = triangles
        self.__nb_triangles__ = len(triangles)

    def set_vertices_flow(self, vertices_flow):
        if vertices_flow.ndim == 2:
            self.__vertices_flow__ = vertices_flow[
                np.newaxis].astype(self.__dtype__)
            self.__flow_length__ = 1
            self.__nb_vertices__ = len(vertices_flow)

        elif vertices_flow.ndim == 3:
            self.__vertices_flow__ = vertices_flow.astype(self.__dtype__)
            self.__flow_length__ = vertices_flow.shape[0]
            self.__nb_vertices__ = vertices_flow.shape[1]

    def set_vertices_flow_from_memmap(self, vertices_flow_memmap,
                                      flow_length, nb_vertices):
        self.__vertices_flow__ = np.array(np.memmap(
            vertices_flow_memmap, dtype=self.get_dtype(), mode='r',
            shape=(flow_length, nb_vertices, 3)))

    def set_vertices_flow_from_hdf5(self, vertices_flow_hdf5):
        self.__vertices_flow__ = vtk_u.load_hdf5_vertices(vertices_flow_hdf5)
        self.__flow_length__ = self.__vertices_flow__.shape[0]
        self.__nb_vertices__ = self.__vertices_flow__.shape[1]

    def set_vertices(self, vertices_flow):
        logging.error("For TriMeshFlow instances use 'set_vertices_flow()' instead ")
        raise NotImplementedError()
