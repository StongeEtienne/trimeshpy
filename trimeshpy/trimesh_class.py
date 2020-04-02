# Etienne St-Onge

"""
Triangular Mesh processing class

Most of these mesh operations are from :
    Meyer, M., Desbrun, M., Schroder, P. and Barr, A.H., 2003.
    Discrete differential-geometry operators for triangulated 2-manifolds.
    In Visualization and mathematics III, (pp. 35-57). Springer.
"""

import logging

import numpy as np

import trimeshpy.math as tmath


class TriMesh(object):
    """
     Triangle Mesh class
        fast python mesh processing
        with numpy and scipy

    Triangle list
    Vertex list
    """

    # Init and test arguments
    def __init__(self, triangles, vertices, dtype=np.float64,
                 atol=1e-8, assert_args=True):
        self.__dtype__ = dtype
        self.__atol__ = atol

        self.set_triangles(triangles)
        self.set_vertices(vertices.astype(dtype))
        
        if assert_args:
            self._assert_init_args_()

    def _assert_init_args_(self):
        self._assert_triangles_()
        self._assert_vertices_()
        self._assert_edges_()
        self._assert_dtype_()
        self._assert_atol_()

    def _assert_triangles_(self):
        # test "triangles" arguments, type and shape
        if type(self.__triangles__).__module__ != np.__name__:
            logging.error("triangles should be a numpy array, not: %r" % type(self.__triangles__))
        if not np.issubdtype(self.__triangles__.dtype, np.integer):
            logging.error("triangles should be an integer(index), not: %r" % self.__triangles__.dtype)
        if self.__triangles__.shape[1] != 3:
            logging.error("each triangle should have 3 points, not: %r" % self.__triangles__.shape[1])
        if self.__triangles__.ndim != 2:
            logging.error("triangles array should only have 2 dimension, not: %r" % self.__triangles__.ndim)
        if not np.issubdtype(self.__triangles__.dtype, np.integer):
            logging.error("triangles should be an integer(index), not: %r" % self.__triangles__.dtype)

    def _assert_vertices_(self):
        # test "vertices" arguments, type and shape
        if type(self.__vertices__).__module__ != np.__name__:
            logging.error("vertices should be a numpy array, not: %r" % type(self.__vertices__))
        if not (np.issubdtype(self.__vertices__.dtype, np.floating)
                or np.issubdtype(self.__vertices__.dtype, np.integer)):
            logging.error("vertices should be number(float or integer), not: %r" % type(self.__vertices__))
        if self.__vertices__.shape[1] != 3:
            logging.error("each vertex should be 3 dimensional, not: %r" % self.__vertices__.shape[1])
        if self.__vertices__.ndim != 2:
            logging.error("vertices array should only have 2 dimension, not: %r" % self.__vertices__.ndim)

    def _assert_edges_(self):
        e_sqr_length = (tmath.util.square_length(self.__vertices__[np.roll(self.__triangles__, 1, axis=1)]
                                                 - self.__vertices__[np.roll(self.__triangles__, -1, axis=1)], axis=2))
        if (e_sqr_length < self.__atol__).any():
            logging.error("triangles should not have zero length edges")

    def _assert_dtype_(self):
        if not np.issubdtype(self.__dtype__, np.floating):
            logging.error("dtype should be a float, not: %r" % self.__dtype__)

    def _assert_atol_(self):
        if not np.issubdtype(type(self.__atol__), np.floating):
            logging.error("dtype should be a float, not: %r" % type(self.__atol__))
        if self.__atol__ < np.finfo(self.__dtype__).eps:
            logging.error("atol should be bigger than dtype machine epsilon: %r !> %r " % (self.__atol__, np.finfo(self.__dtype__).eps))

    # Get class variable
    def get_nb_triangles(self):
        return self.__nb_triangles__

    def get_nb_vertices(self):
        return self.__nb_vertices__

    def get_vertices(self):
        return self.__vertices__

    def get_triangles(self):
        return self.__triangles__

    def get_atol(self):
        return self.__atol__

    def get_dtype(self):
        return self.__dtype__

    # Set class variable
    def set_triangles(self, triangles):
        self.__triangles__ = triangles
        self.__nb_triangles__ = len(triangles)

    def set_vertices(self, vertices):
        self.__vertices__ = vertices.astype(self.__dtype__)
        self.__nb_vertices__ = len(vertices)

    # Get Functions
    def nb_triangles_per_vertex(self):
        return self.vertices_degree()

    # Math functions
    # Points Transformations Functions
    def vertices_translation(self, translation):
        return tmath.vertices_translation(self.get_triangles(), self.get_vertices(), translation)

    def vertices_rotation(self, rotation):
        return tmath.vertices_rotation(self.get_triangles(), self.get_vertices(), rotation)

    def vertices_transformation(self, transfo):
        return tmath.vertices_transformation(self.get_triangles(), self.get_vertices(), transfo)

    def vertices_affine(self, affine):
        return tmath.vertices_affine(self.get_triangles(), self.get_vertices(), affine)

    def vertices_flip(self, flip=(1, 1, 1)):
        return tmath.vertices_flip(self.get_triangles(), self.get_vertices(), flip)

    def triangles_face_flip(self, flip=(0, 2)):
        return tmath.triangles_face_flip(self.get_triangles(), self.get_vertices(), flip)

    def flip_triangle_and_vertices(self, flip=(1, 1, 1)):
        return tmath.flip_triangle_and_vertices(self.get_triangles(), self.get_vertices(), flip)

    def is_transformation_flip(self, transfo):
        return tmath.is_transformation_flip(transfo)

    # Map ( Adjacency / Connectivity ) Functions
    def edge_map(self, l2_weighted=False):
        return tmath.edge_map(self.get_triangles(), self.get_vertices(), l2_weighted=l2_weighted)

    def triangle_vertex_map(self):
        return tmath.triangle_vertex_map(self.get_triangles(), self.get_vertices())

    def edge_triangle_map(self):
        return tmath.edge_triangle_map(self.get_triangles(), self.get_vertices())

    def edge_opposing_vertex(self):
        return tmath.edge_opposing_vertex(self.get_triangles(), self.get_vertices())

    def vertices_degree(self):
        return tmath.vertices_degree(self.get_triangles(), self.get_vertices())

    # Angles Functions
    def triangle_trigo_angle(self, angle_function):
        return tmath.triangle_trigo_angle(self.get_triangles(), self.get_vertices(), angle_function=angle_function)

    def edge_trigo_angle(self, rot, angle_function):
        return tmath.edge_trigo_angle(self.get_triangles(), self.get_vertices(), rot=rot, angle_function=angle_function)

    def triangle_dot_angle(self):
        return tmath.triangle_dot_angle(self.get_triangles(), self.get_vertices())

    def triangle_cos_angle(self):
        return tmath.triangle_cos_angle(self.get_triangles(), self.get_vertices())

    def triangle_sin_angle(self):
        return tmath.triangle_sin_angle(self.get_triangles(), self.get_vertices())

    def triangle_cotan_angle(self):
        return tmath.triangle_cotan_angle(self.get_triangles(), self.get_vertices())

    def triangle_is_obtuse(self):
        return tmath.triangle_is_obtuse(self.get_triangles(), self.get_vertices())

    def triangle_is_acute(self):
        return tmath.triangle_is_acute(self.get_triangles(), self.get_vertices())

    def triangle_is_right(self):
        return tmath.triangle_is_right(self.get_triangles(), self.get_vertices())

    def edge_cotan_map(self):
        return tmath.edge_cotan_map(self.get_triangles(), self.get_vertices())

    # Area Functions
    def triangles_area(self):
        return tmath.triangles_area(self.get_triangles(), self.get_vertices())

    def vertices_area(self, normalize=False):
        return tmath.vertices_area(self.get_triangles(), self.get_vertices(), normalize=normalize)

    def vertices_voronoi_area(self):
        return tmath.vertices_voronoi_area(self.get_triangles(), self.get_vertices())

    def vertices_mix_area(self):
        return tmath.vertices_mix_area(self.get_triangles(), self.get_vertices())

    def edge_area(self):
        return tmath.edge_area(self.get_triangles(), self.get_vertices())

    def edge_voronoi_area(self):
        return tmath.edge_voronoi_area(self.get_triangles(), self.get_vertices())

    def edge_mix_area(self):
        return tmath.edge_mix_area(self.get_triangles(), self.get_vertices())

    # Normals Functions
    def triangles_normal(self, normalize=True):
        return tmath.triangles_normal(self.get_triangles(), self.get_vertices(), normalize=normalize)

    def vertices_normal(self, normalize=True, area_weighted=True):
        return tmath.vertices_normal(self.get_triangles(), self.get_vertices(), normalize=normalize, area_weighted=area_weighted)

    def vertices_cotan_normal(self, normalize=True, area_weighted=True):
        return tmath.vertices_cotan_normal(self.get_triangles(), self.get_vertices(), normalize=normalize, area_weighted=area_weighted)

    def vertices_cotan_direction(self, normalize=True, area_weighted=True):
        return tmath.vertices_cotan_direction(self.get_triangles(), self.get_vertices(), normalize=normalize, area_weighted=area_weighted)

    # Mesh operation matrix
    def laplacian(self, adjacency_matrix, diag_of_1=True):
        return tmath.laplacian(adjacency_matrix, diag_of_1=diag_of_1)

    def mean_curvature_normal_matrix(self, area_weighted=False):
        return tmath.mean_curvature_normal_matrix(self.get_triangles(), self.get_vertices(), area_weighted=area_weighted)

    def mass_matrix(self):
        return tmath.mass_matrix(self.get_triangles(), self.get_vertices())

    # Curvature Functions
    def vertices_cotan_curvature(self, area_weighted=True):
        return tmath.vertices_cotan_curvature(self.get_triangles(), self.get_vertices(), area_weighted=area_weighted)

    def vertices_gaussian_curvature(self, area_weighted=False):
        return tmath.vertices_gaussian_curvature(self.get_triangles(), self.get_vertices(), area_weighted=area_weighted)

    # Mesh smoothing (flow)
    def laplacian_smooth(self, nb_iter=1, diffusion_step=1.0, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.laplacian_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, l2_dist_weighted=l2_dist_weighted, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)

    def curvature_normal_smooth(self, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.curvature_normal_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)

    def positive_curvature_normal_smooth(self, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.positive_curvature_normal_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)

    def volume_curvature_normal_smooth(self, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.volume_curvature_normal_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)

    def mass_stiffness_smooth(self, nb_iter=1, diffusion_step=1.0, flow_file=None):
        return tmath.mass_stiffness_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, flow_file=flow_file)

    def positive_mass_stiffness_smooth(self, nb_iter=1, diffusion_step=1.0, flow_file=None, gaussian_threshold=0.2, angle_threshold=1.0, subsample_file=1):
        return tmath.positive_mass_stiffness_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, flow_file=flow_file, gaussian_threshold=gaussian_threshold, angle_threshold=angle_threshold, subsample_file=subsample_file)

    def volume_mass_stiffness_smooth(self, nb_iter=1, diffusion_step=1.0, flow_file=None):
        return tmath.volume_mass_stiffness_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, flow_file=flow_file)

    def gaussian_curv_smooth(self, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.gaussian_curv_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)
