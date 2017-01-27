# by Etienne.St-Onge@usherbrooke.ca

import numpy as np  # numerical python
import trimeshpy.math as tmath # python triangular mesh processing
# python triangular mesh processing

################################################################################
# Most of these Mesh operation come from :
# -Discrete Differential-Geometry Operators for Triangulated 2-Manifolds :
#     by Mark Meyer[1], Mathieu Desbrun[1,2], Peter Schroder[1] and Alan H. Barr[1]; [1]Caltech, [2]USC
# -Geometric Signal Processing on Polygonal Meshes
################################################################################

################################################################################
# TODO
# Decorator to use all these function as static method with inputs
# Where you always need to give (self.get_triangles, self.get_vertices()) and with option (dtype, atol)
# where : my_adjancency_mtx = trimesh.TriMesh(my_triangles, my_vertices).vertex_vertex_map()
# become: my_adjancency_mtx = trimesh.vertex_vertex_map(my_triangles, my_vertices)
################################################################################
class TriMesh(object):
    """
     Triangle Mesh class
        fast python mesh processing
        with numpy and scipy

    Triangle list
    Vertex list
    """

    # Init and test arguments
    def __init__(self, triangles, vertices, dtype=np.float64, atol=1e-8, assert_args=True):
        if assert_args:
            self._assert_init_args_(triangles, vertices, dtype, atol)
            # self._test_mesh_(self.get_triangles, self.get_vertices())

        # Never use private variable
        # Always use Get and Set!
        self.__dtype__ = dtype
        self.__atol__ = atol
        self.set_triangles(triangles)
        self.set_vertices(vertices.astype(dtype))

    def _assert_init_args_(self, triangles, vertices, dtype, atol):
        self._assert_triangles_(triangles)
        self._assert_vertices_(vertices)
        self._assert_dtype_(dtype)
        self._assert_atol_(atol, dtype)

    def _assert_triangles_(self, triangles):
        # test "triangles" arguments, type and shape
        assert type(triangles).__module__ == np.__name__, \
            "triangles should be a numpy array, not: %r" % type(triangles)
        assert(np.issubdtype(triangles.dtype, np.integer)), \
            "triangles should be an integer(index), not: %r" % triangles.dtype
        assert(triangles.shape[1] == 3), \
            "each triangle should have 3 points, not: %r" % triangles.shape[1]
        assert(triangles.ndim == 2), \
            "triangles array should only have 2 dimension, not: %r" % triangles.ndim

    def _assert_vertices_(self, vertices):
        # test "vertices" arguments, type and shape
        assert(type(vertices).__module__ == np.__name__), \
            "vertices should be a numpy array, not: %r" % type(vertices)
        assert(np.issubdtype(vertices.dtype, np.floating) or np.issubdtype(vertices.dtype, np.integer)), \
            "vertices should be number(float or integer), not: %r" % type(vertices)
        assert(vertices.shape[1] == 3), \
            "each vertex should be 3 dimensional, not: %r" % vertices.shape[1]
        assert(vertices.ndim == 2), \
            "vertices array should only have 2 dimension, not: %r" % vertices.ndim

    def _assert_dtype_(self, dtype):
        assert(np.issubdtype(dtype, np.floating)), \
            "dtype should be a float, not: %r" % dtype

    def _assert_atol_(self, atol, dtype):
        assert(np.issubdtype(type(atol), np.floating)), \
            "dtype should be a float, not: %r" % type(atol)
        assert(atol > np.finfo(dtype).eps), \
            "atol should be bigger than dtype machine epsilon: %r !> %r " % (atol, np.finfo(dtype).eps)

    def _test_mesh_(self, triangles, vertices):
        raise NotImplementedError()

    # Get class variable
    def get_triangles(self):
        return self.__triangles__

    def get_nb_triangles(self):
        return self.__nb_triangles__

    def get_nb_vertices(self):
        return self.__nb_vertices__

    def get_vertices(self):
        return self.__vertices__

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

    # Points Transformations Functions
    def vertices_translation(self, translation):
        return tmath.vertices_translation(self.get_triangles(), self.get_vertices(), translation)

    def vertices_rotation(self, rotation):
        return tmath.vertices_rotation(self.get_triangles(), self.get_vertices(), rotation)

    def vertices_transformation(self, transfo):
        return tmath.vertices_transformation(self.get_triangles(), self.get_vertices(), transfo)

    def flip_triangle_and_vertices(self, flip=[1, 1, 1]):
        return tmath.flip_triangle_and_vertices(self.get_triangles(), self.get_vertices(), flip)
    
    def vertices_flip(self, flip=[1, 1, 1]):
        return tmath.vertices_flip(self.get_triangles(), self.get_vertices(), flip)

    def triangles_face_flip(self, flip=[0, 2]):
        return tmath.triangles_face_flip(self.get_triangles(), self.get_vertices(), flip)

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
    def triangle_angle(self):
        return tmath.triangle_angle(self.get_triangles(), self.get_vertices())

    def triangle_is_obtuse(self):
        return tmath.triangle_is_obtuse(self.get_triangles(), self.get_vertices())

    def triangle_is_acute(self):
        return tmath.triangle_is_acute(self.get_triangles(), self.get_vertices())

    def triangle_is_right(self):
        return tmath.triangle_is_right(self.get_triangles(), self.get_vertices())

    def edge_theta_angle(self):
        return tmath.edge_theta_angle(self.get_triangles(), self.get_vertices())

    def edge_alpha_angle(self):
        return tmath.edge_alpha_angle(self.get_triangles(), self.get_vertices())

    def edge_gamma_angle(self):
        return tmath.edge_gamma_angle(self.get_triangles(), self.get_vertices())

    def cotan_alpha_beta_angle(self):
        return tmath.cotan_alpha_beta_angle(self.get_triangles(), self.get_vertices())

    def edge_triangle_is_obtuse(self):
        return tmath.edge_triangle_is_obtuse(self.get_triangles(), self.get_vertices())

    def edge_triangle_is_acute(self):
        return tmath.edge_triangle_is_acute(self.get_triangles(), self.get_vertices())

    def edge_theta_is_obtuse(self):
        return tmath.edge_theta_is_obtuse(self.get_triangles(), self.get_vertices())

    def edge_theta_is_acute(self):
        return tmath.edge_theta_is_acute(self.get_triangles(), self.get_vertices())
    
    def edge_triangle_normal_angle(self):
        return tmath.edge_triangle_normal_angle(self.get_triangles(), self.get_vertices())

    # Area Functions
    def triangles_area(self):
        return tmath.triangles_area(self.get_triangles(), self.get_vertices())

    def vertices_area(self, normalize=False):
        return tmath.vertices_area(self.get_triangles(), self.get_vertices(), normalize=normalize)

    def vertices_voronoi_area(self):
        return tmath.vertices_voronoi_area(self.get_triangles(), self.get_vertices())

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

    def positive_mass_stiffness_smooth(self, nb_iter=1, diffusion_step=1.0, flow_file=None):
        return tmath.positive_mass_stiffness_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, flow_file=flow_file)

    def volume_mass_stiffness_smooth(self, nb_iter=1, diffusion_step=1.0, flow_file=None):
        return tmath.volume_mass_stiffness_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, flow_file=flow_file)

    def gaussian_curv_smooth(self, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
        return tmath.gaussian_curv_smooth(self.get_triangles(), self.get_vertices(), nb_iter=nb_iter, diffusion_step=diffusion_step, area_weighted=area_weighted, backward_step=backward_step, flow_file=flow_file)
