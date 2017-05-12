# Etienne St-Onge

import numpy as np
import scipy.linalg.interpolative  
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import spsolve, LinearOperator
from sys import stdout

from trimeshpy.math.area import vertices_mix_area
from trimeshpy.math.angle import edge_triangle_normal_angle
from trimeshpy.math.curvature import vertices_cotan_curvature, vertices_gaussian_curvature
from trimeshpy.math.mesh_map import edge_length, edge_adjacency
from trimeshpy.math.matrix import laplacian, mass_matrix, mean_curvature_normal_matrix
from trimeshpy.math.normal import vertices_normal, vertices_cotan_normal
from trimeshpy.math.util import dot, euler_step, normalize_vectors

from trimeshpy.math.mesh_global import G_DTYPE, G_ATOL
from scipy.sparse.csc import csc_matrix

# Mesh smoothing (flow)
# and maybe it should be in another type of mesh, test all the rest before testing this
def laplacian_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, l2_dist_weighted=False, area_weighted=False, backward_step=False, flow_file=None):

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    vertices_csc = csc_matrix(vertices)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()

        if l2_dist_weighted:
            # if l2_dist_weighted, we need to compute laplacian_matrix each iteration (because ||e_ij|| change)
            # A_ij_l2_dist_weighted = A_ij / ||e_ij||
            adjacency_matrix = edge_length(triangles, vertices_csc)
            ####################################################################
            # adjacency_matrix.data **= -1
            # laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=False)
            ####################################################################
            adjacency_matrix.data **= 1  # 1
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)
        else:
            adjacency_matrix = edge_adjacency(triangles, vertices_csc)
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)

        if area_weighted:
            vts_mix_area = vertices_mix_area(triangles, vertices_csc)
            # laplacian_matrix = diags((vts_mix_area / vts_mix_area.min()) ** -1, 0).dot(laplacian_matrix)
            laplacian_matrix = diags(vts_mix_area ** -1, 0).dot(laplacian_matrix)

        next_vertices_csc = euler_step(laplacian_matrix, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return next_vertices_csc.toarray()


def curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    vertices_csc = csc_matrix(vertices)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()

        # get curvature_normal_matrix
        curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)

        next_vertices_csc = euler_step(curvature_normal_mtx, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return vertices_csc.toarray()


# positive constrain curvature
def positive_curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))
        
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, csc_matrix(vertices), diffusion_step, backward_step).toarray()
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_normal(triangles, next_vertices, normalize=False)
        pos_curv = dot(direction, normal_dir, keepdims=True) < 0
        vertices += direction * pos_curv

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices


# positive weighted constrain curvature
def volume_curvature_normal_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
        
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    # curvature_normal_mtx_start = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=area_weighted)
    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices
        # get curvature_normal_matrix
        # todo not optimal, because operation done twice etc
        curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=area_weighted)
        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, csc_matrix(vertices), diffusion_step, backward_step).toarray()
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_cotan_normal(triangles, vertices, normalize=True)
        dotv = dot(normalize_vectors(direction), normal_dir, keepdims=True)
        vertices += direction * np.maximum(0.0, -dotv)

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices


def mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
    
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()
        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)

        """ # forward step: Dy = (D+d*L)x = b
        b_matrix = (mass_mtx + diags(diffusion_step,0).dot(curvature_normal_mtx)).dot(vertices_csc)
        vertices_csc = spsolve(mass_mtx, b_matrix) - vertices_csc
        """
        
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - (diags(diffusion_step,0).dot(curvature_normal_mtx))
        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)
        
        

    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    # return next_vertices_csc
    return vertices_csc.toarray()


# positive mass_stiffness
def positive_mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))

    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.todense()
        
        # third try
        mass_mtx = mass_matrix(triangles, vertices_csc)
        # curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices_csc, area_weighted=False)
        
        # Dy = (D+d*L)x = b
        #b_matrix = (mass_mtx + diags(diffusion_step,0).dot(curvature_normal_mtx)).dot(vertices_csc)
        #y = spsolve(mass_mtx, b_matrix)
        #pos_curv = dot(y - vertices_csc, vertices_normal(triangles, vertices, False, False)) < 0
        
        pos_curv = vertices_cotan_curvature(triangles, vertices_csc, False) > - G_ATOL
        
        """# Gaussian fix
        GAUSSIAN_TRESHOLD = 0.2 # max_gauss: PI, cube corner = PI/2 # = 0.8
        deg_vts = np.abs(vertices_gaussian_curvature(triangles, vertices_csc, False)) > GAUSSIAN_TRESHOLD
        pos_curv = np.logical_or(pos_curv, deg_vts)
        
        # Sheet fix
        ANGLE_TRESHOLD = 1.0 # max_gauss: PI, cube corner = PI/2 # = 1.7
        deg_seg = edge_triangle_normal_angle(triangles, vertices_csc).max(1).toarray().squeeze() > ANGLE_TRESHOLD
        pos_curv = np.logical_or(pos_curv, deg_seg)
        #print " ", np.count_nonzero(deg_vts), " ", np.count_nonzero(deg_seg)"""
        
        possitive_diffusion_step = pos_curv * diffusion_step
        
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - (diags(possitive_diffusion_step,0).dot(curvature_normal_mtx))
        """A_matrix = identity(A_matrix.shape[0])
        test = LinearOperator( A_matrix.shape, matvec=A_matrix.dot, rmatvec=A_matrix.T.dot)
        print
        print "matrix shape", A_matrix.shape
        print "rank", scipy.linalg.interpolative.estimate_rank(test, 0.1)"""
        
        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)
        
        
    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices_csc.toarray()


# positive weighted mass_stiffness
def volume_mass_stiffness_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, flow_file=None):
    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(triangles, vertices, area_weighted=False)
    
    if isinstance(diffusion_step, (int, long, float)):
        diffusion_step = diffusion_step*np.ones(len(vertices))
    
    if flow_file is not None:
        mem_map = np.memmap(flow_file, dtype=G_DTYPE, mode='w+', shape=(nb_iter, vertices.shape[0], vertices.shape[1]))

    for i in range(nb_iter):
        stdout.write("\r step %d on %d done" % (i,  nb_iter))
        stdout.flush()
        if flow_file is not None:
            mem_map[i] = vertices_csc.toarray()
        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices)
        
        raise NotImplementedError()
        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - diags(diffusion_step,0).dot(curvature_normal_mtx)
        b_matrix = mass_mtx.dot(csc_matrix(vertices_csc))
        next_vertices = spsolve(A_matrix, b_matrix)
        # test if direction is positive
        direction = next_vertices.toarray() - vertices_csc
        normal_dir = vertices_cotan_normal(triangles, next_vertices, normalize=True)
        dotv = normalize_vectors(direction).multiply(normal_dir)
        vertices_csc += direction * np.maximum(0.0, -dotv)
        # vertices_csc += direction * sigmoid(-np.arctan(dotv)*np.pi - np.pi)
        # vertices_csc += direction * softplus(-dotv)
        
    stdout.write("\r step %d on %d done \n" % (nb_iter,  nb_iter))
    return vertices_csc.toarray()
    
    
def gaussian_curv_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0, area_weighted=False, backward_step=False, flow_file=None):
    raise NotImplementedError()
