# Etienne St-Onge

from __future__ import division

import h5py
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from trimeshpy.math.angle import edge_triangle_normal_angle
from trimeshpy.math.area import vertices_mix_area
from trimeshpy.math.mesh_map import edge_length, edge_adjacency
from trimeshpy.math.matrix import (laplacian, mass_matrix,
                                   mean_curvature_normal_matrix)
from trimeshpy.math.normal import vertices_normal
from trimeshpy.math.geo_diff import (vertices_cotan_normal,
                                     vertices_cotan_curvature,
                                     vertices_gaussian_curvature)
from trimeshpy.math.util import (dot, euler_step, normalize_vectors,
                                 is_numeric, logging_trange)

from trimeshpy.math.mesh_global import G_DTYPE, G_ATOL
from scipy.sparse.csc import csc_matrix


# Mesh smoothing (flow)
def laplacian_smooth(triangles, vertices, nb_iter=1, diffusion_step=1.0,
                     l2_dist_weighted=False, area_weighted=False,
                     backward_step=False, flow_file=None):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    vertices_csc = csc_matrix(vertices)

    for i in logging_trange(nb_iter, desc="laplacian_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices_csc.toarray()

        if l2_dist_weighted:
            # if l2_dist_weighted, we need to compute laplacian_matrix
            #   each iteration (because ||e_ij|| change)
            # A_ij_l2_dist_weighted = A_ij / ||e_ij||
            adjacency_matrix = edge_length(triangles, vertices_csc)
            # adjacency_matrix.data **= -1
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)
        else:
            adjacency_matrix = edge_adjacency(triangles, vertices_csc)
            laplacian_matrix = laplacian(adjacency_matrix, diag_of_1=True)

        if area_weighted:
            vts_mix_area = vertices_mix_area(triangles, vertices_csc)
            laplacian_matrix = diags(
                vts_mix_area ** -1, 0).dot(laplacian_matrix)

        next_vertices_csc = euler_step(
            laplacian_matrix, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    out_vts = vertices_csc.toarray()

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


def curvature_normal_smooth(triangles, vertices, nb_iter=1,
                            diffusion_step=1.0, area_weighted=False,
                            backward_step=False, flow_file=None):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    vertices_csc = csc_matrix(vertices)

    for i in logging_trange(nb_iter, desc="curvature_normal_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices_csc.toarray()

        # get curvature_normal_matrix
        curvature_normal_mtx = mean_curvature_normal_matrix(
            triangles, vertices_csc, area_weighted=area_weighted)

        next_vertices_csc = euler_step(
            curvature_normal_mtx, vertices_csc, diffusion_step, backward_step)
        vertices_csc = next_vertices_csc

    out_vts = vertices_csc.toarray()

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


# positive constrain curvature
def positive_curvature_normal_smooth(triangles, vertices, nb_iter=1,
                                     diffusion_step=1.0, area_weighted=False,
                                     backward_step=False, flow_file=None):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices, area_weighted=area_weighted)

    for i in logging_trange(nb_iter, desc="positive_curvature_normal_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices

        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, csc_matrix(
            vertices), diffusion_step, backward_step).toarray()
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_normal(triangles, next_vertices, normalize=False)
        pos_curv = dot(direction, normal_dir, keepdims=True) < 0
        vertices += direction * pos_curv

    out_vts = vertices

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


# positive weighted constrain curvature
def volume_curvature_normal_smooth(triangles, vertices, nb_iter=1,
                                   diffusion_step=1.0, area_weighted=False,
                                   backward_step=False, flow_file=None):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    for i in logging_trange(nb_iter, desc="volume_curvature_normal_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices

        # get curvature_normal_matrix
        # todo not optimal, because operation done twice etc
        vertices_csc = csc_matrix(vertices)
        curvature_normal_mtx = mean_curvature_normal_matrix(
            triangles, vertices, area_weighted=area_weighted)
        # do the first step
        next_vertices = euler_step(curvature_normal_mtx, vertices,
                                   diffusion_step, backward_step)
        # test if direction is positive
        direction = next_vertices - vertices
        normal_dir = vertices_cotan_normal(triangles, vertices, normalize=True)
        dotv = dot(normalize_vectors(direction), normal_dir, keepdims=True)
        vertices += direction * np.maximum(0.0, -dotv)

    out_vts = vertices

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


def mass_stiffness_smooth(triangles, vertices, nb_iter=1,
                          diffusion_step=1.0, flow_file=None):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices_csc, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)

    for i in logging_trange(nb_iter, desc="mass_stiffness_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices_csc.toarray()

        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices_csc).astype(np.float)

        # (D - d*L)*y = D*x = b
        A_matrix = mass_mtx - \
            (diags(diffusion_step, 0).dot(curvature_normal_mtx))
        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)

    out_vts = vertices_csc.toarray()

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


# positive mass_stiffness
def positive_mass_stiffness_smooth(triangles, vertices, nb_iter=1,
                                   diffusion_step=1.0, flow_file=None,
                                   gaussian_threshold=0.2, angle_threshold=2.0,
                                   subsample_file=1):

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)

        subsample_file = np.min((nb_iter, subsample_file))
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter//subsample_file + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices_csc, area_weighted=False)
    # mass_mtx = mass_matrix(triangles, vertices_csc)

    for i in logging_trange(nb_iter, desc="positive_mass_stiffness_smooth"):
        if flow_file is not None and i % subsample_file == 0:
            flow_data[i//subsample_file] = vertices_csc.toarray()

        mass_mtx = mass_matrix(triangles, vertices_csc)

        pos_curv = vertices_cotan_curvature(triangles, vertices_csc, False) > 0.0 #- G_ATOL

        if gaussian_threshold is not None:
            # Gaussian threshold: maximum value PI, cube corner = PI/2 # = 0.8
            deg_vts = np.abs(vertices_gaussian_curvature(triangles, vertices_csc, False)) > gaussian_threshold
            pos_curv = np.logical_or(pos_curv, deg_vts)

        if angle_threshold is not None:
            # angle_threshold: PI, cube corner = PI/2 # = 1.7
            deg_seg = edge_triangle_normal_angle(triangles, vertices_csc).max(1).toarray().squeeze() > angle_threshold
            pos_curv = np.logical_or(pos_curv, deg_seg)

        positive_diffusion_step = pos_curv * diffusion_step

        # (D - d*L)*y = D*x = b
        A_matrix = (mass_mtx - 
            diags(positive_diffusion_step, 0).dot(curvature_normal_mtx))

        b_matrix = mass_mtx.dot(vertices_csc)
        vertices_csc = spsolve(A_matrix, b_matrix)

    out_vts = vertices_csc.toarray()

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


# positive weighted mass_stiffness
def volume_mass_stiffness_smooth(triangles, vertices, nb_iter=1,
                                 diffusion_step=1.0, flow_file=None):

    raise NotImplementedError()

    if flow_file is not None:
        flow_h5py = h5py.File(flow_file, mode='w')
        flow_h5py.create_dataset("triangles", data=triangles)
        flow_data = flow_h5py.create_dataset(
            "vertices", (nb_iter + 1, vertices.shape[0], vertices.shape[1]),
            dtype=G_DTYPE, chunks=(1, vertices.shape[0], 3))

    if is_numeric(diffusion_step):
        diffusion_step = diffusion_step * np.ones(len(vertices), dtype=G_DTYPE)

    vertices_csc = csc_matrix(vertices)
    curvature_normal_mtx = mean_curvature_normal_matrix(
        triangles, vertices_csc, area_weighted=False)

    for i in logging_trange(nb_iter, desc="volume_mass_stiffness_smooth"):
        if flow_file is not None:
            flow_data[i] = vertices_csc.toarray()

        # get curvature_normal_matrix
        mass_mtx = mass_matrix(triangles, vertices)

        # (D - d*L)*y = D*x = b
        A_matrix = (mass_mtx - 
            diags(diffusion_step, 0).dot(curvature_normal_mtx))
        b_matrix = mass_mtx.dot(csc_matrix(vertices_csc))
        next_vertices = spsolve(A_matrix, b_matrix)
        # test if direction is positive
        direction = next_vertices.toarray() - vertices_csc
        normal_dir = vertices_cotan_normal(
            triangles, next_vertices, normalize=True)
        dotv = normalize_vectors(direction).multiply(normal_dir)
        vertices_csc += direction * np.maximum(0.0, -dotv)
        # vertices_csc += direction * sigmoid(-np.arctan(dotv)*np.pi - np.pi)
        # vertices_csc += direction * softplus(-dotv)

    out_vts = vertices_csc.toarray()

    if flow_file is not None:
        flow_data[-1] = out_vts
        flow_h5py.close()

    return out_vts


def gaussian_curv_smooth(triangles, vertices, nb_iter=1,
                         diffusion_step=1.0, area_weighted=False,
                         backward_step=False, flow_file=None):
    raise NotImplementedError()
