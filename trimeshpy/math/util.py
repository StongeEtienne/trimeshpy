# Etienne St-Onge

from __future__ import division

import logging
import tqdm
import sys

import numpy as np
import scipy
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import spsolve

from trimeshpy.math.mesh_global import G_ATOL


# ###################################################################
# Generic Math Functions
def square_length(vectors, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    return np.sum(np.square(vectors), axis, keepdims=keepdims)


def length(vectors, axis=1, keepdims=False):
    return np.sqrt(square_length(vectors, axis, keepdims))


def normalize_vectors(vectors, axis=1, safe_divide=True):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    lengths = length(vectors, axis, True)
    if safe_divide:
        non_zero = np.squeeze(lengths) > G_ATOL
        vectors[non_zero] / lengths[non_zero]
    else:
        vectors /= lengths
    return vectors


def dot(vectors1, vectors2, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors1).__module__:
        vectors1 = vectors1.toarray()
    if scipy.sparse.__name__ in type(vectors2).__module__:
        vectors2 = vectors2.toarray()
    return np.sum(vectors1 * vectors2, axis, keepdims=keepdims)


# Dot computation
# Discrete Differential-Geometry Operatorsfor Triangulated 2-Manifolds
#  by Mark Meyer, Mathieu Desbrun, Peter Schroder, Alan H. Barr
# http://multires.caltech.edu/pubs/diffGeoOps.pdf
def dot_area(u, v):
    sqr_norms = square_length(u) * square_length(v)
    u_dot_v = dot(u, v)
    return 0.5 * np.sqrt(sqr_norms - np.square(u_dot_v))


def dot_cos_angle(u, v):
    sqr_norms = square_length(u) * square_length(v)
    u_dot_v = dot(u, v)
    return u_dot_v/np.sqrt(sqr_norms)


def dot_sin_angle(u, v):
    return np.sqrt(1.0 - np.square(dot_cos_angle(u, v)))


def dot_cotan_angle(u, v):
    cos_u_v = dot_cos_angle(u, v)
    sin_u_v = np.sqrt(1.0 - np.square(cos_u_v))
    return cos_u_v/sin_u_v


def dot_angle(u, v):
    # Try to only use if needed
    return np.arccos(dot_cos_angle(u, v))


# Metric Tensor function
def tensor_dot(vectors1, vectors2, metric, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors1).__module__:
        vectors1 = vectors1.toarray()
    if scipy.sparse.__name__ in type(vectors2).__module__:
        vectors2 = vectors2.toarray()
    return np.sum(np.inner(vectors1, metric)
                  * vectors2, axis, keepdims=keepdims)


def tensor_area(u, v, metric):
    sqr_norms = tensor_dot(u, u, metric) * tensor_dot(v, v, metric)
    u_dot_v = tensor_dot(u, v, metric)
    return 0.5 * np.sqrt(sqr_norms - np.square(u_dot_v))


def tensor_cos_angle(u, v, metric):
    sqr_norms = tensor_dot(u, u, metric) * tensor_dot(v, v, metric)
    u_dot_v = tensor_dot(u, v, metric)
    return u_dot_v/np.sqrt(sqr_norms)


def tensor_sin_angle(u, v, metric):
    return np.sqrt(1.0 - np.square(tensor_cos_angle(u, v, metric)))


def tensor_cotan_angle(u, v, metric):
    cos_u_v = tensor_cos_angle(u, v, metric)
    sin_u_v = np.sqrt(1.0 - np.square(cos_u_v))
    return cos_u_v/sin_u_v


def tensor_angle(u, v, metric):
    # Try to only use if needed
    return np.arccos(tensor_cos_angle(u, v, metric))


# Step method
def euler_step(D_matrix, b_matrix, diffusion_step, backward_step=False):
    if backward_step:
        return backward_euler_step(D_matrix, b_matrix, diffusion_step)
    else:
        return forward_euler_step(D_matrix, b_matrix, diffusion_step)


# matrix need to be csc_matrix (not float128)
def forward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : x = ( I + d*D )b  <=> x = Af*b
    Af_matrix = csc_matrix(
        identity(b_matrix.shape[0])) + diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = Af_matrix.dot(b_matrix)
    return x_matrix


# matrix need to be csc_matrix (not float128)
def backward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : ( I - d*D )x = b  <=> Ab*x = b
    Ab_matrix = csc_matrix(
        identity(b_matrix.shape[0])) - diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = spsolve(Ab_matrix, b_matrix)
    return x_matrix


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def softplus(values):
    return np.log(1.0 + np.exp(values))


# python 2-3 support
def get_integer_types():
    if sys.version_info < (3,):
        return (int, long,)
    else:
        return (int,)


def get_numeric_types():
    return get_integer_types() + (float,)


def is_numeric(value):
    return isinstance(value,  get_numeric_types())


# Utils for test and print
def allclose_to(array, value):
    return np.allclose(np.squeeze(np.asarray(array)), value, atol=G_ATOL)


def is_logging_in_debug():
    return logging.getLogger().getEffectiveLevel() == logging.DEBUG


def logging_trange(nb_iter, desc=None):
    return tqdm.trange(nb_iter, desc=desc, disable=(not is_logging_in_debug()))
