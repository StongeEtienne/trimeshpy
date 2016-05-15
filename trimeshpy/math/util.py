# Etienne.St-Onge@usherbrooke.ca

from trimeshpy.math import *

# ###################################################################
# Generic Functions
def square_length(vectors, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    return np.sum(np.square(vectors), axis, keepdims=keepdims)


def length(vectors, axis=1, keepdims=False):
    return np.sqrt(square_length(vectors, axis, keepdims))


def normalize_vectors(vectors, axis=1):
    if scipy.sparse.__name__ in type(vectors).__module__:
        vectors = vectors.toarray()
    return vectors / length(vectors, axis, True)


def dot(vectors1, vectors2, axis=1, keepdims=False):
    if scipy.sparse.__name__ in type(vectors1).__module__:
        vectors1 = vectors1.toarray()
    if scipy.sparse.__name__ in type(vectors2).__module__:
        vectors2 = vectors2.toarray()
    return np.sum(vectors1 * vectors2, axis, keepdims=keepdims)


# Step method
def euler_step(D_matrix, b_matrix, diffusion_step, backward_step=False):
    if backward_step:
        return backward_euler_step(D_matrix, b_matrix, diffusion_step)
    else:
        return forward_euler_step(D_matrix, b_matrix, diffusion_step)


# matrix need to be csc_matrix (not float128)
def forward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : x = ( I + d*D )b  <=> x = Af*b
    Af_matrix = csc_matrix(identity(b_matrix.shape[0])) + diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = Af_matrix.dot(b_matrix)
    return x_matrix


# matrix need to be csc_matrix (not float128)
def backward_euler_step(D_matrix, b_matrix, diffusion_step):
    # find 'x' where : ( I - d*D )x = b  <=> Ab*x = b
    Ab_matrix = csc_matrix(identity(b_matrix.shape[0])) - diags(diffusion_step, 0).dot(D_matrix)
    x_matrix = spsolve(Ab_matrix, b_matrix)
    return x_matrix

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))

def softplus(values):
    return np.log(1.0 + np.exp(values))
