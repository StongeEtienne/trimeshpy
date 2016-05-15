# Etienne.St-Onge@usherbrooke.ca

import theano
import theano.numpy as np
# import scipy

from theano.sparse.linalg import spsolve
from theano.sparse import csc_matrix
import theano.sparse.diag as diags
import theano.sparse.identity

# to debug
from theano.sparse import coo_matrix  
from theano.sparse.linalg import inv