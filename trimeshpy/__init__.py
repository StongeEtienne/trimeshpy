###########################################################
#   TriMeshPy
###     for Triangular Mesh Processing in Python
###     with SciPy sparse matrix representation
###         
###         by Etienne.St-Onge@usherbrooke.ca
###########################################################


from trimesh_class import TriMesh
from trimeshflow_class import TriMeshFlow
from trimesh_vtk import TriMesh_Vtk
from trimeshflow_vtk import TriMeshFlow_Vtk

import trimeshpy.math

__all__ = ["trimesh_class", "trimeshflow_class", "trimesh_vtk", "trimeshflow_vtk", "math"]


try:
    import trimeshpy_data as data
except:
    print("cannot load 'trimeshpy_data'")
