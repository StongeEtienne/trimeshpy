###########################################################
#   TriMeshPy
###     Triangular Mesh Processing in Python
###     with SciPy sparse matrix representation
###         
###         by Etienne St-Onge
###########################################################

# README #
Startup of a simple and efficient triangle mesh processing library in python
using scipy sparse matrix and numpy math functions

# Paper #
St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M., 2017. Surface-enhanced tractography (SET). NeuroImage.


# TODO Code #
```
1) Modular "TriMesh_Class" that contain all triangle_mesh_processing functions
2) Python "UnitTest" in each module, in a standart way (not only a test file with "print")
3) add comments for each functions
4) Link graphical library (VTK and/or pyOpenGL) for visualisation (maybe in a class ex. TriMesh_VTK,  TriMesh_OGL
5) html, javascript graphical library (WebGL or Three) for visualisationwith interaction on webpage / Ipython notebook
6) GPU programming for sparse matrix (Theano or other)
```

# TODO Algo #
```
1) "No free Lunch" Laplacian operator
2) Multiresolution Mesh, Fuse zero-area triangles (maybe edge collapsing method)
```
