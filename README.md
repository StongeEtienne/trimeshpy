###########################################################
#    TriMeshPy
###        for Triangular Mesh Processing in Python
###        with SciPy sparse matrix representation
###
###########################################################

# README #
Startup of a simple and efficient triangle mesh processing library in python
using scipy sparse matrix and numpy math functions

# TODO Code #
```
1) Separate "processing.py" in different relevent file (Connection map, Angles, Area ...) 
2) Modular "TriMesh_Class" that contain all triangle_mesh_processing functions
3) Python "UnitTest" in each module, in a standart way (not only a test file with "print")
4) add comments for each functions
5) Fix the __init__.py, include class, package and tests
6) Link graphical library (VTK and/or pyOpenGL) for visualisation (maybe in a class ex. TriMesh_VTK,  TriMesh_OGL
7) html, javascript graphical library (WebGL or Three) for visualisationwith interaction on webpage / Ipython notebook
8) GPU programming for sparse matrix (Theano or other)
```

# TODO Algo #
```
1) "No free Lunch" Laplacian operator
2) Multiresolution Mesh, Fuse zero-area triangles (maybe edge collapsing method)
```
