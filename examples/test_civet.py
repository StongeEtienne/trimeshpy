
from trimeshpy.trimesh_vtk import TriMesh_Vtk

file_name = "../data/brain_mesh/100307.obj"
save_file = "../data/brain_mesh/100307.vtk"

mesh1 = TriMesh_Vtk(file_name, None)

mesh1.save(save_file)