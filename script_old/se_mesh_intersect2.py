#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
import vtk
import numpy as np
from sys import stdout

from trimeshpy.trimesh_vtk import TriMesh_Vtk, save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.trimesh_vtk import load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='FS Surface tractography')
parser.add_argument('rh_tracking', type=str, default=None, help='right tractography input (.fib)')
parser.add_argument('lh_tracking', type=str, default=None, help='left tractography input (.fib)')
parser.add_argument('rh_surface', type=str, default=None, help='input right surface (WM)')
parser.add_argument('lh_surface', type=str, default=None, help='input left surface (WM)')
parser.add_argument('rh_out_surface', type=str, default=None, help='input right outside surface (GM)')
parser.add_argument('lh_out_surface', type=str, default=None, help='input left outside surface (GM)')
parser.add_argument('rh_mask', type=str, default=None, help='input right surface mask array')
parser.add_argument('lh_mask', type=str, default=None, help='input left surface mask array')
parser.add_argument('output', type=str, default=None, help='output right tracking result .fib (VTK FORMAT)')

parser.add_argument('intersection', type=str, default=None, help='output points direction where flow stopped')
parser.add_argument('surf_idx_inter', type=str, default=None, help='output surface index for intersection')

# option
parser.add_argument('-nuclei', nargs='+' , default=None, help='input nuclei surface (hard stop)')
parser.add_argument('-nuclei_soft', nargs='+' , default=None, help='input nuclei surface (soft stop)')
parser.add_argument('-report', type=str, default=None, help='output intersection report')

args = parser.parse_args()

print "Read .vtk surface file"
rh_mesh = TriMesh_Vtk(args.rh_surface, None)
lh_mesh = TriMesh_Vtk(args.lh_surface, None)

print "Generate OBB-Tree"
#tree = vtk.vtkModifiedBSPTree()
rh_tree = vtk.vtkOBBTree()
rh_tree.SetDataSet(rh_mesh.get_polydata())
rh_tree.BuildLocator()

lh_tree = vtk.vtkOBBTree()
lh_tree.SetDataSet(lh_mesh.get_polydata())
lh_tree.BuildLocator()
wm_trees = [rh_tree, lh_tree]

print "Read out_surface .vtk surface file and OBB-Tree"
rh_out_mesh = TriMesh_Vtk(args.rh_out_surface, None)
rh_out_tree = vtk.vtkOBBTree()
rh_out_tree.SetDataSet(rh_out_mesh.get_polydata())
rh_out_tree.BuildLocator()

lh_out_mesh = TriMesh_Vtk(args.lh_out_surface, None)
lh_out_tree = vtk.vtkOBBTree()
lh_out_tree.SetDataSet(lh_out_mesh.get_polydata())
lh_out_tree.BuildLocator()
gm_trees = [rh_out_tree, lh_out_tree]

# triangle and vertices mask
tris = [rh_mesh.get_triangles(), lh_mesh.get_triangles(), rh_mesh.get_triangles(), lh_mesh.get_triangles()]
rh_mask = np.load(args.rh_mask)
lh_mask = np.load(args.lh_mask)
masks = [rh_mask, lh_mask, rh_mask, lh_mask]

surf_name = [args.rh_surface, args.rh_surface, args.rh_out_surface, args.rh_out_surface ]
wm_index = [0,1]
gm_index = [2,3]
print "Read hard stop nuclei .vtk surface file and OBB-Tree"
if args.nuclei is not None :
    nu_index = range(len(surf_name), len(args.nuclei) + len(surf_name))
    nu_trees = []
    for nucleus in args.nuclei:
        surf_name.append(nucleus)
        nu_mesh = TriMesh_Vtk(nucleus, None)
        nu_tree = vtk.vtkOBBTree()
        nu_tree.SetDataSet(nu_mesh.get_polydata())
        nu_tree.BuildLocator()
        nu_trees.append(nu_tree)
else:
    nu_index = []
    nu_trees = []
    
print "Read soft stop nuclei .vtk surface file and OBB-Tree"
if args.nuclei_soft is not None :
    nusoft_index = range(len(surf_name), len(args.nuclei_soft) + len(surf_name))
    nusoft_trees = []
    for soft_nucleus in args.nuclei_soft:
        surf_name.append(soft_nucleus)
        nusoft__mesh = TriMesh_Vtk(soft_nucleus, None)
        nusoft_tree = vtk.vtkOBBTree()
        nusoft_tree.SetDataSet(nusoft__mesh.get_polydata())
        nusoft_tree.BuildLocator()
        nusoft_trees.append(nusoft_tree)
else:
    nusoft_index = []
    nusoft_trees = []
    
print "Load .fib fiber"
rh_lines = get_streamlines(load_streamlines_poyldata(args.rh_tracking))
lh_lines = get_streamlines(load_streamlines_poyldata(args.lh_tracking))
lines = rh_lines + lh_lines
nb_lines = len(lines)

# report info [rh, lh]
start_count = nb_lines
bad_start_count = 0
bad_pft_count = 0
end_count = 0
swap_end_count = 0
bad_end_count = 0

# id List (intersection list)
id_list =  -np.ones([nb_lines, 2], dtype=np.longlong)
id_surf =  -np.ones([nb_lines, 2], dtype=np.short)

# starting index 
rh_nonzero = np.nonzero(rh_mask)[0]
lh_nonzero = np.nonzero(lh_mask)[0]
rh_len_nonzero = len(rh_nonzero)

print nb_lines, rh_len_nonzero + len(lh_nonzero)
if nb_lines == rh_len_nonzero + len(lh_nonzero):
    id_list[:rh_len_nonzero,0] = rh_nonzero
    id_surf[:rh_len_nonzero,0] = 0
    id_list[rh_len_nonzero:,0] = lh_nonzero
    id_surf[rh_len_nonzero:,0] = 1
else: # if test
    mid_nb = nb_lines//2
    id_list[:mid_nb,0] = rh_nonzero[:mid_nb]
    id_surf[:mid_nb,0] = 0
    id_list[mid_nb:,0] = lh_nonzero[:mid_nb]
    id_surf[mid_nb:,0] = 1


# Utility fonction to test intersection
def pt_in_trees(pt0, trees):
    is_pt_in = False
    for tree in trees:
        if tree.InsideOrOutside(pt0) == -1:
            is_pt_in = True
    return is_pt_in

def seg_intersect_trees(pt0, pt1, trees, index_list=None): 
    seg_intersections = []
    for i in range(len(trees)):
        intersect = seg_intersect_tree(pt0, pt1, trees[i])
        if intersect is not None:
            if index_list is None:
                intersect.insert(0, i)
            else:
                intersect.insert(0, index_list[i])
            seg_intersections.append(intersect)
    return seg_intersections

def seg_intersect_tree(pt0, pt1, tree): 
    pts = vtk.vtkPoints()
    idList = vtk.vtkIdList()
    p0_in_out = tree.IntersectWithLine(pt0, pt1, pts, idList)
    if pts.GetNumberOfPoints() > 0:
        return [p0_in_out, pts.GetPoint(0), idList.GetId(0)]
    else:
        return None
    
def intersect_sqr_dist(pt0, intersections):
    sqr_dists = []
    for i in range(len(intersections)):
        sqr_dist = np.sum((pt0 - np.array(intersections[i][2]))**2)
        sqr_dists.append(sqr_dist)
    return sqr_dists

def sort_intersections(pt0, intersections): #todo sort
    dist = intersect_sqr_dist(pt0, intersections)
    raise NotImplementedError()

def is_intersections_in_mask( intersection, mask, triangles):
    #np.all(mask[rh_lh][tris[rh_lh][idList.GetId(0)]])
    tri = intersection[3]
    vts_idx = triangles[tri]
    return not(np.all(mask[vts_idx]))

def intersections_in_mask( intersections, mask, triangles, mask_index):
    intersections[:] = [x for x in intersections if not (x[0] in mask_index and is_intersections_in_mask(x, mask[x[0]], triangles[x[0]]))]
    return intersections

print "VTK Calculating first intersection"
print "and Generate new streamlines"
new_lines = []
print nb_lines
for i in range(nb_lines):
    stdout.write("\r %d%%" % (i*101//len(id_list)))
    stdout.flush()
    
    line = lines[i]
    #test if lines have at least 3 pts
    if len(line) < 2:
        id_list[i,0] = -1
        new_line = np.tile(line[0],(2,1))
        bad_pft_count += 1
    #test if first point in GM
    elif pt_in_trees(line[1], gm_trees) is False:
        id_list[i,0] = -1
        new_line = np.tile(line[0],(2,1))
        bad_start_count += 1
    else:
        # find the first intersection
        last_pt_in = None
        intersection_info = None
        for pt_index in range(1,len(line)-1):
            pt0 = line[pt_index]
            pt1 = line[pt_index + 1]
            
            # intersections
            
            # ROI (Hard stop)
            nu_inter = seg_intersect_trees(pt0, pt1, nu_trees, nu_index)
            if len(nu_inter) > 0:
                last_pt_in = pt_index
                intersection_info = nu_inter[0]
                #print "LINE STOPED IN ROI"
                break
            
            # ROI (Soft intersection)
            nusoft_inter = seg_intersect_trees(pt0, pt1, nusoft_trees, nusoft_index)
            for inter in nusoft_inter:
                if inter[1] == 1:
                    last_pt_in = pt_index
                    intersection_info = inter
                    #print "LINE TOUCHED SOFT ROI"
            
            # White matter (Soft intersection)
            wm_inter = seg_intersect_trees(pt0, pt1, wm_trees, wm_index)
            wm_inter = intersections_in_mask(wm_inter, masks, tris, mask_index=[0,1])
            #intersections = wm_inter + gm_inter + nu_inter
            for inter in wm_inter:
                if inter[1] == -1:
                    last_pt_in = pt_index
                    intersection_info = inter
                    #print "LINE TOUCHED WM"
                   
            # Grey matter (Hard stop)
            gm_inter = seg_intersect_trees(pt0, pt1, gm_trees, gm_index)
            gm_inter = intersections_in_mask(gm_inter, masks, tris, mask_index=[2,3])
            if len(gm_inter) > 0:
                #print "LINE STOPED at GM"
                break
                    
        # generate new line
        if last_pt_in is None:
            id_surf[i,1] = -1
            id_list[i,1] = -1
            new_line = np.tile(line[0],(2,1))
            bad_end_count += 1
        else:
            new_line = line[:last_pt_in+1]
            id_surf[i,1] = intersection_info[0]
            id_list[i,1] = intersection_info[3]
            new_line[last_pt_in] = intersection_info[2]
            end_count += 1
            
    new_lines.append(new_line)

stdout.write("\n done")
stdout.flush()

print "save intersection"
np.save(args.intersection, id_list)
np.save(args.surf_idx_inter, id_surf)

print "save as .fib"
lines_polydata = lines_to_vtk_polydata(new_lines, None, np.float32)
save_polydata(lines_polydata, args.output , True)

print "Do something with intersection !"

# Report
valid_count = start_count - bad_start_count - bad_end_count - bad_pft_count

if args.report is not None:
    report = open(args.report,"w")
    report.write("# Surface Order: " +  str(surf_name) + "\n")
    report.write("start_count: " + str(start_count) + "\n")
    report.write("bad_pft_count: " + str(bad_pft_count) + "\n")
    report.write("bad_start_count: " + str(bad_start_count) + "\n")
    report.write("end_count: " + str(end_count) + "\n")
    report.write("bad_end_count: " + str(bad_end_count) + "\n") 
    report.write("valid_count: " + str(valid_count) + "\n") 
    report.close() 
else:
    print "Surface Order: " +  str(surf_name)
    print "start_count: " + str(start_count)
    print "bad_pft_count: " + str(bad_pft_count)
    print "bad_start_count: " + str(bad_start_count)
    print "end_count: " + str(end_count)
    print "bad_end_count: " + str(bad_end_count)
    print "valid_count: " + str(valid_count)
