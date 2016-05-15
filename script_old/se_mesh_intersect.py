#!/usr/bin/env python

# Etienne.St-Onge@usherbrooke.ca
import argparse
from sys import stdout
import vtk
import numpy as np

from trimeshpy.trimesh_vtk import TriMesh_Vtk, save_polydata
from trimeshpy.trimeshflow_vtk import lines_to_vtk_polydata
from trimeshpy.trimesh_vtk import load_streamlines_poyldata, get_streamlines

parser = argparse.ArgumentParser(description='FS Surface tractography')
parser.add_argument('rh_tracking', type=str, default=None, help='right tractography input (.fib)')
parser.add_argument('lh_tracking', type=str, default=None, help='left tractography input (.fib)')
parser.add_argument('rh_surface', type=str, default=None, help='input right surface')
parser.add_argument('lh_surface', type=str, default=None, help='input left surface')
parser.add_argument('output', type=str, default=None, help='output right tracking result .fib (VTK FORMAT)')

parser.add_argument('intersection', type=str, default=None, help='output points direction where flow stopped')
parser.add_argument('surf_idx_inter', type=str, default=None, help='output surface index for intersection')

parser.add_argument('-rh_mask', type=str, default=None, help='input right surface mask array')
parser.add_argument('-lh_mask', type=str, default=None, help='input left surface mask array')

# option
parser.add_argument('-report', type=str, default=None, help='output intersection report')

args = parser.parse_args()

rh_lines = get_streamlines(load_streamlines_poyldata(args.rh_tracking))
lh_lines = get_streamlines(load_streamlines_poyldata(args.lh_tracking))
lines = [rh_lines, lh_lines]

print "Read .vtk surface file"
rh_mesh = TriMesh_Vtk(args.rh_surface, None)
lh_mesh = TriMesh_Vtk(args.lh_surface, None)
tris = [rh_mesh.get_triangles(), lh_mesh.get_triangles()]
vts = [rh_mesh.get_vertices(), lh_mesh.get_vertices()]

print "Generate OBB-Tree"
#tree = vtk.vtkModifiedBSPTree()
rh_tree = vtk.vtkOBBTree()
rh_tree.SetDataSet(rh_mesh.get_polydata())
rh_tree.BuildLocator()

lh_tree = vtk.vtkOBBTree()
lh_tree.SetDataSet(lh_mesh.get_polydata())
lh_tree.BuildLocator()
tree = [rh_tree, lh_tree]


# report info [rh, lh]
start_count = [len(rh_lines), len(lh_lines)]
bad_start_count = [0, 0]
bad_pft_count = [0, 0]
end_count = [0, 0]
swap_end_count = [0, 0]
bad_end_count = [0, 0]

# MASK
masking = False
if args.rh_mask is not None and args.lh_mask is not None:
    masking = True
    mask = [np.load(args.rh_mask), np.load(args.lh_mask)]
    
    #nb_lines = np.count_nonzero(rh_mask) + np.count_nonzero(lh_mask)
    
nb_lines = len(rh_lines) + len(lh_lines)
    
# id List (intersection list)
id_list =  -np.ones([nb_lines, 2], dtype=np.longlong)
id_surf =  -np.ones([nb_lines, 2], dtype=np.short)

print "VTK Calculating first intersection"
print "and Generate new streamlines"
# rh=0, lh=1

if masking:
    # with mask
    new_lines = []
    i = 0
    for rh_lh in range(len(tree)):
        inv_rh_lh = 1 - rh_lh
        line_idx = 0
        for idx in np.nonzero(mask[rh_lh])[0][:len(lines[rh_lh])]:
            stdout.write("\r %d%%" % (i*101//len(id_list)))
            stdout.flush()
              
            line_colision = []
            j = 1
            # initial test if empty or less than 1 points
            if len(lines[rh_lh][line_idx]) < 2:
                new_line = np.tile(vts[rh_lh][idx],(2,1))
                bad_pft_count[rh_lh] += 1
            # initial test inside
            elif tree[rh_lh].InsideOrOutside(lines[rh_lh][line_idx][1]) != -1:
                # if first point outside
                #id_list[i,:] = -1
                new_line = np.tile(lines[rh_lh][line_idx][0],(2,1))
                bad_start_count[rh_lh] += 1
            else:
                # stop at first intersection
                id_list[i,0] = idx
                id_surf[i,0] = rh_lh
                while j < len(lines[rh_lh][line_idx])-1:
                    p0 = lines[rh_lh][line_idx][j]
                    p1 = lines[rh_lh][line_idx][j+1]
                    pts = vtk.vtkPoints()
                    idList = vtk.vtkIdList()
                    
                    p0_in_out = tree[rh_lh].IntersectWithLine(p0, p1, pts, idList);
                    
                    if p0_in_out == 1: # in=-1, out=1
                        new_line = np.tile(lines[rh_lh][line_idx][0],(2,1))#test
                        bad_end_count[rh_lh] += 1
                        #id_list[i,1] = -1
                        #id_surf[i,1] = -1
                        break
                    elif pts.GetNumberOfPoints() > 0:
                        if np.all(mask[rh_lh][tris[rh_lh][idList.GetId(0)]]):
                            # crossing Gray matter area
                            new_line = np.zeros([j+2, 3], dtype=np.float)
                            new_line[:j+1] = lines[rh_lh][line_idx][:j+1] # copy segment before intersection
                            new_line[j+1] = pts.GetPoint(0) # add, intersection point
                            id_list[i,1] = idList.GetId(0)
                            id_surf[i,1] = rh_lh
                            end_count[rh_lh] += 1
                            break
                        else:
                            # crossing non-mask area
                            # count as a bad end for now
                            new_line = np.tile(lines[rh_lh][line_idx][0],(2,1))
                            bad_end_count[rh_lh] += 1
                            #id_list[i,1] = -1
                            #id_surf[i,1] = -1
                            break  
                    j += 1
                        
                if j == len(lines[rh_lh][line_idx])-1:
                    #new_line = lines[idx].copy()
                    new_line = np.tile(lines[rh_lh][line_idx][0],(2,1))#test
                    bad_end_count[rh_lh] += 1
                    #id_list[i,1] = -1
                    #id_surf[i,1] = -1
                else:
                    # test intersection with 2nd surface
                    is_inside_2surf = False
                    while j < len(lines[rh_lh][line_idx])-1:
                        p0 = lines[rh_lh][line_idx][j]
                        p1 = lines[rh_lh][line_idx][j+1]
                        pts = vtk.vtkPoints()
                        idList = vtk.vtkIdList()
                        
                        p0_in_out = tree[inv_rh_lh].IntersectWithLine(p0, p1, pts, idList);
                        
                        if pts.GetNumberOfPoints() > 0:
                            if np.all(mask[inv_rh_lh][tris[inv_rh_lh][idList.GetId(0)]]):
                                # Crossing Graymatter
                                if p0_in_out == 1:
                                    # cross outside first in a Gray matter region
                                    break
                                elif p0_in_out == -1:
                                    # touch from the inside the second surface
                                    new_line = np.zeros([j+2, 3], dtype=np.float)
                                    new_line[:j+1] = lines[rh_lh][line_idx][:j+1] # copy segment before intersection
                                    new_line[j+1] = pts.GetPoint(0) # add, intersection point
                                    id_list[i,1] = idList.GetId(0)
                                    id_surf[i,1] = inv_rh_lh
                                    
                                    # remove the bad end count from before
                                    bad_end_count[rh_lh] -= 1
                                    # add swap caount and end count to inv_index
                                    swap_end_count[rh_lh] += 1
                                    end_count[inv_rh_lh] += 1
                                    break
                                
                        j += 1
            new_lines += [new_line]
            line_idx += 1
            i += 1
else:
    # without mask
    new_lines = []
    i = 0
    for rh_lh in range(len(tree)):
        inv_rh_lh = 1 - rh_lh
        for idx in range(len(lines[rh_lh])):
            stdout.write("\r %d%%" % (i*101//len(id_list)))
            stdout.flush()
              
            line_colision = []
            j = 1
            # initial test if empty or less than 1 points
            if len(lines[rh_lh][idx]) < 2:
                new_line = np.tile(vts[rh_lh][idx],(2,1))
                bad_pft_count[rh_lh] += 1
            # initial test inside
            elif tree[rh_lh].InsideOrOutside(lines[rh_lh][idx][1]) != -1:
                # if first point outside
                #id_list[i,:] = -1
                new_line = np.tile(lines[rh_lh][idx][0],(2,1))
                bad_start_count[rh_lh] += 1
            else:
                # stop at first intersection
                id_list[i,0] = idx
                id_surf[i,0] = rh_lh
                while j < len(lines[rh_lh][idx])-1:
                    p0 = lines[rh_lh][idx][j]
                    p1 = lines[rh_lh][idx][j+1]
                    pts = vtk.vtkPoints()
                    idList = vtk.vtkIdList()
                    
                    p0_in_out = tree[rh_lh].IntersectWithLine(p0, p1, pts, idList);
                    if p0_in_out == 1: # in=-1, out=1
                        new_line = np.tile(lines[rh_lh][idx][0],(2,1))#test
                        bad_end_count[rh_lh] += 1
                        #id_list[i,1] = -1
                        #id_surf[i,1] = -1
                        break
                    elif pts.GetNumberOfPoints() > 0:
                        new_line = np.zeros([j+2, 3], dtype=np.float)
                        new_line[:j+1] = lines[rh_lh][idx][:j+1] # copy segment before intersection
                        new_line[j+1] = pts.GetPoint(0) # add, intersection point
                        id_list[i,1] = idList.GetId(0)
                        id_surf[i,1] = rh_lh
                        end_count[rh_lh] += 1
                        break
                    j += 1
                        
                if j == len(lines[rh_lh][idx])-1:
                    #new_line = lines[idx].copy()
                    new_line = np.tile(lines[rh_lh][idx][0],(2,1))#test
                    bad_end_count[rh_lh] += 1
                    #id_list[i,1] = -1
                else:
                    # test intersection with 2nd surface
                    is_inside_2surf = False
                    while j < len(lines[rh_lh][idx])-1:
                        p0 = lines[rh_lh][idx][j]
                        p1 = lines[rh_lh][idx][j+1]
                        pts = vtk.vtkPoints()
                        idList = vtk.vtkIdList()
                        
                        if tree[rh_lh].IntersectWithLine(p0, p1, pts, idList) == 1 :
                            # go back in the first surface (not good)
                            break
                        
                        p0_in_out = tree[inv_rh_lh].IntersectWithLine(p0, p1, pts, idList);
                        if p0_in_out == 1:
                            # go inside the second surface
                            # todo add test for CC
                            is_inside_2surf = True
                            if pts.GetNumberOfPoints() > 1:
                                #stop if multiple surface touching
                                break
                        elif p0_in_out == -1 and is_inside_2surf: # in=-1, out=1
                            # touch from the inside the second surface
                            new_line = np.zeros([j+2, 3], dtype=np.float)
                            new_line[:j+1] = lines[rh_lh][idx][:j+1] # copy segment before intersection
                            new_line[j+1] = pts.GetPoint(0) # add, intersection point
                            id_list[i,1] = idList.GetId(0)
                            id_surf[i,1] = inv_rh_lh
                            
                            end_count[rh_lh] -= 1
                            swap_end_count[rh_lh] += 1
                            end_count[inv_rh_lh] += 1
                            break
                        j += 1
            new_lines += [new_line]
            i += 1
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
valid_count = [start_count[0] - bad_start_count[0] - bad_end_count[0] - bad_pft_count[0],
               start_count[1] - bad_start_count[1] - bad_end_count[1] - bad_pft_count[1]]

if args.report is not None:
    report = open(args.report,"w")
    report.write("# Report : [ right , left ]\n")
    report.write("start_count : " + str(start_count) + "\n")
    report.write("bad_pft_count : " + str(bad_pft_count) + "\n")
    report.write("bad_start_count : " + str(bad_start_count) + "\n")
    report.write("end_count : " + str(end_count) + "\n")
    report.write("bad_end_count : " + str(bad_end_count) + "\n") 
    report.write("swap_end_count : " + str(swap_end_count) + "\n") 
    report.write("valid_count : " + str(valid_count) + "\n") 
    report.close() 
else:
    print "\n Report : [ right , left ]"
    print "start_count : " + str(start_count)
    print "bad_pft_count : " + str(bad_pft_count)
    print "bad_start_count : " + str(bad_start_count)
    print "end_count : " + str(end_count)
    print "bad_end_count : " + str(bad_end_count)
    print "swap_end_count : " + str(swap_end_count)
    print "valid_count : " + str(valid_count)
