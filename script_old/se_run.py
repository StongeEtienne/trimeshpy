from __future__ import division
import itertools, os

import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from nibabel.freesurfer.io import read_annot

from trimeshpy.trimesh_vtk import TriMesh_Vtk
from trimeshpy.math import length

subs = range(1,12)
acqs = [1,2,3]

surftracts = ["1_0","100_1"]
pft_npft = [False, True]
det_prob = ["det", "prob"]
rh_lh = ["rh", "lh"]

"""
#avg connectivity
c_matrix = np.load("../home_mint/c_matrix_l10.npy")
titles = np.load("../home_mint/c_matrix_titles.npy")
for i in range(8):
    index = i
    print c_matrix.shape
    #avg_matrix = np.mean(c_matrix, axis=(0,1,2)) # for all index mean
    avg_matrix = np.mean(c_matrix[index], axis=(0,1))
    print titles[index]
    np.fill_diagonal(avg_matrix, 0.0)
    plt.imshow(np.log(avg_matrix + avg_matrix.T + 1.0),  interpolation="nearest")
    plt.axes().xaxis.tick_top()
    plt.tick_params(top=False, bottom=False, pad=1)
    plt.tight_layout()
    plt.savefig(titles[index] + ".png")
    #plt.title("mean connectivity matrix (log-scale)")
    #plt.show()
"""

"""
# reproducibility
def dist_c_matrix(a,b, norm=True, symetrize=True, diag_to_zero=True):
    if diag_to_zero:
        np.fill_diagonal(a, 0.0)
        np.fill_diagonal(b, 0.0)
    if symetrize:
        a = a+a.T
        b = b+b.T
    if norm:
        a = a/np.sum(a)
        b = b/np.sum(b)
    mask = ((a+b) > 0.0)
    return np.sum(np.square(a[mask]-b[mask])/(a[mask]+b[mask]))/2.0 # PDFs Chi dquare
    #return np.mean(np.abs(a[mask]-b[mask])/(a[mask]+b[mask])) # PDFs
    #return np.sqrt(np.sum(np.square(a-b))) # euclidian

vmax = 0.3
c_matrix = np.load("../home_mint/c_matrix_l10.npy")
titles = np.load("../home_mint/c_matrix_titles.npy")

ns = c_matrix.shape[0]
na = c_matrix.shape[1]
nm = c_matrix.shape[2]
dist_matrices = np.zeros([nm, ns*na, ns*na])
plot_index = 0 
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False)
for m_i in range(nm):
    intra_dists = []
    extra_dists = []
    intra_dist_sum = np.zeros([len(subs)])
    intra_dist_min = np.ones([len(subs)])*np.Infinity
    intra_dist_max = np.zeros([len(subs)])
    
    for pos_x in  range(ns*na):
        for pos_y in  range(pos_x):
            s_ix = pos_x//na
            a_ix = pos_x%na
            s_iy = pos_y//na
            a_iy = pos_y%na
            
            #print s_ix, a_ix, " d ", s_iy, a_iy, "///", pos_x, pos_y
            dist = dist_c_matrix(c_matrix[s_ix, a_ix, m_i], c_matrix[s_iy, a_iy, m_i])
            if m_i%2==0:
                dist_matrices[m_i//2, pos_x, pos_y] = dist
            else:
                dist_matrices[m_i//2, pos_y, pos_x] = dist
                
            if s_ix==s_iy:
                intra_dists.append(dist)
                intra_dist_sum[s_ix] += dist
                if dist < intra_dist_min[s_ix]:
                    intra_dist_min[s_ix] = dist
                if dist > intra_dist_max[s_ix]:
                    intra_dist_max[s_ix] = dist
            else:
                extra_dists.append(dist)
                
    print intra_dist_sum
    
    p_title = titles[m_i][6:] #+ " ( %.3f" %np.mean(intra_dists) + ", %.3f" % np.mean(extra_dists) + " ) std:( %.3f" %np.std(intra_dists) + ", %.3f" % np.std(extra_dists) + " )"
    p_title += "_pft"
    p_title = p_title.replace("_npft_pft", "_local_tracking")
    p_title = p_title.replace("100_1", "st")
    p_title = p_title.replace("1_0", "")
    if p_title[0] == "_": p_title = p_title[1:]
    print p_title + "  %.3f" %np.mean(intra_dists) + ", %.3f" %np.std(intra_dists) + " & %.3f" % np.mean(extra_dists) + ", %.3f" % np.std(extra_dists) + " & %.2f" % (np.mean(extra_dists)/np.mean(intra_dists)) 
    p_title = p_title.replace("prob_", "")
    if m_i%2==1:
        #np.fill_diagonal(dist_matrices[m_i//2], np.NAN)
        im = axes.flat[plot_index].imshow(dist_matrices[m_i//2],  interpolation="nearest", vmin=0, vmax=vmax)
        #axes.flat[plot_index].xaxis.set_ticks(np.arange(0, 33, 3.0))
        ticks_v = np.arange(0, 33, 3)
        ticks_s = ['s' + str(i) for i in range(1,12)]
        print len(ticks_v),len(ticks_s)
        axes.flat[plot_index].xaxis.tick_top()
        axes.flat[plot_index].xaxis.set_ticks(ticks_v+1)
        #xes.flat[plot_index].xaxis.set_tick_out()
        axes.flat[plot_index].xaxis.set_ticklabels(ticks_s)
        axes.flat[plot_index].xaxis.set_tick_params(top=False, bottom=False, pad=0)
        #axes.flat[plot_index].xaxis.set_major_locator(ticks_v)
        #axes.flat[plot_index].xaxis.set_ticks_position(ticks_v)
        axes.flat[plot_index].yaxis.set_ticks(ticks_v+1)
        axes.flat[plot_index].yaxis.set_ticklabels(ticks_s)
        axes.flat[plot_index].yaxis.set_tick_params(left=False, right=False,pad=0)
        
        
        #axes.flat[plot_index].set_title(p_title + " (prob)")
        #axes.flat[plot_index].set_ylabel(p_title  + " (det)")
        plot_index += 1
    
    
    
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.10, hspace=0.10)
#fig.tight_layout()
fig.subplots_adjust(right=0.93)
cbar_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
fig.colorbar(im, cax=cbar_ax)
plt.show()
"""
"""
#invalid table
color_list =  ['lightcoral', 'darkred', 'skyblue', 'darkblue', 'lightgreen', 'darkgreen', 'khaki', 'orange']
my_patches =  []
fontsize = 20
min_v = 0
max_v = 100
step_v = 0.2
index = 0
for  st, pft, algo  in itertools.product( surftracts, pft_npft, det_prob):
    if pft:
        if st == "1_0":
            algo_string = str("pft_-_" + algo)
        else:
            algo_string = str("surface_flow_-_pft_-_"  + algo)
    else:
        if st == "1_0":
            algo_string = str("local_tracking_-_" + algo)
        else:
            algo_string = str("surface_flow_-_local_tracking_-_"  + algo)
    
    lengths_avg_list = []
    invalid_list = []
    nb_smaller_10mm = []
    valid = []
    for sub, acq in itertools.product(subs, acqs):
        s_acq = str(acq)
        s_sub = str(sub)
        prefix = "../home_mint/panthera_result/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
        if pft:
            lengths = np.load(  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_length.npy" )
            idv = np.load(  prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+".npy" )
        else:
            lengths = np.load(  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft_length.npy" )
            idv = np.load(  prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy" )
            
        nb_valid = np.count_nonzero(idv[:,1]+1)
        nb_invalid = len(idv) - nb_valid
        invalid_list.append(float(nb_invalid)/len(idv))
        lengths_avg_list.append(np.mean(lengths))
        nb_smaller_10mm.append(np.sum(lengths<10.0)/len(lengths))#todo change if <10 / total seed
        valid.append(np.sum(lengths>=10.0)/len(idv))
        
    print algo_string
    print "invalid:","%.2f" % (np.mean(invalid_list)*100)+"\%,", "%.2f" % (np.std(invalid_list)*100)+"\% std"
    print "length:","%.2f" % (np.mean(lengths_avg_list))+",", "%.2f" % (np.std(lengths_avg_list))+" std"
    print "<10mm:","%.2f" % (np.mean(nb_smaller_10mm)*100)+"\%,", "%.2f" % (np.std(nb_smaller_10mm)*100)+"\% std"
    print "valid:","%.2f" % (np.mean(valid)*100)+"\%,", "%.2f" % (np.std(valid)*100)+"\% std"
    #print "%.2f" % (np.mean(invalid_list)*100)+"\%,", "%.2f" % (np.std(invalid_list)*100)+"\% std &","%.2f" % (np.mean(lengths_avg_list))+",", "%.2f" % (np.std(lengths_avg_list))+" std &","%.2f" % (np.mean(nb_smaller_10mm)*100)+"\%,", "%.2f" % (np.std(nb_smaller_10mm)*100)+"\% std &","%.2f" % (np.mean(valid)*100)+"\%,", "%.2f" % (np.std(valid)*100)+"\% std"
"""
#"""
#length statistic and histogram
index = 0
color_list =  ['lightcoral', 'darkred', 'skyblue', 'darkblue', 'lightgreen', 'darkgreen', 'khaki', 'orange']
name_paper =  ['local tracking - det', 'local tracking - prob', 
               'PFT - det', 'PFT - prob', 
               'surface flow, local tracking - det', 'surface flow, local tracking - prob', 
               'surface flow, PFT - det', 'surface flow, PFT - prob']
my_patches =  []
fontsize = 20
min_v = 0
max_v = 100
step_v = 0.2
index = 0 
for  st, pft, algo  in itertools.product( surftracts, pft_npft, det_prob):
    if pft:
        if st == "1_0":
            algo_string = str("pft_-_" + algo)
        else:
            algo_string = str("surface_flow_-_pft_-_"  + algo)
    else:
        if st == "1_0":
            algo_string = str("local_tracking_-_" + algo)
        else:
            algo_string = str("surface_flow_-_local_tracking_-_"  + algo)
    
    lengths_cat = None
    for sub, acq in itertools.product(subs, acqs):
        s_acq = str(acq)
        s_sub = str(sub)
        prefix = "../home_mint/panthera_result/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
        if pft:
            lengths = np.load(  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_length.npy" )
        else:
            lengths = np.load(  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft_length.npy" )
            
        if lengths_cat is None:
            lengths_cat = lengths
        else:
            lengths_cat = np.concatenate((lengths_cat, lengths))
    print lengths_cat.shape
    hist, bin_edges = np.histogram(lengths_cat, np.arange(min_v,max_v,step_v))
    hist = hist.astype(np.float)/len(lengths_cat)
    bar_size = float(step_v)/8.0
    #plt.bar(bin_edges[:-1]+index*bar_size, hist, width=bar_size, color=color_list[index])
    #plt.plot(bin_edges, np.concatenate(([0], hist)), color=color_list[index], linewidth=2.0, label=algo_string)
    plt.plot(bin_edges, np.concatenate(([0], hist)), color=color_list[index], linewidth=2.0, label=name_paper[index])
    plt.legend()
    plt.xlim(min_v, max_v)
    index += 1
    
plt.xlabel("Streamline length (mm)", fontsize=fontsize)
plt.ylabel("Streamline length distribution ", fontsize=fontsize)
plt.axis(fontsize=fontsize)
plt.show()
#"""
"""
# plot matrix and generate full matrix file
full_coo_array = np.zeros([len(subs),len(acqs),8 , 154, 154])
for i, j in itertools.product(range(len(subs)), range(len(acqs))):
    s_acq = str(acqs[j])
    s_sub = str(subs[i])
    prefix = "../home_mint/panthera_result/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    #fig, axes = plt.subplots(nrows=2, ncols=4, sharex=False, sharey=False)
    titles = []
    plot_index = 0
    for pft, st, algo in itertools.product(pft_npft, surftracts, det_prob):
        if pft:
            file_matrix =  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_coo_avg_l10.npy" 
            title = "S"+s_sub+"A"+s_acq+"_"+st+"_"+algo
        else:
            file_matrix =  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft_coo_avg_l10.npy" 
            title = "S"+s_sub+"A"+s_acq+"_"+st+"_"+algo+"_npft"
        titles.append(title)
        c_matrix = np.load(file_matrix)
        full_coo_array[i,j,plot_index] = c_matrix
        #print i,j,plot_index
        #c_matrix += c_matrix.T
        #np.fill_diagonal(c_matrix, 0)
        #plt.imshow(np.log(c_matrix+1),  interpolation="nearest")
        #plt.title(title)
        #plt.show()
        #im = axes.flat[plot_index].imshow(np.log(c_matrix+1),  interpolation="nearest")
        #axes.flat[plot_index].set_title(title)
        plot_index += 1
        
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #fig.colorbar(im, cax=cbar_ax)
    #plt.title(prefix)
    #plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.10, hspace=0.0)
    #fig.tight_layout()
    #plt.show()
np.save("../home_mint/c_matrix_l10.npy", full_coo_array)
np.save("../home_mint/c_matrix_titles.npy", np.array(titles))
"""
"""
end_weigth = 1.0/3.0
# RESULTS
tag = [subs, surftracts, surftracts, pft_npft, det_prob] 
bir_array = np.zeros([len(subs), len(subs), len(surftracts), len(pft_npft), len(det_prob)])
for sub, acq in itertools.product(subs, acqs):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "../home_mint/panthera_result/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    rh_mask = prefix + "_/surf/rh_st_surf_mask.npy"
    lh_mask = prefix + "_/surf/lh_st_surf_mask.npy"
    
    rh_surf = prefix + "_/surf/rh.white_lps.vtk"
    lh_surf = prefix + "_/surf/lh.white_lps.vtk"
    rh_mesh = TriMesh_Vtk(rh_surf, None)
    lh_mesh = TriMesh_Vtk(lh_surf, None)
    vts = (rh_mesh.get_vertices(), lh_mesh.get_vertices())
    tris = (rh_mesh.get_triangles(), lh_mesh.get_triangles())
    
    rh_annot = prefix + "_/label/rh.aparc.a2009s.annot"
    lh_annot = prefix + "_/label/lh.aparc.a2009s.annot"
    [rh_vts_label, rh_label_color, rh_label_name] = read_annot(rh_annot)
    [lh_vts_label, lh_label_color, lh_label_name] = read_annot(lh_annot)
    vts_label = (rh_vts_label, lh_vts_label)
    nb_labels = len(rh_label_name) + len(lh_label_name) + 2 # rh + lh + spinal + gray 
    
    for pft, st, algo in itertools.product(pft_npft, surftracts, det_prob):
        
        # load npy array (lengths , streamlines end points surface and vertex/triangle index)
        if pft:
            lengths = np.load( prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_length.npy" )
            ids = np.load(  prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+".npy" )
            idv = np.load( prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+".npy" )
            
            save_matrix =  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_coo_avg.npy" 
            title = "S"+s_acq+"A"+s_acq+"_"+st+"_"+algo
        else:
            lengths = np.load(  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft_length.npy" )
            ids = np.load(  prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy" )
            idv = np.load(  prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy" )
            
            save_matrix =  prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft_coo_avg.npy" 
            title = "S"+s_acq+"A"+s_acq+"_"+st+"_"+algo+"_npft"
            
        idv_to_idv = -np.ones_like(idv)
        #idv_to_idv[:,0] = idv[:,0]
        
        #label_to_label = -np.ones_like(idv)
        c_matrix = np.zeros([nb_labels,nb_labels], dtype=np.float)
        l_index = 0
        save_matrix = save_matrix.replace(".npy", "_l10.npy")
        for i in range(len(idv)):
            start_vts_idx = idv[i,0]
            start_surf_id = ids[i,0]
            
            tri_idx = idv[i,1]
            end_surf_id = ids[i,1]
            if start_surf_id != -1 and end_surf_id  != -1:
                if lengths[l_index] > 10.0:
                    if start_surf_id == 0:
                        #label_to_label[i,0]  = vts_label[start_surf_id][start_vts_idx]
                        start_label = vts_label[start_surf_id][start_vts_idx]
                    elif start_surf_id == 1:
                        start_label = vts_label[start_surf_id][start_vts_idx] + len(rh_label_name)
                    else:
                        start_label = nb_labels + start_surf_id - 6
                    
                    if end_surf_id == 0:
                        tri_pt_idx = tris[end_surf_id][tri_idx]
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[0]]] += end_weigth
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[1]]] += end_weigth
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[2]]] += end_weigth
                        #todo improve ?
                        #idv_to_idv[i,1] = tri_pt_idx[0]
                        #label_to_label[i,1]  = vts_label[end_surf_id][tri_pt_idx[0]]
                    elif end_surf_id == 1:
                        tri_pt_idx = tris[end_surf_id][tri_idx]
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[0]] + len(rh_label_name)] += end_weigth
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[1]] + len(rh_label_name)] += end_weigth
                        c_matrix[start_label, vts_label[end_surf_id][tri_pt_idx[2]] + len(rh_label_name)] += end_weigth
                    else:
                        c_matrix[start_label, nb_labels + end_surf_id - 6] += 1.0
                
                l_index += 1
                
        print save_matrix
        np.save(save_matrix, c_matrix)
"""   
"""
#se_stats / length  TODO
for sub, acq, pft, st, algo in itertools.product(subs, acqs, pft_npft, surftracts, det_prob):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq

    if pft:
        tracto = prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+".fib"
    else:
        tracto = prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
        
    out_length = tracto.replace(".fib", "_length.npy")
        
    
    process = "python se_length.py "
    total = process + tracto +" "+ out_length
    
    print total
    os.system(total)
"""
"""
#se_fusion
for sub, acq, pft, st, algo in itertools.product(subs, acqs, pft_npft, surftracts, det_prob):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    if pft:
        out_tracto_cut = prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+".fib"
    else:
        out_tracto_cut = prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
        
    rh_surf = prefix + "_/rh_white_lps_smooth_2_5_flow_"+st+".vtk"
    lh_surf = prefix + "_/lh_white_lps_smooth_2_5_flow_"+st+".vtk"
    
    rh_flow = prefix + "_/rh_white_lps_smooth_2_5_flow_"+st+".dat"
    lh_flow = prefix + "_/lh_white_lps_smooth_2_5_flow_"+st+".dat"
    
    rh_flow_info = prefix + "_/rh_white_lps_smooth_2_5_flow_"+st+".npy"
    lh_flow_info = prefix + "_/lh_white_lps_smooth_2_5_flow_"+st+".npy"
    
    if pft:
        out_inter_idx = prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+".npy"
        out_surf_idx = prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+".npy"
        output_fiber = prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+".fib"
    else:
        out_inter_idx = prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy"
        out_surf_idx = prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy"
        output_fiber = prefix + "_/final_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
    
    process = "python se_fusion.py "
    
    total = process + out_tracto_cut +" "+ rh_surf +" "+ lh_surf +" "+ rh_flow +" "+ lh_flow +" "+ rh_flow_info +" "+ lh_flow_info +" "+ out_inter_idx +" "+ out_surf_idx +" "+ output_fiber
    print total
    #os.system(total)
"""
"""
#se_intersection
for sub, acq, pft, st, algo in itertools.product(subs, acqs, pft_npft, surftracts, det_prob):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    if pft:
        rh_fiber = prefix + "_/rh_smooth_2_5_flow_"+st+"_"+algo+".fib"
        lh_fiber = prefix + "_/lh_smooth_2_5_flow_"+st+"_"+algo+".fib"
    else:
        rh_fiber = prefix + "_/rh_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
        lh_fiber = prefix + "_/lh_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
    
    rh_surf = prefix + "_/rh_white_lps_smooth_2_5_flow_"+st+".vtk"
    lh_surf = prefix + "_/lh_white_lps_smooth_2_5_flow_"+st+".vtk"
    
    rh_out_surf = prefix + "_/surf/rh.pial_lps.vtk"
    lh_out_surf = prefix + "_/surf/lh.pial_lps.vtk"
    
    rh_mask = prefix + "_/surf/rh_st_surf_mask.npy"
    lh_mask = prefix + "_/surf/lh_st_surf_mask.npy"
    
    if pft:
        out_tracto_cut = prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+".fib"
        out_inter_idx = prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+".npy"
        out_surf_idx = prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+".npy"
        report = " -report " + prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+".txt"
    else:
        out_tracto_cut = prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"
        out_inter_idx = prefix + "_/cut_idv_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy"
        out_surf_idx = prefix + "_/cut_ids_smooth_2_5_flow_"+st+"_"+algo+"_npft.npy"
        report = " -report " + prefix + "_/cut_smooth_2_5_flow_"+st+"_"+algo+"_npft.txt"
    
    spinal = " -nuclei " + prefix + "_wmparc.a2009s.nii.gz.spinal2.vtk"
    nuclei = " -nuclei_soft " + prefix + "_wmparc.a2009s.nii.gz.nuclei.vtk"
    
    process = "python se_mesh_intersect2.py "
    
    total = process + rh_fiber +" "+ lh_fiber +" "+ rh_surf +" "+ lh_surf +" "+ rh_out_surf +" "+ lh_out_surf +" "+ rh_mask +" "+ lh_mask +" "+ out_tracto_cut +" "+ out_inter_idx +" "+ out_surf_idx + report + spinal + nuclei
    print total
    #os.system(total)
"""
"""
#se_tracto/pft
for sub, acq, pft, st, algo, side in itertools.product(subs, acqs, pft_npft, surftracts, det_prob, rh_lh):
    s_acq = str(acq)
    s_sub = str(sub)
    #prefix = "scratch/panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    prefix2 = prefix + "_/" + side
    
    #todo for lh, rh,   smooth:2-5,2-2    st:100-1,1-0
    
    fodf = prefix + "_fod.nii.gz"
    points = prefix2 + "_white_lps_smooth_2_5_pts_"+st+".npy"
    normal = prefix2 + "_white_lps_smooth_2_5_nls_"+st+".npy"
    include_m = prefix + "_map_include.nii.gz"
    exclude_m = prefix + "_map_exclude.nii.gz"
    
    if pft:
        output_fiber = prefix2 + "_smooth_2_5_flow_"+st+"_"+algo+".fib"
    else:
        output_fiber = prefix2 + "_smooth_2_5_flow_"+st+"_"+algo+"_npft.fib"

    
    process = "python compute_pft_tracking_mesh.py"
    params = " --basis mrtrix  --sh_interp tl --mask_interp tl -inv_seed_dir --all -f --tq --pft_theta 40 --step 0.2 "
    params += " --algo " + algo
    if not(pft): params += " --no_pft "
    
    # test
    #params += " -test 500"
    
    total = process + " " + fodf + " " + points + " " + normal + " " + include_m + " " + exclude_m  + " " + output_fiber + params
    
    print total
    #os.system(total)
"""   
"""
#se_tracking
for sub, acq, side in itertools.product(subs, acqs, rh_lh):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    midfix = "_/surf/" + side
    midfix_out = "_/" + side
    
    #todo for lh, rh,  st:100-1,1-0
    f1_sufix = "_white_lps_smooth_2_5.vtk"
    f2_sufix = "_white_lps_smooth_2_5_flow_100_1.dat"
    file1 =  prefix + midfix + f1_sufix
    file2 =  prefix + midfix_out + f2_sufix
    
    mask = "_st_surf_mask.npy"
    end_points = "_white_lps_smooth_2_5_pts_100_1.npy"
    end_normal = "_white_lps_smooth_2_5_nls_100_1.npy"
    end_surf = "_white_lps_smooth_2_5_flow_100_1.vtk"
    info = "_white_lps_smooth_2_5_flow_100_1.npy"
    mask_param =  " -mask " + prefix + midfix + mask
    end_points_param = " -end_points " + prefix + midfix_out + end_points
    end_normal_param = " -end_normal " + prefix + midfix_out + end_normal
    end_surf_param = " -end_surf " + prefix + midfix_out + end_surf
    info_param = " -info " + prefix + midfix_out + info
    
    process = "python se_tracking.py"
    params = " -nb_step 100 -step_size 1 "
    
    total = process + " " + file1 + " " + file2 + " " + params + mask_param + end_points_param + end_normal_param + end_surf_param + info_param
    
    print total
    #print prefix, params
    #os.system(total)
"""
"""
#se_smooth
for sub, acq, smooth in itertools.product(subs, acqs, smooths):
    s_acq = str(acq)
    s_sub = str(sub)
    s_smooth = str(smooth)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    midfix = "_/"
    
    f1_sufix = "surf/lh.white_lps.vtk"
    f2_sufix = "surf/lh_st_surf_mask.npy"
    file1 =  prefix + midfix + f1_sufix
    file2 =  prefix + midfix + f2_sufix
    
    out_sufix = "surf/lh_white_lps_smooth_2_"+s_smooth+".vtk"
    out_file = prefix + midfix + out_sufix
    
    process = "python se_smooth.py"
    params = " -nb_step 2 -step_size " + s_smooth + " -mask " + file2
    
    total = process + " " + file1 + " " + out_file + " " + params
    
    print total
    #os.system(total)
"""
    
""" #se_atlas
for sub, acq in itertools.product(subs, acqs):
    s_acq = str(acq)
    s_sub = str(sub)
    prefix = "panthera/S"+s_sub+"/A"+s_acq+"/S"+s_sub+"-A"+s_acq
    
    midfix = "_/"
    
    f1_sufix = "surf/rh.white.vtk"
    f2_sufix = "label/rh.aparc.a2009s.annot"
    
    
    file1 =  prefix + midfix + f1_sufix
    file2 =  prefix + midfix + f2_sufix
    
    out_sufix = "surf/rh_st_surf_mask.npy"
    out_file = prefix + midfix + out_sufix
    
    process = "python se_atlas.py"
    params = "-index 32 67 -1 --inverse_mask --v "
    
    total = process + " " + file1 + " " + file2 + " " + params
    #total = process + " " + + file1 + " " + file2 + " " + param
    
    print total
    #os.system(total)
"""
    
