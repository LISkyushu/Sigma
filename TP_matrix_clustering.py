# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, maxdists
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import normalized_mutual_info_score

#problems
from OverlapChunkTest1 import *
from OverlapChunkTest2 import *
from LongChunkTest import *
from FixedChunkTest import *
from GraphWalkTest import *
from HierarchyFixedChunkTest import *
from HierarchyProbChunkTest import *
from DynamicHierarchyProbChunkTest import *

def hierarchical_organize(x, hierarchy=None):
    
    input_size = x.shape[0]
    method = "single"
    Z = linkage(x, method)
    # dendrogram(Z)
    
    Z_maxdists = maxdists(Z)
    d_diff_list = []
    for d in range(len(Z_maxdists)-1):
        d_diff = Z_maxdists[d+1] - Z_maxdists[d]
        d_diff_list.append(d_diff)
    
    d_diff_index = np.argsort(d_diff_list)[::-1]
    
    max_diff = d_diff_index[0]
    tmp_d_diff_index = [max_diff]
    for d in d_diff_index[1:]:
        if max_diff > d:
            max_diff = d
            tmp_d_diff_index.append(d)
    d_diff_index = tmp_d_diff_index  
    
    total_hierarchy = len(d_diff_index)
    if hierarchy is not None:
        total_hierarchy = hierarchy
        
    labels = np.empty((total_hierarchy, input_size), dtype=int)
    for h in range(total_hierarchy):
        n_cluster = input_size - d_diff_index[h] - 1
        label = AgglomerativeClustering(n_clusters=n_cluster, linkage=method).fit_predict(x)
        labels[h,:] = label
        
    # self.labels = np.flip(labels, axis=0)           

    return labels

time_delay = 10
sequence_length = 600000
simulation = 30

problem_list = ["IH", "HB", "IEH", "DIH", "DCH", "EC2EH", "EH2EC", "DCS"]

problem_dict = {
    "IH": HierarchyProbChunkTest(time_delay, filename="fixed_chunk_unbalanced.dot"), # Unbalance Tree
    "HB": HierarchyProbChunkTest(time_delay, filename="fixed_chunk_branches.dot"), # Tree with branches
    "IEH": HierarchyProbChunkTest(time_delay, filename="fixed_chunk_deep_hierarchy.dot"), # Tree with branches
    "DIH": DynamicHierarchyProbChunkTest(time_delay, ["fixed_chunk_branches.dot", "fixed_chunk_branches_2.dot"]), # Dynamic Hierarchy
    "DCH": DynamicHierarchyProbChunkTest(time_delay, ["fixed_chunk_6_3.dot", "fixed_chunk_6_3_dynamic.dot"]), # Dynamic Hierarchy
    "EC2EH": DynamicHierarchyProbChunkTest(time_delay, ["fixed_chunk_15_3.dot", "fixed_chunk_15_3_deep.dot"]), # Dynamic from extra chunk to extra hierarchy
    "EH2EC": DynamicHierarchyProbChunkTest(time_delay, ["fixed_chunk_15_3_deep.dot", "fixed_chunk_15_3.dot"]), # Dynamic from extra hierarchy to extra chunk
    "DCS": DynamicHierarchyProbChunkTest(time_delay, ["fixed_chunk_15_3.dot", "fixed_chunk_15_3_swap_partner.dot"]) # Dynamic from extra hierarchy to extra chunk
}

nmi_list = []

for problem_index in problem_list:
    for i in range(simulation):
        
        env = problem_dict[problem_index]
        output_size= env.getOutputSize()
        input_sequence, input_class = env.getSequence(sequence_length)
        tpmatrix = env.TP_matrix()
        
        hierarchy = None
        
        labels = hierarchical_organize(tpmatrix)
        
        print("number of layers it thinks: " + str(len(labels)))
        
        total_env_hierarchy = len(env.trueLabel())
        true_label = np.flip(env.trueLabel(), axis=0)
        
        # Handle exception of predicted levels of hierarchy less then true label
        if len(true_label) > len(labels):
            remaining_levels = len(true_label) - len(labels)
            for _ in range(remaining_levels):
                labels = np.vstack((labels, (np.full(len(labels[0]), -1))))
        
        # print(labels)
        # print(true_label)
        
        nmi_l = []
        for h in range(total_env_hierarchy):
            nmi_l.append(normalized_mutual_info_score(labels[h], true_label[h]))
        
        nmi_score = np.sum(nmi_l)/len(nmi_l)
        nmi_list.append(nmi_score)
        
    nmi_mean = np.mean(nmi_list)
    nmi_std = np.std(nmi_list)
    
    score = "{:.2f}".format(nmi_mean)+"±"+"{:.2f}".format(nmi_std)
    print("Score : " + score)



