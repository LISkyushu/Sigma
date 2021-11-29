# -*- coding: utf-8 -*-
import networkx as nx
# from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.algorithms.community.quality import modularity
from networkx.utils.mapped_queue import MappedQueue
import numpy as np
import matplotlib.pyplot as plt
from utils import *

class modularity_max:   
    """
    Finding communities (chunk) with hierarchy
    
    """
    
    def hierarchy_modularity_communities(self, G):
        """
        Code adapted and modified from (https://networkx.org/documentation/stable/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities).
        Here we incorporate multiple modularities for each iteration to
        extract multiple hierarchies of communities (chunk). code are updated
        by referring to the work (M. Newman, M. Girvan, Physical review. E, 
        Statistical, nonlinear, and soft matter physics69, 026113 (2004)).
        """
        # Override
        modularity_list = []
        community_list = []
        
        # First create one community for each node
        communities = list([frozenset([u]) for u in G.nodes()])
        output_size = len(communities)
        # Track merges
        merges = []
        # Greedily merge communities until no improvement is possible
        old_modularity = None
        new_modularity = modularity(G, communities)
        while old_modularity is None or len([i for i in communities if len(i) != 0]) > 2:
            # Save modularity for comparison
            old_modularity = new_modularity
            # Find best pair to merge
            trial_communities = list(communities)
            to_merge = None
            max_merge = -1000
            for i, u in enumerate(communities):
                for j, v in enumerate(communities):
                    # Skip i=j and empty communities
                    if j <= i or len(u) == 0 or len(v) == 0:
                        continue
                    # Merge communities u and v
                    trial_communities[j] = u | v
                    trial_communities[i] = frozenset([])
                    trial_modularity = modularity(G, trial_communities)
                    if trial_modularity > max_merge:
                        max_merge = trial_modularity
                        to_merge = (i, j, new_modularity - old_modularity)
                    trial_communities[i] = u
                    trial_communities[j] = v
            new_modularity = max_merge
            modularity_list.append(new_modularity)
            # max_merge
            if to_merge is not None:
                # If the best merge improves modularity, use it
                merges.append(to_merge)
                i, j, dq = to_merge
                u, v = communities[i], communities[j]
                communities[j] = u | v
                communities[i] = frozenset([])
            community_list.append(communities.copy())
        # Remove empty communities and sort
        return_community_list = []
        for communitie in community_list:
            comm = [c for c in communitie if len(c) > 0]
            comm = sorted(comm, key=lambda x: len(x), reverse=True)
            return_community_list.append(comm)
        return return_community_list, modularity_list
    
    def __init__(self, hierarchy_num = 2):
        self.name = "Modularity MAX"
        self.label = []
        self.hierarchy_num = hierarchy_num
        
    def input(self, x_input):
        
        x = x_input
    
        output_size = x.shape[1]
        
        Dtable = np.zeros([output_size, output_size])
        TPMatrix = np.zeros([output_size, output_size])
        
        prev_state = np.argmax(x[0])
        state = None
        for i in x[1:]:
            if np.max(i) != 1:
                continue
            state = np.argmax(i)
            Dtable[prev_state][state]+= 1                
            prev_state = state
                
        for i, j in enumerate(Dtable):
            state_total = np.sum(j)
            if state_total == 0:
                continue
            TPMatrix[i] = Dtable[i]/state_total

        g = utils.matrix_to_graph(TPMatrix, False)
        c,m = list(self.hierarchy_modularity_communities(g))
        c.append(list([frozenset(g.nodes)])) # Communities are merge into one 
        m.append(0.0)
        
        d_diff_list = []
        for d in range(1,len(m)-1):
            d_diff =  abs((m[d] - m[d-1]) - ((m[d+1] - m[d-1])/2))
            d_diff_list.append(d_diff)
        
        d_diff_index = (np.argsort(d_diff_list)[::-1])+1
        
        hierarchy = self.hierarchy_num
        
        d_diff_index = np.sort(d_diff_index[:hierarchy])
        
        label = np.zeros((hierarchy, output_size), dtype=int)
        for h in range(hierarchy):
            for index, i in enumerate(c[d_diff_index[h]]):
                for k in i:
                    label[h,k] = index
              
        self.label = label
            
        return self.label
    
    def evaluation(self, true_label, self_label):     
        """
        Evaluate predicted label using NMI score
        Return the average NMI score of every hierarchies.
        True label provided by environment
        """
        if len(self_label) == 0:
            return
        print("Modularity Max Learned Labels: ",self_label)
        print("Modularity Max Correct Labels: ",true_label)
        nmi_l = []
        
        total_hierarchy = true_label.shape[0]
        
        for h in range(total_hierarchy):
            nmi_l.append(normalized_mutual_info_score(self_label[h], true_label[h]))
        nmi_score = np.sum(nmi_l)/len(nmi_l)
        return nmi_score
    
    def evaluation(self, true_label, self_label):    
        """
        Evaluate predicted label using NMI score
        Return the average NMI score of every hierarchies.
        True label provided by environment
        """
        hierarchy = None                       
        labels = self_label
        print("number of layers it thinks: " + str(len(labels)))
        total_env_hierarchy = len(true_label)
        # true_label = np.flip(true_label, axis=0)
        
        # Handle exception of predicted levels of hierarchy less then true label
        if len(true_label) > len(labels):
            remaining_levels = len(true_label) - len(labels)
            for _ in range(remaining_levels):
                labels = np.vstack((labels, (np.full(len(labels[0]), -1))))
        
        nmi_l = []
        for h in range(total_env_hierarchy):
            nmi_l.append(normalized_mutual_info_score(labels[h], true_label[h]))
        nmi_score = np.sum(nmi_l)/len(nmi_l)
        return nmi_score
    
    def plot_animation(self, name, true_labels=None):
        print("Movie not supported for Modularity Maximization")
        return
    

