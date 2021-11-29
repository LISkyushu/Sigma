# -*- coding: utf-8 -*-
"""
Created on Thu May 20 06:26:44 2021

@author: zfoong
"""

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
import matplotlib.pyplot as plt
import random

class utils:    
    
    def matrix_to_graph(tpmatrix, plot=False):
        # rows, cols = np.where(tpmatrix >= 0.1)
        # edges = zip(rows.tolist(), cols.tolist())
        # G = nx.Graph()
        # G.add_edges_from(edges)
        G=nx.from_numpy_matrix(tpmatrix, create_using=nx.DiGraph)
        # G=nx.from_numpy_matrix(tpmatrix)
        
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        
        
        if plot is True:
            plt.figure(figsize=(5,5))
            nx.draw(G, pos = nx.circular_layout(G), width=2.0, node_size=300, with_labels=True)
            # plt.show()
            plt.savefig("graph/plot.png", dpi=1000)
        return G
        
    