#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import math
import numpy as np
import pandas as pd 
from tqdm import tqdm
from cycler import cycler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, maxdists, set_link_color_palette

# Problems
from HierarchyProbChunkTest import *
from HierarchyFixedChunkTest import *
from DynamicHierarchyProbChunkTest import *

# Methods
from VAE import *
from Sigma import *
from SyncMap import *
from modularity_max import *

# Stat test
from math import sqrt
from numpy import mean
from scipy.stats import t
from scipy.stats import sem


# In[2]:

def independent_ttest(data1, data2, alpha):
    """
    Function for calculating the t-test for two independent samples
    """
    mean1, mean2 = mean(data1), mean(data2)
    se1, se2 = sem(data1), sem(data2)
    sed = sqrt(se1**2.0 + se2**2.0)
    t_stat = (mean1 - mean2) / sed
    df = len(data1) + len(data2) - 2
    cv = t.ppf(1.0 - alpha, df)
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    return t_stat, df, cv, p

### Param ###
simulation = 30
algorithm_list = [1, 2, 3, 4] # 1:Syncmap; 2:VAE; 3:Sigma; 4:Mod Max
problem_list = ["IH", "HB", "IEH", "DIH", "DCH", "EC2EH", "EH2EC", "DCS"]
time_delay = 10
sequence_length = 600000
verbose = 1 # 0: no output; 1:avg+std; 2:avg;
movie = False # Only works when map dimension equals 2 or 3

### Initialize environment ###
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

result_dict = {}

for algorithm in algorithm_list:
    print("==============================================")
    print("Simulation initiated for algorithm: ", algorithm)
    for problem_index in problem_list:
        env = problem_dict[problem_index]
        output_size= env.getOutputSize()
        print("Output Size",output_size)
        
        nmi_list = []
        print("------------------------------------------------")
        print("Simulation initiated for environment: ", problem_index)
        for i in tqdm(range(simulation)):
            
            algorithm_class = None
            if algorithm == 1:
                ####### SyncMap ######
                number_of_nodes= output_size
                adaptation_rate= 0.001*output_size
                map_dimensions= 3
                algorithm_class = SyncMap(number_of_nodes, map_dimensions, adaptation_rate)
                ####### SyncMap ######
            if algorithm == 2:
                ###### Word2vec ######
                input_size= output_size
                latent_dim= 3
                timesteps= 100
                algorithm_class = VAE(input_size, latent_dim, timesteps)
                ###### Word2vec ######              
            if algorithm == 3:
                ####### Sigma ######
                number_of_nodes= output_size
                adaptation_rate= 0.001*output_size
                map_dimensions= 3
                algorithm_class = Sigma(number_of_nodes, map_dimensions, adaptation_rate)
                ####### Sigma ######
            if algorithm == 4:
                ####### Modularity MAX ######
                number_of_nodes = output_size
                algorithm_class = modularity_max()
                ####### Modularity MAX ######
            
            # Initalize method
            neuron_group = algorithm_class
            
            # Generate sequence data from environment
            input_sequence, input_class = env.getSequence(sequence_length)
            
            # Feed data into method
            neuron_group.input(input_sequence)

            if algorithm == 1 or algorithm == 2 or algorithm == 3:
                neuron_group.hierarchical_organize()

            # Evaluate method using NMI score
            nmi_score=neuron_group.evaluation(env.trueLabel(), neuron_group.label)
            nmi_list.append(nmi_score)
            
            if movie is True:
                neuron_group.plot_animation(problem_index, env.trueLabel())
            
        print("Simulation completed for environment: ", problem_index)
        print("------------------------------------------------")
    
        nmi_mean = np.mean(nmi_list)
        nmi_std = np.std(nmi_list)
        score = "{:.2f}".format(nmi_mean)+"±"+"{:.2f}".format(nmi_std)
        print("Score : " + score)
        if verbose == 1:
            if neuron_group.name not in result_dict:
                result_dict[neuron_group.name] = [score]
            else:
                result_dict[neuron_group.name].append(score)
        if verbose == 2:
            if neuron_group.name not in result_dict:
                result_dict[neuron_group.name] = [round(nmi_mean, 2)]
            else:
                result_dict[neuron_group.name].append(round(nmi_mean, 2))
          
        # Perform statistical test
        # Generate two independent samples
        num_slice = math.floor(len(nmi_list)/2) 
        data1 = nmi_list[:num_slice]
        data2 = nmi_list[num_slice:]
        # Calculate the t test
        alpha = 0.05
        t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
        print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
        # Interpret via critical value
        if abs(t_stat) <= cv:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
        # Interpret via p-value
        if p > alpha:
            print('Accept null hypothesis that the means are equal.')
        else:
            print('Reject the null hypothesis that the means are equal.')
            
    print("Simulation completed for algorithm: ", algorithm)
    print("==============================================")

# neuron_group.save_weight("saved/hierarchy_fixed_weight.npy")

##### Export result to Excel file ##########
if verbose != 0:
    result_path =  "result/result_" + datetime.now().strftime('%d%m%Y_%H%M%S')
    os.mkdir(result_path)
    result_pd = pd.DataFrame(result_dict) 
    row_list = []
    for problem_index in problem_list:
        row_list.append(problem_index)
    result_pd.index = row_list
    result_pd.to_excel(result_path+"/output.xlsx")
    
    param_file = open(result_path+"/param_setting.txt", "w")
    param_file.write("simulation: " + str(simulation) + "\n")
    param_file.write("time_delay: " + str(time_delay) + "\n")
    param_file.write("sequence_length: " + str(sequence_length) + "\n")
    param_file.close()
    
    if verbose == 2:
        prev_result = pd.read_excel('output.xlsx', engine='openpyxl')
        prev_result = prev_result.set_index('Unnamed: 0')
        prev_result.update(result_pd)
        prev_result.index.name = None
        prev_result.to_excel("output.xlsx")
##### Export result to Excel file ##########







