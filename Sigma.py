from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from datetime import datetime
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, maxdists
from sklearn.metrics.cluster import normalized_mutual_info_score

class Sigma:
    
    def __init__(self, input_size, dimensions, adaptation_rate, sigma1=6, sigma2=3, sigma1_r=0, sigma2_r=2, noise=False):
    
        # np.random.seed(100)
        self.name = "Sigma"
        self.organized= False
        self.space_size= 10 # scaling dimension of sigma space/map
        self.dimensions= dimensions # dimension of sigma space/map
        self.input_size= input_size # number of variables
        
        # Initialize weights in sigma space/map
        self.map= np.random.rand(input_size,dimensions) * self.space_size - (self.space_size/2)

        # Initialize velocity of each weights
        self.velocity= np.zeros((input_size,dimensions))
        
        # Learning rate
        self.adaptation_rate= adaptation_rate

        # Force coefficient
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma1_r = sigma1_r
        self.sigma2_r = sigma2_r
        
        # Recording weights/map history for animation
        self.map_history = []
        
    def input(self, x):
        """
        Sigma takes sequence of input x as input and update its weights on 
        every iteration.
        """
        plus= x > 0.1
        minus = ~ plus
        sequence_size = x.shape[0]

        for i in range(sequence_size):
            
            vplus= plus[i,:]
            vminus= minus[i,:]
            plus_mass = vplus.sum()
            minus_mass = vminus.sum()

            if plus_mass <= 1:
                continue
            
            if minus_mass <= 1:
                continue
            
            center_plus= np.dot(vplus,self.map)/plus_mass
            center_minus= np.dot(vminus,self.map)/minus_mass

            dist_plus= distance.cdist(center_plus[None,:], self.map, 'euclidean')
            dist_minus= distance.cdist(center_minus[None,:], self.map, 'euclidean')
            dist_plus= np.transpose(dist_plus)
            dist_minus= np.transpose(dist_minus)
                        
            update_plus= vplus[:,np.newaxis]*(self.sigma1*(center_plus - self.map)/dist_plus - self.sigma1_r*(self.map - center_minus)/dist_minus) 
            update_minus= vminus[:,np.newaxis]*(self.sigma2*(center_minus - self.map)/dist_minus - self.sigma2_r*(self.map - center_plus)/pow(dist_plus, 2))
            
            # Damping force
            self.velocity*= 0.999
            
            update= update_plus - update_minus
            self.velocity += update
            self.map+= self.adaptation_rate*self.velocity

            # Normalization
            maximum=self.map.max()
            self.map= self.space_size * self.map/maximum
            
            self.map_history.append(self.map.copy())
        
    def organize(self, eps=3):
        """
        Extract chunking information using DBSCAN
        """
        self.organized= True
        self.label= DBSCAN(eps=eps, min_samples=2).fit_predict(self.map)

        return self.label
    
    def hierarchical_organize(self, hierarchy=None):
        """
        Extract hierarchical structure (chunks on each hierarchy) using
        Hierarchical Clustering with single linkage, and determine which
        levels of hierarchy are important. the output label, a matrix 
        include the number of hierarchies and chunking on each hierarchies.        
        """
        method = "single"
        Z = linkage(self.map, method)
        # fig = plt.figure(dpi=150)
        # label_list = [i for i in range(1, self.input_size+1)]
        # dendrogram(Z, labels=label_list)
        
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
            
        labels = np.empty((total_hierarchy, self.input_size), dtype=int)
        for h in range(total_hierarchy):
            label = [-1 for _ in range(self.input_size)]
            if h < len(d_diff_index):    
                n_cluster = self.input_size - d_diff_index[h] - 1
                label = AgglomerativeClustering(n_clusters=n_cluster, linkage=method).fit_predict(self.map)
            labels[h,:] = label
   
        self.label = labels     

        return self.label
    
    def evaluation(self, true_label, self_label):    
        """
        Evaluate predicted label using NMI score
        Return the average NMI score of every hierarchies.
        True label provided by environment
        If predicted label contains more hierarchies than true label, compare
        only the most important hierarchies in predicted label.
        If predited label contains less hierarchies than true label, insert
        a new hierarchies filled with -1; Yielding 0 NMI score for the 
        particular hierarchies when comparing to true label.
        """
        hierarchy = None                       
        labels = self_label
        print("number of predicted level of hierarchies: " + str(len(labels)))
        total_env_hierarchy = len(true_label)
        true_label = np.flip(true_label, axis=0)
        
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

    def activate(self, x):
        '''
        Return the label of the index with maximum input value
        '''
        if self.organized == False:
            print("Activating a non-organized map")
            return
        
        #maximum output
        max_index= np.argmax(x)

        return self.labels[max_index]   

    def save(self, filename):
        """save class as self.name.txt"""
        file = open(filename+'.txt','w')
        file.write(cPickle.dumps(self.__dict__))
        file.close()

    def load(self, filename):
        """try load self.name.txt"""
        file = open(filename+'.txt','r')
        dataPickle = file.read()
        file.close()

        self.__dict__ = cPickle.loads(dataPickle)
    
    def save_weight(self, path):
        np.save(path, self.map)
    
    def animation_init(self):
        return self.img1, self.anos
    
    def animation_update(self, n):      
        if self.dimensions == 2:
            self.img1.set_offsets(self.map_history[n])
        if self.dimensions == 3:
            self.img1._offsets3d = (self.map_history[n,:,0], self.map_history[n,:,1], self.map_history[n,:,2])
        return self.img1, self.anos
    
    def plot_animation(self, name, true_labels=None):
        """
        Produce an animation (MP4) that shows the dynamic of Sigma
        throughout the time step. Only works when Sigma space dimension
        equals to 2 or 3.
        """
        fig = plt.figure(dpi=150)
        if self.dimensions == 2:
            self.ax = fig.add_subplot()
        if self.dimensions ==3:
            self.ax = fig.add_subplot(111, projection='3d')
            # self.ax.axes.xaxis.set_ticks([])
            # self.ax.axes.yaxis.set_ticks([])
            # self.ax.axes.zaxis.set_ticks([])
            # self.ax.grid(True)
        
        sampling_rate = 50
        self.map_history = np.array(self.map_history[::sampling_rate])
        
        self.true_labels = true_labels
        self.anos = []
        ax_lim_max = np.max(self.map_history)
        ax_lim_max = ax_lim_max + ax_lim_max * 0.1
        ax_lim_min = np.min(self.map_history)
        ax_lim_min = ax_lim_min + ax_lim_min * 0.1

        self.ax.set_xlim(ax_lim_min, ax_lim_max)
        self.ax.set_ylim(ax_lim_min, ax_lim_max)
        if self.dimensions == 2:
            self.img1 = self.ax.scatter([],[], cmap="rgb")
        if self.dimensions ==3:
            self.ax.set_zlim(ax_lim_min, ax_lim_max)
            self.img1 = self.ax.scatter3D([],[],[], cmap="rgb")
        
        print("Generating animation...")
        ani = animation.FuncAnimation(fig, self.animation_update, frames=len(self.map_history), init_func=self.animation_init, interval=30, blit=False)
        print("Generating animation completed!")
        print("Saving animation as MP4")
        animation_path =  "movies_output/movie_" + name + "_" + self.name + datetime.now().strftime('_%d%m%Y_%H%M%S') + ".mp4"
        ani.save(animation_path)
        print("Saving animation as MP4 completed!")
        
        self.map_history = []
    
    
    

