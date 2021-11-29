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

class SyncMap:
    
    def __init__(self, input_size, dimensions, adaptation_rate, noise=False):
        
        # np.random.seed(100)
        self.name = "SyncMap"
        self.organized= False
        self.space_size= 10
        self.dimensions= dimensions
        self.input_size= input_size
        self.syncmap= np.random.rand(input_size,dimensions) * self.space_size - (self.space_size/2)
        self.synapses_matrix = np.zeros([input_size, input_size])
        self.adaptation_rate= adaptation_rate
        
        self.ims = []
        self.fps = 0
        
        self.label_history = []
        
        # self.anim = True
        # if self.anim == True:
        #     self.fig = plt.figure()
        #     self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.syncmap_history = []
        
        self.noise_b = 0
        if noise is True:
            self.noise_b = 1
            self.name = "SyncMap with Noise"
        
    def input(self, x):
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
            
            center_plus= np.dot(vplus,self.syncmap)/plus_mass
            center_minus= np.dot(vminus,self.syncmap)/minus_mass
        
            dist_plus= distance.cdist(center_plus[None,:], self.syncmap, 'euclidean')
            dist_minus= distance.cdist(center_minus[None,:], self.syncmap, 'euclidean')
            dist_plus= np.transpose(dist_plus)
            dist_minus= np.transpose(dist_minus)
            
            #update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus + (self.syncmap - center_minus)/dist_minus)
            #update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus + (self.syncmap - center_plus)/dist_plus)
            update_plus= vplus[:,np.newaxis]*((center_plus - self.syncmap)/dist_plus)# + (self.syncmap - center_minus)/dist_minus)
            update_minus= vminus[:,np.newaxis]*((center_minus -self.syncmap)/dist_minus)# + (self.syncmap - center_plus)/dist_plus)
            
            
            noise = np.random.normal(0, 0.01, self.syncmap.shape)
            
            update= update_plus - update_minus
            self.syncmap+= self.adaptation_rate*update + noise*self.noise_b
            # self.noise_b *= 0.99
        
            maximum=self.syncmap.max()
            self.syncmap= self.space_size*self.syncmap/maximum

            
            self.syncmap_history.append(self.syncmap.copy())

    def organize(self, eps=3):
    
        self.organized= True
        self.labels= DBSCAN(eps=eps, min_samples=2).fit_predict(self.syncmap)

        return self.labels
    
    def hierarchical_organize(self, hierarchy=None):
    
        Z = linkage(self.syncmap, "single")
        # fig = plt.figure(dpi=150)
        # label_list = [i for i in range(1, self.input_size+1)]
        # dendrogram(Z, color_threshold=0, above_threshold_color='k', labels=label_list)
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
                label = AgglomerativeClustering(n_clusters=n_cluster, linkage='single').fit_predict(self.syncmap)
            labels[h,:] = label
            
        # self.labels = np.flip(labels, axis=0)        
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
            print("Activating a non-organized SyncMap")
            return
        
        #maximum output
        max_index= np.argmax(x)

        return self.labels[max_index]

    def plotSequence(self, input_sequence, input_class,filename="plot.png"):

        input_sequence= input_sequence[500:1000]
        input_class= input_class[500:1000]

        a= np.asarray(input_class)
        t = [i for i,value in enumerate(a)]
        c= [self.activate(x) for x in input_sequence] 
        

        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.k')
        #plt.ylim([-0.01,1.2])


        plt.savefig(filename,quality=1, dpi=300)
        plt.show()
        plt.close()
    

    # def plot(self, label, color=None, save = False, filename= "plot_map.png"):

    #     if color is None:
    #         color= label
        
    #     print(self.syncmap)
    #     #print(self.syncmap)
    #     #print(self.syncmap[:,0])
    #     #print(self.syncmap[:,1])
    #     if self.dimensions == 2:
    #         #print(type(color))
    #         #print(color.shape)
    #         ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
            
    #     if self.dimensions == 3:
    #         fig = plt.figure()
    #         ax = plt.axes(projection='3d')

    #         ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color)
    #         #ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
        
    #     if save == True:
    #         plt.savefig(filename)
        
    #     plt.show()
    #     plt.close()
    
    def plot(self, color=None, save = False, filename= "plot_map.png"):

        if color is None:
            color= self.labels
            
        #print(self.syncmap)
        #print(self.syncmap[:,0])
        #print(self.syncmap[:,1])
        if self.dimensions == 2:
            #print(type(color))
            #print(color.shape)
            fig = plt.figure(figsize=(5,5), dpi=150)
            ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], s=100, c=color, edgecolors='black')
            
        if self.dimensions == 3:
            fig = plt.figure(dpi=150)
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2]);
            #ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
        
        if save == True:
            plt.savefig(filename)
        
        plt.show()
        plt.close()
        
    def plot_w_line(self, label, color=None, save = False, filename= "plot_map.png"):

        if color is None:
            color= label
        
        print(self.syncmap)
        #print(self.syncmap)
        #print(self.syncmap[:,0])
        #print(self.syncmap[:,1])
        if self.dimensions == 2:
            #print(type(color))
            #print(color.shape)
            ax= plt.scatter(self.syncmap[:,0],self.syncmap[:,1], c=color)
            
        if self.dimensions == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2], c=color)
        
        if save == True:
            plt.savefig(filename)
        
        plt.show()
        plt.close()
        
    def TP_matrix(self, x):
        
        if len(x) == 0:
            return
        
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
            TPMatrix[i] = Dtable[i]/state_total
        
        self.TPMatrix = TPMatrix
        return TPMatrix

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
        np.save(path, self.syncmap)
        
    def animation_init(self):
        return self.img1, self.anos
        
    def animation_update(self, n):    
        # lab = DBSCAN(eps=1, min_samples=2).fit_predict(self.syncmap_history[n])
        if self.dimensions == 2:
            self.img1.set_offsets(self.syncmap_history[n])
        if self.dimensions == 3:
            self.img1._offsets3d = (self.syncmap_history[n,:,0], self.syncmap_history[n,:,1], self.syncmap_history[n,:,2])
        # self.img1.set_array(lab)
        # for i, txt in enumerate(self.true_labels):
        #     self.anos[i].set_x(self.syncmap_history[n,i,0])
        #     self.anos[i].set_y(self.syncmap_history[n,i,1])
        return self.img1, self.anos
    
    def plot_animation(self, name, true_labels=None):
        fig = plt.figure(dpi=150)
        if self.dimensions == 2:
            self.ax = fig.add_subplot()
        if self.dimensions ==3:
            self.ax = fig.add_subplot(111, projection='3d')
        
        self.syncmap_history = np.array(self.syncmap_history[::50])
        
        self.true_labels = true_labels
        self.anos = []
        # for i, txt in enumerate(self.true_labels):
        #     self.anos.append(self.ax.annotate(txt, (self.syncmap_history[0,i,0], self.syncmap_history[0,i,1])))
        
        ax_lim_max = np.max(self.syncmap_history)
        ax_lim_max = ax_lim_max + ax_lim_max * 0.1
        ax_lim_min = np.min(self.syncmap_history)
        ax_lim_min = ax_lim_min + ax_lim_min * 0.1

        self.ax.set_xlim(ax_lim_min, ax_lim_max)
        self.ax.set_ylim(ax_lim_min, ax_lim_max)
        if self.dimensions == 2:
            self.img1 = self.ax.scatter([],[], cmap="rgb")
        if self.dimensions ==3:
            self.ax.set_zlim(ax_lim_min, ax_lim_max)
            self.img1 = self.ax.scatter3D([],[],[], cmap="rgb")
        
        print("Generating animation...")
        ani = animation.FuncAnimation(fig, self.animation_update, frames=len(self.syncmap_history), init_func=self.animation_init, interval=30, blit=False)
        print("Generating animation completed!")
        print("Saving animation as MP4")
        animation_path =  "movies_output/movie_" + name + "_" + self.name + datetime.now().strftime('_%d%m%Y_%H%M%S') + ".mp4"
        ani.save(animation_path)
        print("Saving animation as MP4 completed!")
        
        self.syncmap_history = []
        
        
        

