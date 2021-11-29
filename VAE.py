from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Dropout
from tensorflow.keras.models import Model
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, maxdists
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from sklearn.metrics.cluster import normalized_mutual_info_score

class CallbackSaveHistory(keras.callbacks.Callback):
    def __init__(self, encoder, output_size):
        self.his = []
        self.encoder = encoder
        self.output_size = output_size
        self.map = np.zeros((self.output_size,3))
    
    def on_train_batch_end(self, batch, logs=None):
        self.his.append(self.createMap().copy())
        # print(batch)
        
    def return_history(self):
        return self.his
    
    def createMap(self):
        all_possible_inputs = [np_utils.to_categorical(i, self.output_size) for i in range(self.output_size)]
        for i,a in enumerate(all_possible_inputs):
            sample = a[None,:]
            #print("a shape", a.shape)
            predicted= self.encoder.predict(sample)
            #print(predicted)
            self.map[i] = predicted
        return self.map


class VAE:
    
    def __init__(self, input_size, latent_dim, timesteps):
        # np.random.seed(100)
        self.name = "VAE"
        self.organized= False
        self.latent_dim= latent_dim
        self.input_size= input_size
        self.timesteps = timesteps
        self.counter = 0
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        # self.label_history = []

        self.map = np.zeros((self.input_size,latent_dim))
        self.createModel()
        
        self.createMap()
        self.history = []
        
    
    def createModel(self):
        
        self.createConvModel()
        
    def createConvModel(self):

        input_shape= (self.input_size)
        input_layer = Input(shape=input_shape)
        layer = Dense(self.latent_dim)(input_layer)
        #layer = Droupout(0.4)(layer)
        #layer = Dense(latent_dim)(layer)
        output = Dense(self.input_size,activation='sigmoid')(layer)
        
        self.model = Model(input_layer, output)
        self.encoder = Model(input_layer, layer)


    def createLSTMModel(self):
        
        inputs = Input(shape=(self.timesteps, self.input_size))
        encoded = LSTM(self.latent_dim)(inputs)

        decoded = RepeatVector(self.timesteps)(encoded)
        decoded = LSTM(self.input_size, return_sequences=True)(decoded)

        
        self.model = Model(inputs, decoded)
        self.encoder = Model(inputs, encoded)

    def input(self, x):
        
        # sample_size = 1999
        # sampling_rate = 300
        learning_rate= 1e-3
        epochs=10
        batch_size=640
        loss= "mean_squared_error"
        
        # convert to n-gram or skip-gram
        for i,sample in enumerate(x):
            if i - self.timesteps >= 0:
                position = int(self.timesteps/2)
                y = x[i-position]
                a = i-self.timesteps
                b = i-position
                c = i-position+1
                d = i+1

                sample = x[np.r_[a:b,c:d]]
                sample = np.sum(sample, axis=0)
                sample/= sample.sum()

                self.x_train.append(y)
                self.y_train.append(sample)

        

        self.createModel()
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

        np_x_train= np.array(self.x_train)
        np_y_train= np.array(self.y_train)
        
        # save_history = CallbackSaveHistory(self.encoder)

        self.model.fit(
            np_x_train,
            np_y_train,
            epochs=epochs,
            #validation_data=(x_val, val_labels),
            verbose=0,  # Logs once per epoch.
            batch_size=batch_size#,
            # callbacks=[save_history]
            )


        #print(len(self.dataset))
        #print(self.dataset[0])
        #k= self.dataset[0]
        #print(k[0].shape)
        
        # self.history = save_history.return_history()
            
    def createMap(self):
        
        all_possible_inputs = [np_utils.to_categorical(i, self.input_size) for i in range(self.input_size)]
        for i,a in enumerate(all_possible_inputs):
            sample = a[None,:]
            #print("a shape", a.shape)
            predicted= self.encoder.predict(sample)
            #print(predicted)
            self.map[i] = predicted
        return self.map

    def organize(self, eps=0.5):
    
        self.organized= True
        self.createMap()
        self.labels= DBSCAN(eps=eps, min_samples=2).fit_predict(self.map)

        return self.labels
    
    def hierarchical_organize(self, hierarchy=None):
        self.createMap()
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

        input_sequence= input_sequence[1:500]
        input_class= input_class[1:500]

        a= np.asarray(input_class)
        t = [i for i,value in enumerate(a)]
        c= [self.activate(x) for x in input_sequence] 
        

        plt.plot(t, a, '-g')
        plt.plot(t, c, '-.k')
        #plt.ylim([-0.01,1.2])


        plt.savefig(filename,quality=1, dpi=300)
        plt.show()
        plt.close()
    

    def plot(self, color=None, save = False, filename= "plot_map.png"):

        if color is None:
            color= self.labels
        
        #print(self.syncmap)
        #print(self.syncmap[:,0])
        #print(self.syncmap[:,1])
        if self.latent_dim == 2:
            #print(type(color))
            #print(color.shape)
            fig = plt.figure(figsize=(5,5), dpi=150)
            ax= plt.scatter(self.map[:,0],self.map[:,1], s=100, c=color, edgecolors='black')
            
        if self.latent_dim == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            ax.scatter3D(self.map[:,0],self.map[:,1], self.map[:,2], c=color);
            #ax.plot3D(self.syncmap[:,0],self.syncmap[:,1], self.syncmap[:,2])
        
        if save == True:
            plt.savefig(filename)
        
        plt.show()
        plt.close()
        
    def evaluation(self, true_label, self_label):    
        hierarchy = None                       
        labels = self.hierarchical_organize(hierarchy)
        print("number of layers it thinks: " + str(len(labels)))
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
        
    def animation_init(self):
        return self.img1, self.anos
    
    def animation_update(self, n):      
        if self.latent_dim == 2:
            self.img1.set_offsets(self.history[n])
        if self.latent_dim == 3:
            self.img1._offsets3d = (self.history[n,:,0], self.history[n,:,1], self.history[n,:,2])
        return self.img1, self.anos
    
    def plot_animation(self, name, true_labels=None):
        return
    
        fig = plt.figure(dpi=150)
        if self.latent_dim == 2:
            self.ax = fig.add_subplot()
        if self.latent_dim ==3:
            self.ax = fig.add_subplot(111, projection='3d')
        
        self.history = np.array(self.history)
        
        self.true_labels = true_labels
        self.anos = []
        # if true_labels is not None:
        #     for i, txt in enumerate(self.true_labels):
        #         self.anos.append(self.ax.annotate(txt, (self.syncmap_history[0,i,0], self.syncmap_history[0,i,1])))
        
        ax_lim_max = np.max(self.history)
        ax_lim_max = ax_lim_max + ax_lim_max * 0.1
        ax_lim_min = np.min(self.history)
        ax_lim_min = ax_lim_min + ax_lim_min * 0.1

        self.ax.set_xlim(ax_lim_min, ax_lim_max)
        self.ax.set_ylim(ax_lim_min, ax_lim_max)
        if self.latent_dim == 2:
            self.img1 = self.ax.scatter([],[], cmap="rgb")
        if self.latent_dim ==3:
            self.ax.set_zlim(ax_lim_min, ax_lim_max)
            self.img1 = self.ax.scatter3D([],[],[], cmap="rgb")
        
        print("Generating animation...")
        ani = animation.FuncAnimation(fig, self.animation_update, frames=len(self.history), init_func=self.animation_init, interval=30, blit=False)
        print("Generating animation completed!")
        print("Saving animation as MP4")
        animation_path =  "movies_output/movie_" + name + "_" + self.name + datetime.now().strftime('_%d%m%Y_%H%M%S') + ".mp4"
#         animation_path =  "movie.mp4"
        ani.save(animation_path)
        print("Saving animation as MP4 completed!")
        
        self.syncmap_history = []
        

