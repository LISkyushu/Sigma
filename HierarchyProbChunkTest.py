from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt  
from sklearn.metrics.cluster import normalized_mutual_info_score  
import networkx as nx
import pygraphviz       
from networkx.drawing.nx_pydot import graphviz_layout

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

class HierarchyProbChunkTest:
    """
    Static hierarchical environment problem.
    This environment will generate sequence data embeded with hierarchical 
    structure in it.
    """
    
    def __init__(self, time_delay, filename=None, r=4, h=3):
        '''
        Chunks are written in the filename in which every line is a sequence of outputs followed by the number of the respective chunk
        All chunk numbers must be in ascending order and must have the same number of outputs
        Chunks will be shuffled and presented repeatedly throughout.
        When filename is not provided, environment will generetes a input
        with a hierachical structure as a balanced tree.
        '''
        self.name = "Hierarchy Probability Chunk Test"
        self.time_delay = time_delay
        self.time_counter = 0
        self.current_index= np.zeros(2)
        
        #================
        if filename != None:
            self.name += " with " + filename
            data_path = "data/"
            data=data_path+filename
            self.T = nx.Graph(nx.nx_agraph.read_dot(data))
        else:
            self.name += "_r="+str(r)+"_h="+str(h)
            self.T = nx.balanced_tree(r, h)
            mapping = {}
            for index, i in enumerate(reversed(range(len(self.T.nodes)))):
                mapping[i] = index
            self.T = nx.relabel_nodes(self.T, mapping)
                        
            total_leaf_node = pow(r,h)
            hierarchy_list = []
            for i in reversed(range(1, h)):
                hierarchy_list.append(pow(r, i))
                
            label_m = np.empty([total_leaf_node, h-1], dtype=int)
            for index, i in enumerate(hierarchy_list):
                partition = total_leaf_node/i
                counter = 0
                for k in range(total_leaf_node):
                    if k % partition == 0:
                        counter+=1
                    label_m[k][index] = counter
        
            for index, i in enumerate(label_m):
                true_label_str = np.array2string(i, separator=',')
                true_label_str = true_label_str[1:-1]
                self.T.nodes[index]['label'] = true_label_str      
                
            H = nx.Graph()
            H.add_nodes_from(sorted(self.T.nodes(data=True)))
            H.add_edges_from(self.T.edges(data=True))
            
            self.T = H
            mapping = {}
            for i in range(len(self.T.nodes)):
                mapping[i] = str(i)
            self.T = nx.relabel_nodes(self.T, mapping)
            
        plt.figure(figsize=(18, 8))
        nx.draw(self.T, pos = nx.spring_layout(self.T), node_size=300, alpha=0.5, node_color="blue", with_labels=True)
        plt.axis("equal")
        plt.show()
        
        self.output_size = len([i for i in list(self.T.nodes("label")) if i[1] != None])
        
        label= self.T.nodes(data="label")
        label = np.asarray(list(label))
        label = label[:self.output_size,1]
        label = [np.fromstring(label[i], dtype=int, sep=',') for i in range(len(label))]
        self.true_label = np.asmatrix(label).T - 1
        
        self.G = nx.empty_graph(self.output_size, create_using=nx.DiGraph)
            
        p = nx.shortest_path(self.T)        
        for i in range(self.output_size):
            for k in range(self.output_size):
                if i == k:
                    continue
                edge_weight = (len(p[str(i)][str(k)]) - 3) / 2
                edge_weight = 1/(pow(edge_weight+1, 3))
                self.G.add_weighted_edges_from([(i, k, edge_weight)])

        # plt.figure(figsize=(5,5))
        # edges,weights = zip(*nx.get_edge_attributes(self.G,'weight').items())
        # nx.draw(self.G, pos = nx.circular_layout(self.G), width=1.0, edge_color=weights, node_size=300, with_labels=True)
        # plt.show()
            
        self.A = nx.adj_matrix(self.G)
        self.A = self.A.todense()
        self.A = np.array(self.A, dtype = np.float64)        
        #================  

        for i in range(self.output_size):
            accum = self.A[i].sum()
            if accum != 0:
                self.A[i]= self.A[i]/accum
            else:
                print("ERROR: Node ",i," without connections from found")
                exit()
            # print(self.A[i])
        
        # Display transition probability of environment
        # fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        # # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        # ax.matshow(self.A, cmap='jet')
        
        # for (i, j), z in np.ndenumerate(self.A):
        #     ax.text(j, i, '{:.3f}'.format(z).lstrip('0'), ha='center', va='center', c='white', size=6)
        # plt.ylabel("State")
        # plt.xlabel("Next State")
        # plt.xticks(np.arange(0, len(self.A)))
        # plt.yticks(np.arange(0, len(self.A)))
        # labels = np.arange(1, len(self.A)+1)
        
        # ax.set_xticklabels(labels, fontsize=8)
        # ax.set_yticklabels(labels, fontsize=8)
        # ax.xaxis.set_ticks_position('bottom')
        # plt.show()

        #random start    
        self.output_class= np.random.randint(self.output_size)
        self.previous_output_class= None
        self.previous_previous_output_class= None
        self.previous_class_list = []
        self.previous_class_list_count = 5
        self.time_counter_list = np.zeros(self.previous_class_list_count)
        self.levels_count = self.true_label.shape[0]
            
    
    def getOutputSize(self):
        return self.output_size
    
    def trueLabel(self):
        return np.asarray(self.true_label)

    def updateTimeDelay(self):
        self.time_counter+= 1
        self.time_counter_list += 1
        if self.time_counter > self.time_delay:
            self.time_counter = 0 
            self.previous_previous_output_class= self.previous_output_class
            self.previous_output_class= self.output_class
            return True
        else:
            return False

    #create an input pattern for the system
    def getInput(self, reset = False):
    
        update = self.updateTimeDelay()
        
        if update == True:
        
            self.previous_output_class= self.output_class
            self.output_class= np.random.choice(self.output_size ,1 ,p= self.A[self.output_class])[0]
            self.previous_class_list.append(self.previous_output_class)
            if len(self.previous_class_list)>self.previous_class_list_count:
                self.time_counter_list = self.time_counter_list[1:]
                self.time_counter_list = np.append(self.time_counter_list, 0)
                self.previous_class_list.pop(0)
        
        if self.previous_output_class is None or self.previous_output_class == self.output_class:
            input_value = np_utils.to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter)
        else:
            input_value = np_utils.to_categorical(self.output_class, self.output_size)*np.exp(-0.1*self.time_counter) 
            for t, i in enumerate(self.previous_class_list):
                input_value +=  np_utils.to_categorical(i, self.output_size)*np.exp(-0.1*(self.time_counter_list[t]+self.time_delay))

        return input_value

    def getSequence(self, sequence_size):
    
        #print(self.data.shape[0])
        #print(input_sequence.shape)
        #exit()
        self.input_sequence = np.empty((sequence_size, self.output_size))
        self.input_class = np.empty((sequence_size, self.true_label.shape[0]), dtype=int)
        
        for i in range(sequence_size):
            
            input_value = self.getInput()
            
            #input_class.append(self.chunk)
            #input_sequence.append(input_value)
            self.input_class[i] = self.true_label.T[self.output_class].tolist()[0]
            self.input_sequence[i] = input_value

        return self.input_sequence, self.input_class

    
    def plot(self, input_class, input_sequence = None, save = False):
        
        a = np.asarray(input_class)
        t = [i for i,value in enumerate(a)]

        plt.plot(t, a)
        
        if input_sequence != None:
            sequence = [np.argmax(x) for x in input_sequence]
            plt.plot(t, sequence)

        if save == True:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
    
    def plotSuperposed(self, input_class, input_sequence = None, save = False):
    
        input_sequence= np.asarray(input_sequence)
        
        t = [i for i,value in enumerate(input_sequence)]

        #exit()

        for i in range(input_sequence.shape[1]):
            a = input_sequence[:,i]
            plt.plot(t, a)
        
        a = np.asarray(input_class)
        plt.plot(t, a)

        if save == True:
            plt.savefig("plot.png")
        
        plt.show()
        plt.close()
        
    def TP_matrix(self):
        
        if len(self.input_sequence) == 0:
            return
        
        Dtable = np.zeros([self.output_size, self.output_size])
        TPMatrix = np.zeros([self.output_size, self.output_size])
        
        prev_state = np.argmax(self.input_sequence[0])
        state = None
        for i in self.input_sequence[1:]:
            if np.max(i) != 1:
                continue
            state = np.argmax(i)
            Dtable[prev_state][state]+= 1                
            prev_state = state
                
        for i, j in enumerate(Dtable):
            state_total = np.sum(j)
            TPMatrix[i] = Dtable[i]/state_total
            
        # fig, ax = plt.subplots()
        # plt.imshow(TPMatrix, interpolation='none')
        # plt.show()
        # plt.ylabel("State")
        # plt.xlabel("Next State")
        # plt.xticks(np.arange(0, self.output_size))
        # plt.yticks(np.arange(0, self.output_size))
        # plt.colorbar()
        # labels = np.arange(1, self.output_size+1)
        
        # ax.set_xticklabels(labels)
        # ax.set_yticklabels(labels)
        
        return TPMatrix
    
    def evaluation(self, label):
        return normalized_mutual_info_score(label, self.trueLabel())

