from keras.utils import np_utils
import numpy as np
import math
import matplotlib.pyplot as plt           

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

class HierarchyFixedChunkTest:
    
    def __init__(self, time_delay, filename="hierarchy_chunk.txt"):
        '''
        Chunks are written in the filename in which every line is a sequence of outputs followed by the number of the respective chunk
        All chunk numbers must be in ascending order and must have the same number of outputs
        Chunks will be shuffled and presented repeatedly throughout
        '''
        self.name = "Hierarchy Fixed Chunk Test"
        
        data_path = "data/"
        data=data_path+filename
        
        dataset= np.loadtxt(data, dtype="i", delimiter=",")
        self.time_delay = time_delay
        self.time_counter = 0
        self.current_index= np.zeros(2)

        self.output_size= dataset.shape[1]-2
        self.data = dataset[:,:self.output_size]
        self.data_class= dataset[:,self.output_size:]
        self.chunk_dimension = self.data_class.shape[1]
        
        self.current_chunk_id = np.zeros(self.chunk_dimension+1, dtype=np.int8)
        
        # TODO: HARD-CODED! FIX!
        self.data_hierarchy = []
        self.data_hierarchy = [[],[]]
        self.data_hierarchy[0] = [[], [], []]
        self.data_hierarchy[1] = [[], [], []]
        
        data_index = 0
        for i in range(2):
            for j in range(3):
                for k in range(3):
                    self.data_hierarchy[i][j].append(self.data[data_index])
                    data_index += 1
                    
        self.chunks_len = dim(self.data_hierarchy)
        self.chunks_len = np.flip(self.chunks_len)
        
        self.true_labels = self.data_class-1
        
        self.previous_output_class= None
        self.previous_previous_output_class= None
            
    
    def getOutputSize(self):
        return self.output_size
    
    def trueLabel(self):
        return self.true_labels.T

    def updateTimeDelay(self):
        self.time_counter+= 1
        if self.time_counter > self.time_delay:
            self.time_counter = 0 
            self.previous_previous_output_class= self.previous_output_class
            self.previous_output_class= self.output_class
            return True
        else:
            return False

    #create an input pattern for the system
    def getInput(self, reset = False):
        
        if reset == True:
            self.current_index=0
            self.time_counter=0

        update = self.updateTimeDelay()
        if update == True:
            self.current_chunk_id[0] += 1

            #check if a new chunk should start
            if self.current_chunk_id[0] >= self.chunks_len[0]:
                
                # change chunk
                self.current_chunk_id[1] = np.random.randint(self.chunks_len[1])
                
                if np.random.randint(100) < 30:
                    self.current_chunk_id[2] = np.random.randint(self.chunks_len[2])
                
                self.current_chunk_id[0] = 0
                
        self.output_class = self.data_hierarchy[self.current_chunk_id[2]][self.current_chunk_id[1]][self.current_chunk_id[0]]
        
        noise_intensity= 0
        if self.previous_output_class is None or np.array_equal(self.previous_output_class, self.output_class):
            input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity
        else:
            input_value = self.output_class*np.exp(-0.1*self.time_counter) + np.random.randn(self.output_size)*noise_intensity + self.previous_output_class*np.exp(-0.1*(self.time_counter+self.time_delay))

        return input_value

    def getSequence(self, sequence_size):
    
        #print(self.data.shape[0])
        #print(input_sequence.shape)
        #exit()
        self.input_sequence = np.empty((sequence_size, self.data.shape[1]))
        self.input_class = [None] * sequence_size
        
        for i in range(sequence_size):
            
            input_value = self.getInput()
            
            #input_class.append(self.chunk)
            #input_sequence.append(input_value)
            current_chunk_id = (self.current_chunk_id+1)
            input_class_id = current_chunk_id[1] * current_chunk_id[2]
            tmp = [0,0,0,1,1,1]
            self.input_class[i] = [input_class_id-1, tmp[input_class_id-1]]
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
            
        return TPMatrix
        
        
CHUNK = HierarchyFixedChunkTest(10)
y = CHUNK.getSequence(100000)
tpmatrix = CHUNK.TP_matrix()
# fig, ax = plt.subplots()
# plt.imshow(tpmatrix, interpolation='none')
# plt.ylabel("State")
# plt.xlabel("Next State")
# plt.colorbar()
# plt.xticks(np.arange(0, CHUNK.output_size))
# plt.yticks(np.arange(0, CHUNK.output_size))

# labels = np.arange(1, CHUNK.output_size+1)

# ax.set_xticklabels(labels)
# ax.set_yticklabels(labels)

# plt.show()

