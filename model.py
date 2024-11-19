import numpy as np

import torch

# import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from tqdm import tqdm

class Classifier(Module):
    def __init__(self,input_dim, dis_dims,st_ed): # input_dim - number features, dis_dims - A list of integers, size(s) of a hidden layer(s), st_ed - indeces of columns for labels
        super(Classifier,self).__init__()
        # layer construction

        dim = input_dim-(st_ed[1]-st_ed[0]) # number of features without target column(s)
        seq = [] # list TO store the layers of the neural network sequentially
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [
                Linear(dim, item), #  fully connected layer with dim input features and item output features. dim is initially set to input_dim - (st_ed[1] - st_ed[0]) but will be .
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item # updated with each layer to match the number of output features from the previous layer
        
        # create output layer, depending on the sise of labels (one-hot encoded labels [0,0,1])
        #type of activation function used in the final layer depends on the task
        if (st_ed[1]-st_ed[0])==1:
            seq += [Linear(dim, 1)] # single unit with no activation function (a regression problem) 
        
        elif (st_ed[1]-st_ed[0])==2:
            seq += [Linear(dim, 1),Sigmoid()] # binary classification one-hot encoded => 2 columns, eather or => sigmoid
        else:
            seq += [Linear(dim,(st_ed[1]-st_ed[0]))]  # multi-class classification (will be later)
        
        # Sequential container that stacks all the layers defined in seq.
        # nn.Sequential allows you to pass an input through the layers one by one in the defined order. 
        # The *seq syntax unpacks the list seq and passes each layer to nn.Sequential.
        self.seq = Sequential(*seq)

    def forward(self, input):
        
        label=None
        
        if (self.str_end[1]-self.str_end[0])==1: # one target column
            label = input[:, self.str_end[0]:self.str_end[1]] # extracted directly
        else:
            label = torch.argmax(input[:, self.str_end[0]:self.str_end[1]], axis=-1) # finds the index of the highest value in given columns, which is the predicted class.+
        
        new_imp = torch.cat((input[:,:self.str_end[0]],input[:,self.str_end[1]:]),1) # removes the target columns and concatenates the parts 
        
        # pass new input through layers
        if ((self.str_end[1]-self.str_end[0])==2) | ((self.str_end[1]-self.str_end[0])==1):
            return self.seq(new_imp).view(-1), label #  flatten
        else:
            return self.seq(new_imp), label
        

class Sampler(object):
    def __init__(self, data, col_types, col_names, col_dims, categrical_labels):
        super(Sampler, self).__init__()
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categrical_labels = categrical_labels
        self. cat_ones_row_idx = {} # indeces of rows with one for categorical columns

        for key, value in self.col_types.items(): #categrical_labels
            if value=="categorical":
                col_idx = self.col_names.index(key)
                st = self.col_dims[col_idx][0]
                ed = self.col_dims[col_idx][0] + self.col_dims[col_idx][1]
                tmp = []
                for j in range(ed-st):
                    tmp.append(np.nonzero(np.array(data)[:, st+j]))
                self.cat_ones_row_idx[key] = tmp
            else:
                pass
            
    # n: The number of samples to generate.
    # col: Specifies which categorical features to consider during sampling.
    # opt: Specifies which categories (for the features in col) should be used.
    def sample(self, n, col=None, opt=None): # number of samples, column, category
        if col is None:
            idx = np.random.choice(np.arange(self.data_len), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            o_idx = list(self.categrical_labels[c]).index(o)
            idx.append(np.random.choice(self.cat_ones_row_idx[c][o_idx][0])) # get random row by the condition
        return self.data[idx]
    


class Cond(object):
    def __init__(self, data, col_types, col_names, col_dims, categrical_labels):
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categrical_labels = categrical_labels

        self.cat_idx = {}

        for key, value in self.col_types.items(): #categrical_labels
            if value=="categorical":
                col_idx = self.col_names.index(key)
                st = self.col_dims[col_idx][0]
                ed = self.col_dims[col_idx][0] + self.col_dims[col_idx][1]
                # self.cat_idx.append(np.argmax(np.array(transf_data)[:, st:ed], axis=-1))
                self.cat_idx[key] = np.argmax(np.array(transf_data)[:, st:ed], axis=-1)
            else:
                pass
               
        self.interval = [] #col_dims
        self.n_col = len(self.categrical_labels)   #Counter for the total number of categorical features in the data
        self.n_opt = sum(map(len, self.categrical_labels.values()))  #Counter for the total number of options in feature
        self.p_log = []
        self.p_sampling = [] # stores the raw probabilities for sampling from the categories of a specific feature
        
        for key in self.categrical_labels:
            col_idx = self.col_names.index(key)
            st = self.col_dims[col_idx][0]
            ed = self.col_dims[col_idx][0] + self.col_dims[col_idx][1]

            tmp_prob = np.sum(data[:, st:ed], axis=0) # probability of each category in the column
            tmp_prob = tmp_prob / np.sum(tmp_prob)
            self.p_sampling.append(tmp_prob)

            tmp_occur = np.sum(data[:, st:ed], axis=0)  # occurrences of each category
            tmp_occur = np.log(tmp_occur + 1) #(log(x + 1)) to smooth out the counts and normalize them into probabilities. This creates a less extreme distribution when sampling.
            tmp_log_prob = tmp_occur / np.sum(tmp_occur) # log-probability of each category in the column
            self.p_log.append(tmp_log_prob)