import numpy as np

import torch

# import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from tqdm import tqdm


def st_ed_col(target_col, col_names, col_dims):
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

class Classifier(Module):
    def __init__(self,input_dim, dis_dims, target_col, col_names, col_dims): # input_dim - number features, dis_dims - A list of integers, size(s) of a hidden layer(s), st_ed - indeces of columns for labels
        super(Classifier,self).__init__()
        # layer construction
        self.target_col = target_col
        self.col_names = col_names
        self.col_dims = col_dims
        st, ed = st_ed_col(self.target_col, self.col_names, self.col_dims)
        self.str_end = [st,ed]
        dim = input_dim-(self.str_end[1]-self.str_end[0]) # number of features without target column(s)
        seq = [] # list TO store the layers of the neural network sequentially

        for item in list(dis_dims):
            seq += [
                Linear(dim, item), #  fully connected layer with dim input features and item output features. dim is initially set to input_dim - (st_ed[1] - st_ed[0]) but will be .
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item # updated with each layer to match the number of output features from the previous layer
        
        # create output layer, depending on the sise of labels (one-hot encoded labels [0,0,1])
        #type of activation function used in the final layer depends on the task
        if (self.str_end[1]-self.str_end[0])==1:
            seq += [Linear(dim, 1)] # single unit with no activation function (a regression problem) 
        
        elif (self.str_end[1]-self.str_end[0])==2:
            seq += [Linear(dim, 1),Sigmoid()] # binary classification one-hot encoded => 2 columns, eather or => sigmoid
        else:
            seq += [Linear(dim,(self.str_end[1]-self.str_end[0]))]  # multi-class classification (will be later)
        
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
        
    def loss_classification(self, predications, labels):
        if (self.str_end[1] - self.str_end[0])==1:
            c_loss= SmoothL1Loss()
            labels = labels.type_as(predications)
            labels = torch.reshape(labels,predications.size())
        elif (self.str_end[1] - self.str_end[0])==2:
            c_loss = BCELoss()
            labels = labels.type_as(predications)
        else:
            c_loss = CrossEntropyLoss() 
        return c_loss(predications, labels)


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
                st, ed = st_ed_col(key, self.col_names, self.col_dims)
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
    


# selects option for each sample in batch, cumulative sum maybe to change 
def random_choice_prob_index(a, axis=1): # takes indices based on cumulative probability distributions in array of probabilities - not sampling from log-probs 
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis) # random value between 0 and 1 for each category in feature in batch
    #Returns the index of the first threshold where the cumulative sum exceeds the random number (r) - index of category for each sample in batch
    return (a.cumsum(axis=axis) > r).argmax(axis=axis) # Compares cumulative sum of probabilities with the random numbers

def random_choice_prob_index_sampling(probs,col_idx): # sampling from actual probs, not log - selects option for each sample in batch
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Cond_vector(object):
    def __init__(self, data, col_types, col_names, col_dims, categrical_labels):
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categrical_labels = categrical_labels
        self.cat_dims = []

        self.cat_col_dims = {}

        for key, value in self.col_types.items(): #categrical_labels
            if value=="categorical":
                st, ed = st_ed_col(key, self.col_names, self.col_dims)
                col_idx = self.col_names.index(key)
                self.cat_col_dims[key] = np.argmax(self.data[:, st:ed], axis=-1)
                if not self.cat_dims:
                    counter = 0
                else:
                    counter = self.cat_dims[len(self.cat_dims)-1][0]+self.cat_dims[len(self.cat_dims)-1][1]
                self.cat_dims.append([counter, self.col_dims[col_idx][1]])
            else:
                pass
               
        self.interval = [] #col_dims
        self.n_col = len(self.categrical_labels)   #Counter for the total number of categorical features in the data
        self.n_opt = sum(map(len, self.categrical_labels.values()))  #Counter for the total number of options in feature
        self.p_log = np.zeros((self.n_col, max(map(len, self.categrical_labels.values()))))  # stores log probs
        self.p_sampling = [] # stores the raw probabilities for sampling from the categories of a specific feature
        counter = 0
        for key, value in self.categrical_labels.items():
            st, ed = st_ed_col(key, self.col_names, self.col_dims)
        
            tmp_prob = np.sum(self.data[:, st:ed], axis=0) # probability of each category in the column
            tmp_prob = tmp_prob / np.sum(tmp_prob)
            self.p_sampling.append(tmp_prob)

            tmp_occur = np.sum(self.data[:, st:ed], axis=0)  # occurrences of each category
            tmp_occur = np.log(tmp_occur + 1) #(log(x + 1)) to smooth out the counts and normalize them into probabilities. This creates a less extreme distribution when sampling.
            tmp_log_prob = tmp_occur / np.sum(tmp_occur) # log-probability of each category in the column
            self.p_log[counter, :len(value)] = tmp_log_prob
            counter += 1
        
    def sample_train(self, batch): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None

        col_idx = np.random.choice(np.arange(self.n_col), batch) # Column selection for each sample in batch

        # here only categorical columns
        vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

        mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample ### maybe delete later
        mask[np.arange(batch), col_idx] = 1  ### maybe delete later

        opt1prime = random_choice_prob_index(self.p_log[col_idx]) # p_log[col_idx] - batch with selected columns, select option in the current column
        for i in np.arange(batch):
            vec[i, self.cat_dims[col_idx[i]][0] + opt1prime[i]] = 1
        return vec, mask, col_idx, opt1prime

    def sample(self, batch): # conditional vectors for testing / evaluation - using actual probabilities
        if self.n_col == 0:
            return None
      
        col_idx = np.random.choice(np.arange(self.n_col), batch) # choice of the column

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx) # choice of the option in the current column based on actual probs, (not log)
        
        for i in np.arange(batch):
            vec[i, self.cat_dims[col_idx[i]][0] + opt1prime[i]] = 1
            
        return vec