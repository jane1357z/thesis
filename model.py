import numpy as np

import torch

# import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Sigmoid, BCELoss, CrossEntropyLoss,SmoothL1Loss, BatchNorm1d)

from tqdm import tqdm


def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

class Classifier(Module):
    def __init__(self,input_dim, dis_dims, target_col, col_names, col_dims): # input_dim - number features, dis_dims - A list of integers, size(s) of a hidden layer(s)
        super(Classifier,self).__init__()
        # layer construction
        self.target_col = target_col
        self.col_names = col_names
        self.col_dims = col_dims
        st, ed = st_ed_col(self.target_col, self.col_names, self.col_dims)
        self.str_end = [st,ed]
        dim = input_dim-(self.str_end[1]-self.str_end[0]) # number of features without target column(s)
        seq = [] # list to store the layers of the neural network sequentially
        
        for item in list(dis_dims):
            seq += [
                Linear(dim, item), #  fully connected layer with dim input features and item output features. dim is initially set to input_dim - (st_ed[1] - st_ed[0])
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
        
        # stacks all the layers to pass an input through the layers 
        self.seq = Sequential(*seq)

    def forward(self, input_d):
        
        label=None
        
        if (self.str_end[1]-self.str_end[0])==1: # one target column
            label = input_d[:, self.str_end[0]:self.str_end[1]] # extracted directly
        else:
            label = torch.argmax(input_d[:, self.str_end[0]:self.str_end[1]], axis=-1) # finds the index of the highest value in given columns, which is the predicted class
        
        new_imp = torch.cat((input_d[:,:self.str_end[0]],input_d[:,self.str_end[1]:]),1) # removes the target columns and concatenates the parts 
        
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
    def __init__(self, data, col_types, col_names, col_dims, categorical_labels):
        super(Sampler, self).__init__()
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categorical_labels = categorical_labels
        self.cat_ones_row_idx = []

        for key, value in self.col_types.items(): #categorical_labels
            if value=="categorical":
                st, ed = st_ed_col(key, self.col_names, self.col_dims)
                tmp = []
                for j in range(ed-st):
                    tmp.append(np.nonzero(np.array(data)[:, st+j]))
                self.cat_ones_row_idx.append(tmp)
            else:
                pass
            
    # n: the number of samples to generate.
    # col: which categorical features to consider during sampling
    # opt:  which categories (for the features in col) should be used
    def sample(self, n, col=None, opt=None): # number of samples, column, category
        if col is None:
            idx = np.random.choice(np.arange(self.data_len), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.cat_ones_row_idx[c][o][0])) # get random row by the condition
            print(idx)
        return self.data[idx]
    


# selects option for each sample in batch, cumulative sum maybe to change 
def random_choice_prob_index(a, axis=1): # takes indices based on cumulative probability distributions in array of probabilities - not sampling from log-probs 
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis) # random value between 0 and 1 for each category in feature in batch
    # eeturns the index of the first threshold where the cumulative sum exceeds the random number (r) - index of category for each sample in batch
    return (a.cumsum(axis=axis) > r).argmax(axis=axis) # compares cumulative sum of probabilities with the random numbers

def random_choice_prob_index_sampling(probs,col_idx): # sampling from actual probs, not log - selects option for each sample in batch
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
    return np.array(option_list).reshape(col_idx.shape)

class Cond_vector(object):
    def __init__(self, data, col_types, col_names, col_dims, categorical_labels):
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categorical_labels = categorical_labels
        self.cat_col_dims = []


        for key, value in self.col_types.items(): #categorical_labels
            if value=="categorical":
                st, ed = st_ed_col(key, self.col_names, self.col_dims)
                col_idx = self.col_names.index(key)
                if not self.cat_col_dims:
                    counter = 0
                else:
                    counter = self.cat_col_dims[len(self.cat_col_dims)-1][0]+self.cat_col_dims[len(self.cat_col_dims)-1][1]
                self.cat_col_dims.append([counter, self.col_dims[col_idx][1]])
            else:
                pass
               
        self.n_col = len(self.categorical_labels)   # total number of categorical features in the data
        self.n_opt = sum(map(len, self.categorical_labels.values()))  # total number of options in feature
        self.p_log = np.zeros((self.n_col, max(map(len, self.categorical_labels.values()))))  # stores log probs
        self.p_sampling = [] # stores the raw probabilities for sampling from the categories of a specific feature
        counter = 0
        for key, value in self.categorical_labels.items():
            st, ed = st_ed_col(key, self.col_names, self.col_dims)
        
            tmp_prob = np.sum(self.data[:, st:ed], axis=0) # probability of each category in the column
            tmp_prob = tmp_prob / np.sum(tmp_prob)
            self.p_sampling.append(tmp_prob)

            tmp_occur = np.sum(self.data[:, st:ed], axis=0)  # occurrences of each category
            tmp_occur = np.log(tmp_occur + 1) # (log(x + 1)) to smooth out the counts and normalize them into probabilities, creates a less extreme distribution when sampling.
            tmp_log_prob = tmp_occur / np.sum(tmp_occur) # log-probability of each category in the column
            self.p_log[counter, :len(value)] = tmp_log_prob
            counter += 1
        
    def sample_train(self, batch): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None

        col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch

        # here only categorical columns
        vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

        mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
        mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection

        opt1prime = random_choice_prob_index(self.p_log[col_idx]) # p_log[col_idx] - batch with selected columns, select option in the current column
        for i in np.arange(batch):
            vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1
        return vec, mask, col_idx, opt1prime

    def sample(self, batch): # conditional vectors for testing / evaluation - using actual probabilities
        if self.n_col == 0:
            return None
      
        col_idx = np.random.choice(np.arange(self.n_col), batch) # choice of the column

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx) # choice of the option in the current column based on actual probs, (not log)
        
        for i in np.arange(batch):
            vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1
            
        return vec
    
class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10): # input_dim - features, discriminator_dim - hidden layers size
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in discriminator_dim:
            seq += [Linear(dim, item),
                    LeakyReLU(0.2),
                    Dropout(0.5)]
            dim = item # updated with each layer to match the number of output features from the previous layer

        # output layer
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)
        self.seq_info = Sequential(*seq[:-1]) # to exclude last layer

    def calc_gradient_penalty(self, real_data, fake_data,lambda_=10): # lambda_ - hyperparam for GP / regularization term
        # ctab-gan use spherical linear interpolation (SLERP)
        alpha = torch.rand(real_data.size(0) // self.pac, 1, 1) # interpolation
        alpha = alpha.repeat(1, self.pac, real_data.size(1)) # same random value is used across the feature dimension, interpolation factor is repeated pac times for each batch element
        alpha = alpha.view(-1, real_data.size(1)) # change the size wrt pac_dim

        interpolates = alpha * real_data + ((1 - alpha) * fake_data) # interpolated samples (for each sample in batch/pac)

        disc_output_interpolates = self(interpolates) # feeds the interpolated data into the discriminator to get the output
        print("uuuu", type(disc_output_interpolates[0]))
        # computes the gradients of the discriminator’s output with respect to the interpolated inputs
        gradients = torch.autograd.grad(
            outputs=disc_output_interpolates[0],
            inputs=interpolates,
            grad_outputs=torch.ones(disc_output_interpolates[0].size()), #  gradient of each element of the disc_output_interpolates tensor with respect to the interpolates tensor
            create_graph=True,
            retain_graph=True,
            only_inputs=True, # gradients are only computed with respect to the inputs (interpolates)
        )[0] # [0] extracts the gradient tensor

        gradients_norm = gradients.view(-1, self.pac * real_data.size(1)).norm(2, dim=1) - 1 # view reshapes to pac_dim, function computes the L2 (that`s why 2) norm (Euclidean norm) of the gradient vectors for each sample in the batch
        # -1 from the norm to compute how much the gradient deviates from 1, 1-Lipschitz condition
        gradient_penalty = ((gradients_norm) ** 2).mean() * lambda_ # squares the deviations and averages them over the batch

        return gradient_penalty

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0 # checks that the batch size is divisible by pac
        return self.seq(input_.view(-1, self.pacdim)), self.seq_info(input_.view(-1, self.pacdim)) # (250,13)-> (125,26), if pac=2, feature_dim=13
    

class Residual(Module): # to solve the problem of the vanishing/exploding gradient, skip connections

    def __init__(self, i, o): # i - input features, o -output features sizes
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o) # normalizes the outputs of the linear layer across the batch to stabilize training
        self.relu = ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1) # concatinate input and output to retain the original input features at every stage while learning new feature representations

class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        # hidden layers construction
        # each hidden layer consists of a residual block (fully connected transformation, Batch normalization for stable training, ReLU activation)
        seq = []
        for item in generator_dim:
            seq += [Residual(dim, item)]
            dim += item
        
        # output layer
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data


def apply_activate(data, col_types, col_names, col_dims): # transform G`s output to pass to D as input`
    data_t = [] # transformed data
    
    for key, value in col_types.items():
        st, ed = st_ed_col(key, col_names, col_dims)
        if value=="categorical":
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2)) # tau controls how "sharp" or "soft" the output probabilities are
            # tau=0.2  sharper, more confident probabilities, output values are closer to one-hot, meaning the distribution picks a single category with higher certainty
        else: # cont, mixed, general
            data_t.append(torch.tanh(data[:, st:ed]))
    return torch.cat(data_t, dim=1) # dim=1 feature dimension, merges tensors along the feature dimension
    

    # cross entropy loss - generator loss
def loss_generator(data, col_names, col_dims, categorical_labels, cat_col_dims, c, m): # c- cond vector, m - one-hot encoded mask for the chosen column, data - generated data
    loss = []

    for key in categorical_labels:
        st, ed = st_ed_col(key, col_names, col_dims) # in generated data
        st_c, ed_c = st_ed_col(key, list(categorical_labels.keys()), cat_col_dims) # in cond vector
        tmp = F.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction='none') # reduction='none' means the loss is computed per row, torch.argmax(c[:, st_c:ed_c]- getting index of ones
        loss.append(tmp)

    loss = torch.stack(loss, dim=1) # per-feature losses
    # loss * m include only features, that 1 in m (for each sample in batch), loss and m - same shape
    return (loss * m).sum() / data.size()[0] # average loss per sample across the batch (not depending on batch size)



class Synthesizer:
    def __init__(self,
                 noise_dim=128, # dim of z - noise vector for G 
                 generator_dim=[256, 256], # list of integers, size(s) of hidden layers
                 discriminator_dim=[256, 256], # list of integers, size(s) of hidden layers
                 pac=10, # number of samples in one pac
                 batch_size=50,
                 epochs=150):

        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.pac = pac

    def fit(self, train_data, transformation, data_sampler, cond_vector, classifier):

        # transform data
        # initizalize transformer and fit, get train_data
        # train_data = 0
        self.transformation = transformation

        # initialize Sampler, cond vector class
        self.data_sampler = data_sampler 
        self.cond_vector = cond_vector

        # initialize G
        self.generator = Generator(self.noise_dim + cond_vector.n_opt, self.generator_dim, train_data.shape[1])
        optimizer_params_G = dict(lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerG = optim.Adam(self.generator.parameters(), **optimizer_params_G)
        
        # initialize D
        data_dim = train_data.shape[1] # !!!
        discriminator = Discriminator(data_dim + cond_vector.n_opt, self.discriminator_dim, pac=self.pac)
        optimizer_params_D = dict(lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerD = optim.Adam(discriminator.parameters(),**optimizer_params_D)


        epoch = 0
        steps_d = 1 # number of updates D for 1 update G
        
        steps_per_epoch = max(1, len(train_data) // self.batch_size) # to get the number of full batches, but at least 1

        for i in tqdm(range(self.epochs)):
            for id_ in range(steps_per_epoch):
                # update steps_d times D
                for _ in range(steps_d):
                    noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
                    c, m, col, opt = self.cond_vector.sample_train(self.batch_size) # cond vector
                    c = torch.from_numpy(c)
                    m = torch.from_numpy(m)

                    noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector

                    perm = np.arange(self.batch_size) # get index for condition
                    np.random.shuffle(perm)
                    # sample from real data based on conditions for con vector
                    real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm]) # sample real data of the batch size with correct order of columns and categories
                    real = torch.from_numpy(real.astype('float32'))

                    c_perm = c[perm] # order of cond vectors in batch

                    fake = self.generator(noisez_c) # get generated data
                    fake_act = apply_activate(fake, self.transformation.col_types, self.transformation.transformed_col_names, self.transformation.transformed_col_dims) # transform for D input
                    
                    fake_c = torch.cat([fake_act, c], dim=1) # concatenate feature and condition
                    real_c = torch.cat([real, c_perm], dim=1) # concatenate feature and condition

                    y_fake, _ = discriminator(fake_c) # get yfake from D
                    y_real, _ = discriminator(real_c) # get yreal from D
                    
                    penalty_d = discriminator.calc_gradient_penalty(real_c, fake_c, lambda_=10)

                    # discriminator loss
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    penalty_d.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                
                # update 1 time G
                # regenerate data, as before
                noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
                c, m, col, opt = self.cond_vector.sample_train(self.batch_size) # cond vector
                c = torch.from_numpy(c)
                m = torch.from_numpy(m)

                noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector

                optimizerG.zero_grad()
                
                fake = self.generator(noisez_c) # get generated data
                fake_act = apply_activate(fake, self.transformation.col_types, self.transformation.transformed_col_names, self.transformation.transformed_col_dims) # transform for D input
                
                fake_c = torch.cat([fake_act, c], dim=1) # concatenate feature and condition
             
                y_fake = discriminator(fake_c) 


                y_fake,info_fake = discriminator(fake_c)
                # we do not need real samples to update G

                # to mix up the condition vector and break the direct correspondence between the real data and the condition vector that was sampled for the fake data
                # real data is no longer paired with the same condition vector that the generator used to generate fake data
                _,info_real = discriminator(real_c)

                ######### loss calculation

                # generator loss
                cross_entropy = loss_generator(fake, self.transformation.transformed_col_names, self.transformation.transformed_col_dims, self.transformation.categorical_labels, self.cond_vector.cat_col_dims, c, m)
                
                # original and # generator loss
                g_orig_gen = -torch.mean(y_fake) + cross_entropy

                g_orig_gen.backward(retain_graph=True)

                # information loss
                info_fake_data = info_fake[:, :-cond_vector.n_opt]  # exclude dimension of conditional vector
                info_real_data = info_real[:, :-cond_vector.n_opt]  # same for real data

                # losses without the conditional vector, L2 norm (euqlidian distance)
                #error
                loss_mean = torch.norm(torch.mean(info_fake_data.view(self.batch_size, -1), dim=0) - torch.mean(info_real_data.view(self.batch_size, -1), dim=0), 2)
                loss_std = torch.norm(torch.std(info_fake_data.view(self.batch_size, -1), dim=0) - torch.std(info_real_data.view(self.batch_size, -1), dim=0), 2)

                g_loss_info = loss_mean + loss_std 
                g_loss_info.backward()
                optimizerG.step()

                # classification loss G
                fake_pre, fake_label = classifier(fake_act)
                g_loss_class = classifier.loss_classification(fake_pre, fake_label)

                optimizerG.zero_grad()
                g_loss_class.backward()
                optimizerG.step()
                                
            epoch += 1
            
   