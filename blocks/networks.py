import torch
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Sigmoid, BCELoss, CrossEntropyLoss,SmoothL1Loss, BatchNorm1d)


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