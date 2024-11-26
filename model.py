import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


from blocks.data_prep import DataPrep
from blocks.networks import Classifier, Discriminator, Generator, apply_activate, loss_generator
from blocks.sampler import Sampler
from blocks.cond_constr import Cond_vector

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
            
   