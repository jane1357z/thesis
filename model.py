import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm


from blocks.data_prep import DataPrep
from blocks.networks import Classifier, Discriminator, Generator, apply_activate, loss_generator
from blocks.sampler import Sampler
from blocks.cond_constr import Cond_vector, Constraints

class Synthesizer:
    def __init__(self,
                 noise_dim=128, # dim of z - noise vector for G 
                 generator_dim=[256, 256], # list of integers, size(s) of hidden layers
                 discriminator_dim=[256, 256], # list of integers, size(s) of hidden layers
                 classifier_dim = [256, 256], # (64), list of integers of hidden layers
                 pac=10, # number of samples in one pac
                 batch_size=50,
                 epochs=50):

        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.classifier_dim = classifier_dim
        self.pac = pac

    def fit(self, row_data, categorical_cols, continuous_cols, mixed_cols, general_cols, clusters_numbers, mode_threshold, mixed_modes, target_col, class_balance):

        # transform data        
        self.transformer = DataPrep(row_df=row_data,
                    categorical_cols=categorical_cols,
                    general_cols=general_cols,
                    continuous_cols=continuous_cols,
                    mixed_cols=mixed_cols, 
                    clusters_numbers=clusters_numbers,
                    mode_threshold=mode_threshold,
                    mixed_modes=mixed_modes)
        
        train_data = self.transformer.transform(row_data)

        data_dim = train_data.shape[1]

        # initialize Sampler class
        self.data_sampler = Sampler(train_data, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels)
        
        # initialize cond vector class
        self.cond_vector = Cond_vector(train_data,self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels)

        # initialize Constraints class
        self.constraints = Constraints(self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels)

        # initialize C
        train_data = torch.from_numpy(train_data).float()
        classifier = Classifier(data_dim,self.classifier_dim, target_col, self.transformer.transformed_col_names, self.transformer.transformed_col_dims)
        optimizer_params_C = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=1e-5)

        optimizerC = optim.Adam(classifier.parameters(),**optimizer_params_C)


        # train C
        steps_per_epoch=50
        epochs=100
        loss = []
        for i in tqdm(range(epochs)):
            for id in range(steps_per_epoch):
                real_pre, real_label = classifier(train_data)

                loss_cc = classifier.loss_classification(real_pre, real_label)
                loss.append(float(loss_cc))

                optimizerC.zero_grad()
                loss_cc.backward()
                optimizerC.step()


        # initialize G
        self.generator = Generator(self.noise_dim + self.cond_vector.n_opt, self.generator_dim, train_data.shape[1])
        optimizer_params_G = dict(lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerG = optim.Adam(self.generator.parameters(), **optimizer_params_G)
        
        # initialize D

        discriminator = Discriminator(data_dim + self.cond_vector.n_opt, self.discriminator_dim, pac=self.pac)
        optimizer_params_D = dict(lr=2e-4, betas=(0.5, 0.9), weight_decay=1e-6)
        optimizerD = optim.Adam(discriminator.parameters(),**optimizer_params_D)


        epoch = 0
        steps_d = 1 # number of updates D for 1 update G
        
        steps_per_epoch = max(1, len(train_data) // self.batch_size) # to get the number of full batches, but at least 1
        torch.autograd.set_detect_anomaly(True)
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
                    fake_act = apply_activate(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform for D input
                    
                    fake_c = torch.cat([fake_act, c], dim=1) # concatenate feature and condition
                    real_c = torch.cat([real, c_perm], dim=1) # concatenate feature and condition

                    y_fake = discriminator(fake_c) # get yfake from D
                    y_real = discriminator(real_c) # get yreal from D
                    
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
                fake_act = apply_activate(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform for D input
                
                fake_c = torch.cat([fake_act, c], dim=1) # concatenate feature and cond vector

                y_fake = discriminator(fake_c) # y_fake,info_fake = discriminator(fake_c)
                # we do not need real samples to update G

                # to mix up the condition vector and break the direct correspondence between the real data and the condition vector that was sampled for the fake data
                # real data is no longer paired with the same condition vector that the generator used to generate fake data
                # _,info_real = discriminator(real_c)



                ######### loss calculation

                # generator loss
                cross_entropy = loss_generator(fake, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels, self.cond_vector.cat_col_dims, c, m)
                
                # original and # generator loss
                g_orig_gen = -torch.mean(y_fake) + cross_entropy

                g_orig_gen.backward(retain_graph=True)

                # information loss

                # losses without the conditional vector, L2 norm (euqlidian distance)
                loss_mean = torch.norm(torch.mean(fake_act, dim=0) - torch.mean(real, dim=0), 2)
                loss_std = torch.norm(torch.std(fake_act, dim=0) - torch.std(real, dim=0), 2)
                
                g_loss_info = loss_mean + loss_std 
                g_loss_info.backward(retain_graph=True)

                # classification loss G
                fake_pre, fake_label = classifier(fake_act)
                g_loss_class = classifier.loss_classification(fake_pre, fake_label)
                
                g_loss_class.backward(retain_graph=True)
                optimizerG.step()

                # g_loss_class + penalty_constraint
                penalty_constraint = self.constraints.calc_constraint_penalty(fake_act, self.batch_size, class_balance)

                # total_loss = g_orig_gen + g_loss_info + g_loss_class + penalty_constraint

                optimizerG.zero_grad()
                penalty_constraint.backward()
                optimizerG.step()
                                
            epoch += 1
    
    def sample(self, n_rows):
        
        self.generator.eval()

        steps = n_rows // self.batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
            c, _, _, _ = self.cond_vector.sample_train(self.batch_size) # cond vector
            c = torch.from_numpy(c)

            noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector            
            fake = self.generator(noisez_c) # get generated data
            fake_act = apply_activate(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform
            
            data.append(fake_act.detach().numpy())

        data = np.concatenate(data, axis=0)
        # result,resample = self.transformer.inverse_transform(data) #resample = len(invalid_ids)
        
        # while len(result) < n_rows:
        #     data_resample = []    
        #     steps_left = resample// self.batch_size + 1
            
        #     for i in range(steps_left):
        #         noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
        #         c, _, _, _ = self.cond_vector.sample_train(self.batch_size) # cond vector
        #         c = torch.from_numpy(c)

        #         noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector            
        #         fake = self.generator(noisez_c) # get generated data
        #         fake_act = apply_activate(fake, self.transformation.col_types, self.transformation.transformed_col_names, self.transformation.transformed_col_dims) # transform for D input
                

        #         data_resample.append(fake_act.detach().cpu().numpy())

        #     data_resample = np.concatenate(data_resample, axis=0)

        #     res,resample = self.transformer.inverse_transform(data_resample)
        #     result  = np.concatenate([result,res],axis=0)
        result_data = self.transformer.inverse_transform(data[0:n_rows])
        return result_data #result[0:n_rows]

            
   