import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm
import time
from sklearn import metrics
import pandas as pd

from blocks.data_prep import DataPrep
from blocks.networks import Classifier, Discriminator, Generator, apply_activation, loss_generator, loss_constraint
from blocks.sampler import Sampler
from blocks.cond_constr import Cond_vector
from blocks.evaluation import Model_evaluation

class Synthesizer:
    def __init__(self,
                 noise_dim=128, # dim of z - noise vector for G 
                 generator_dim=[512,512], # list of integers, size(s) of hidden layers
                 discriminator_dim=[512,256], # list of integers, size(s) of hidden layers
                 classifier_dim = [256,256], # list of integers of hidden layers
                 constr_loss_coef = 10, # constraint loss coefficient
                 mode_threshold = 0.005,
                 pac=10, # number of samples in one pac
                 batch_size=300,
                 epochs=300,
                 steps_d = 5, # number of updates D for 1 update G
                 epochs_c=100,
                 steps_per_epoch = 12,
                 lr_g=1e-4,
                 weight_decay_g = 1e-5,
                 lr_d=1e-4,
                 weight_decay_d = 1e-6):
        
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.classifier_dim = classifier_dim
        self.pac = pac
        self.steps_d = steps_d
        self.constr_loss_coef = constr_loss_coef
        self.mode_threshold = mode_threshold
        self.epochs_c = epochs_c
        self.lr_g = lr_g
        self.weight_decay_g = weight_decay_g
        self.lr_d = lr_d
        self.weight_decay_d = weight_decay_d

    def fit(self, data_name, raw_data, categorical_cols, continuous_cols, mixed_cols, general_cols, log_transf, components_numbers, mixed_modes, target_col, case_name, class_balance=None, condition_list=None, cond_ratio = None):
        # cases: actual, constr, constr_cv, constr_loss, cond, constr_cond
        
        print("Transformation")
        # transform data        
        self.transformer = DataPrep(raw_df=raw_data,
                    categorical_cols=categorical_cols,
                    continuous_cols=continuous_cols,
                    mixed_cols=mixed_cols,
                    general_cols=general_cols,
                    log_transf=log_transf,
                    components_numbers=components_numbers,
                    mode_threshold=self.mode_threshold,
                    mixed_modes=mixed_modes)
        
        train_data = self.transformer.transform(raw_data)
        
        # self.steps_per_epoch = max(1, len(train_data) // self.batch_size) # to get the number of full batches, but at least 1
        
        data_dim = train_data.shape[1]

        # initialize Sampler class
        self.data_sampler = Sampler(train_data, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels, case_name)
        
        # initialize cond vector class
        self.cond_vector = Cond_vector(train_data,self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels, case_name, cond_ratio, class_balance, condition_list)

        # initialize Evaluation class
        self.evaluation_model = Model_evaluation(self.epochs, self.steps_per_epoch, self.steps_d, case_name, data_name)

        train_data = torch.from_numpy(train_data).float()

        # initialize C
        classifier = Classifier(data_dim,self.classifier_dim, target_col, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.col_types)
        optimizer_params_C = dict(lr=2e-4, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-5)

        optimizerC = optim.Adam(classifier.parameters(),**optimizer_params_C)

        print("train C")
        # train C
        
        for i in tqdm(range(self.epochs_c)):
            for id in range(self.steps_per_epoch):
                real_pre, real_label = classifier(train_data)

                loss_cc = classifier.loss_classification(real_pre, real_label)

                optimizerC.zero_grad()
                loss_cc.backward()
                optimizerC.step()

        # initialize G
        self.generator = Generator(self.noise_dim + self.cond_vector.n_opt, self.generator_dim, train_data.shape[1])
        optimizer_params_G = dict(lr=self.lr_g, betas=(0.5, 0.9), weight_decay=self.weight_decay_g)
        optimizerG = optim.Adam(self.generator.parameters(), **optimizer_params_G)

        # initialize D

        discriminator = Discriminator(data_dim + self.cond_vector.n_opt, self.discriminator_dim, pac=self.pac)
        optimizer_params_D = dict(lr=self.lr_d, betas=(0.5, 0.9), weight_decay=self.weight_decay_d)
        optimizerD = optim.Adam(discriminator.parameters(),**optimizer_params_D)

        print("train G and D")

        # train G and D

        epoch = 0

        # for evaluation later
        epoch_train_time = []
        g_loss_orig_lst = []
        g_loss_gen_lst = []
        g_loss_info_lst = []
        g_loss_class_lst = []
        g_loss_constr_lst = []
        d_loss_lst = []
        g_loss_lst = []
        d_auc_score = []

        gen_probs = None
        
        torch.autograd.set_detect_anomaly(True)
        
        for i in tqdm(range(self.epochs)):
            time_start = time.perf_counter()
            for id_ in range(self.steps_per_epoch):
                # update steps_d times D
                for _ in range(self.steps_d):
                    noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
                    c, m, col, opt = self.cond_vector.sample_train(self.batch_size, gen_probs) # cond vector 
                    c = torch.from_numpy(c)
                    m = torch.from_numpy(m)

                    noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector

                    perm = np.arange(self.batch_size) # get index for condition
                    np.random.shuffle(perm)
                    # sample from real data based on conditions for cond vector
                    real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm]) # sample real data of the batch size with correct order of columns and categories
                    real = torch.from_numpy(real.astype('float32'))

                    c_perm = c[perm] # order of cond vectors in batch

                    fake = self.generator(noisez_c) # get generated data
                    fake_act = apply_activation(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform for D input
                    
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

                    d_loss_lst.append(float(loss_d))

                    y_labels_true = np.concatenate([np.ones_like(y_real.detach().numpy()), np.zeros_like(y_fake.detach().numpy())])
                    d_pred = np.concatenate([y_real.detach().numpy(), y_fake.detach().numpy()])
                    d_auc_score.append(metrics.roc_auc_score(y_labels_true, d_pred))
                
                # update 1 time G
                # regenerate data, as before
                noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
                c, m, col, opt = self.cond_vector.sample_train(self.batch_size, gen_probs) # cond vector
                c = torch.from_numpy(c)
                m = torch.from_numpy(m)

                noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector

                optimizerG.zero_grad()
                
                fake = self.generator(noisez_c) # get generated data
                fake_act = apply_activation(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform for D input
                
                fake_c = torch.cat([fake_act, c], dim=1) # concatenate feature and cond vector

                y_fake = discriminator(fake_c)
                

                ######### loss calculation

                # original loss
                g_loss_orig = -torch.mean(y_fake)

                # generator loss
                g_loss_gen = loss_generator(fake, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, self.transformer.categorical_labels, self.cond_vector.cat_col_dims, c, m)

                # g_loss_orig_gen = g_loss_orig + g_loss_gen
                # optimizerG.zero_grad(set_to_none=False)
                # g_loss_orig_gen.backward(retain_graph=True)
                # optimizerG.step()

                # information loss

                # to have paired real and fake data
                perm = np.arange(self.batch_size) # get index for condition
                np.random.shuffle(perm)
                # sample from real data based on conditions for cond vector
                real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm]) # sample real data of the batch size with correct order of columns and categories
                real = torch.from_numpy(real.astype('float32'))

                # losses without the conditional vector, L2 norm (euqlidian distance)
                loss_mean = torch.norm(torch.mean(fake_act, dim=0) - torch.mean(real, dim=0), 2)
                loss_std = torch.norm(torch.std(fake_act, dim=0) - torch.std(real, dim=0), 2)
                
                g_loss_info = loss_mean + loss_std 
                # g_loss_info.backward(retain_graph=True)

                ##### train C additionaly
                real_pre, real_label = classifier(real)

                loss_cc = classifier.loss_classification(real_pre, real_label)

                optimizerC.zero_grad()
                loss_cc.backward()
                optimizerC.step()
                #####

                # classification loss G
                fake_pre, fake_label = classifier(fake_act)
                g_loss_class = classifier.loss_classification(fake_pre, fake_label)
                
                # g_loss_class.backward(retain_graph=True)
                # optimizerG.step()

                g_loss_orig_lst.append(float(g_loss_orig))
                g_loss_gen_lst.append(float(g_loss_gen))
                g_loss_info_lst.append(float(g_loss_info))
                g_loss_class_lst.append(float(g_loss_class))

                if case_name == "actual" or case_name == "cond":
                    total_loss = g_loss_orig + g_loss_gen + g_loss_info + g_loss_class
                    
                    optimizerG.zero_grad(set_to_none=False)
                    total_loss.backward(retain_graph=True)
                    optimizerG.step()

                    g_loss_lst.append(float(g_loss_orig)+float(g_loss_gen)+float(g_loss_info)+float(g_loss_class))

                else:
                    if case_name == "constr" or case_name == "constr_cond":
                        g_loss_constr, _ = loss_constraint(fake_act, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, class_balance, self.constr_loss_coef)
                        total_loss = g_loss_orig + g_loss_gen + g_loss_info + g_loss_class + g_loss_constr

                    elif case_name == "constr_cv":
                        g_loss_constr, _ = loss_constraint(fake_act, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, class_balance, self.constr_loss_coef)
                        total_loss = g_loss_orig + g_loss_gen + g_loss_info + g_loss_class

                    elif case_name == "constr_loss":
                        g_loss_constr, gen_probs = loss_constraint(fake_act, self.transformer.transformed_col_names, self.transformer.transformed_col_dims, class_balance, self.constr_loss_coef)
                        total_loss = g_loss_orig + g_loss_gen + g_loss_info + g_loss_class + g_loss_constr

                    optimizerG.zero_grad(set_to_none=False)
                    total_loss.backward(retain_graph=True)
                    optimizerG.step()

                    g_loss_constr_lst.append(float(g_loss_constr))
                    g_loss_lst.append(float(g_loss_orig)+float(g_loss_gen)+float(g_loss_info)+float(g_loss_class)+float(g_loss_constr))

            time_end = time.perf_counter()
            epoch_train_time.append(time_end-time_start)
            epoch += 1
        
        self.gen_probs = gen_probs
        
        ####### Evaluation
        if case_name == "actual" or case_name == "cond":
            self.evaluation_model.losses_plot(d_loss_lst, g_loss_lst, g_loss_orig_lst, g_loss_gen_lst, g_loss_info_lst, g_loss_class_lst) 
        else:
            self.evaluation_model.losses_plot(d_loss_lst, g_loss_lst, g_loss_orig_lst, g_loss_gen_lst, g_loss_info_lst, g_loss_class_lst, g_loss_constr_lst)       
        
        self.avg_train_epoch_time = self.evaluation_model.calc_metrics(epoch_train_time, d_auc_score)


    def sample(self, n_rows): # sample data after training is finished. User defines how many rows
        print("Sampling")
        self.generator.eval()

        steps = n_rows // self.batch_size + 1
        
        data = []
        
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
            c = self.cond_vector.sample_case(self.batch_size, self.gen_probs) # cond vector
            c = torch.from_numpy(c)

            noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector            
            fake = self.generator(noisez_c) # get generated data
            fake_act = apply_activation(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform
            
            data.append(fake_act.detach().numpy())

        data = np.concatenate(data, axis=0)

        result_data = self.transformer.inverse_transform(data)

        while len(result_data) < n_rows: # while no invalid samples
            data = []
            
            for i in range(steps):
                noisez = torch.randn(self.batch_size, self.noise_dim) # generate noise for G
                c = self.cond_vector.sample_case(self.batch_size, self.gen_probs) # cond vector
                c = torch.from_numpy(c)

                noisez_c = torch.cat([noisez, c], dim=1) # concatinate noise with cond vector            
                fake = self.generator(noisez_c) # get generated data
                fake_act = apply_activation(fake, self.transformer.col_types, self.transformer.transformed_col_names, self.transformer.transformed_col_dims) # transform
                
                data.append(fake_act.detach().numpy())

            data = np.concatenate(data, axis=0)
            tmp_result_data = self.transformer.inverse_transform(data)
            result_data  = pd.concat([result_data, tmp_result_data], ignore_index=True)
        result_data = result_data[0:n_rows]
        result_data = result_data.reset_index(drop=True)
        return result_data

            
   