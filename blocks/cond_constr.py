import numpy as np
import torch

def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

def random_choice_prob_index_sampling(probs,col_idx,col_opt_single): # sampling from actual probs, not log - selects option for each sample in batch
    option_list = []
    if col_opt_single == True:
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    else:
        for i in range(len(col_idx)):
            pp0 = probs[col_idx[i][0]] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
            opt0 = np.random.choice(np.arange(len(probs[col_idx[i][0]])), p=pp0) # get option based on actual prob
            opt1 = None # no need another conditioned column
            option_list.append([opt0, opt1])
    
    return np.array(option_list).reshape(col_idx.shape)

class Cond_vector(object):
    def __init__(self, data, col_types, col_names, col_dims, categorical_labels, case_name, cond_ratio, class_balance, condition_list):
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.categorical_labels = categorical_labels
        self.cat_col_dims = []
        self.class_balance = class_balance
        if class_balance:
            self.constr_col_ind = list(self.categorical_labels.keys()).index(list(self.class_balance.keys())[0])
        if condition_list:
            self.cond_col1_ind = list(self.categorical_labels.keys()).index(condition_list[0]["col1"]) 
            self.cond_col2_ind = list(self.categorical_labels.keys()).index(condition_list[0]["col2"]) 
        self.condition_list = condition_list
        self.case_name = case_name
        self.cond_ratio = cond_ratio # threshold based on probability (how often we want to generate this condition)

        for key in self.categorical_labels.keys(): #categorical_labels
            st, ed = st_ed_col(key, self.col_names, self.col_dims)
            col_idx = self.col_names.index(key)
            if not self.cat_col_dims:
                counter = 0
            else:
                counter = self.cat_col_dims[len(self.cat_col_dims)-1][0]+self.cat_col_dims[len(self.cat_col_dims)-1][1]
            self.cat_col_dims.append([counter, self.col_dims[col_idx][1]])
               
        self.n_col = len(self.categorical_labels)   # total number of categorical features in the data
        self.n_opt = sum(map(len, self.categorical_labels.values()))  # total number of options in feature
        self.p_sampling = [] # stores the raw probabilities for sampling from the categories of a specific feature
        counter = 0
        for key, value in self.categorical_labels.items():
            st, ed = st_ed_col(key, self.col_names, self.col_dims)
        
            tmp_prob = np.sum(self.data[:, st:ed], axis=0) # sum of 1s of each category in the column
            tmp_prob = tmp_prob / np.sum(tmp_prob) # probability of each category in the column
            self.p_sampling.append(tmp_prob)

            counter += 1
        
    def category_choice_actual_prob(self, probs, col_idx): # sampling from actual probs
        option_list = []
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
        
        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_generator(self, probs, col_idx): # sampling from generator probs
        option_list = []
        for i in col_idx:
            if i == self.constr_col_ind:
                if probs is not None:
                    pp = probs
                    option_list.append(np.random.choice(np.arange(len(probs)), p=pp))            
                else:    
                    pp = self.p_sampling[i]
                    option_list.append(np.random.choice(np.arange(len(self.p_sampling[i])), p=pp))
            else:
                pp = self.p_sampling[i]
                option_list.append(np.random.choice(np.arange(len(self.p_sampling[i])), p=pp))
        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_constraint(self, probs, col_idx):
        option_list = []
        class_balance_col = list(self.class_balance.keys())[0]
        for i in col_idx:
            if list(self.categorical_labels.keys())[i] == class_balance_col: # check if the columns is in constraint
                pp = self.class_balance[class_balance_col] # sampling basen on user`s constraint probability (class balance) for this columns
                option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
            else: # if column is not in constraint, then sample from actual probabilities (probs)
                pp = probs[i]
                option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_condition(self, probs, col_idx):
        option_list = []
        for i in range(len(col_idx)):
            # check if the columns are conditioned
            col_name_1 = list(self.categorical_labels.keys())[int(col_idx[i][0])]
            col_name_2 = list(self.categorical_labels.keys())[int(col_idx[i][1])]
            # if (col_name_1 in self.condition_list[0].values() and col_name_2 in self.condition_list[0].values()) == True:
            epsilon = np.random.uniform(0,1)
            if epsilon <= self.cond_ratio:
                if col_name_1 == self.condition_list[0]["col1"]: # get the row under condition
                    opt0 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
                    opt1 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                else:
                    opt0 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                    opt1 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
            else:
                pp0 = probs[int(col_idx[i][0])] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
                opt0 = np.random.choice(np.arange(len(probs[int(col_idx[i][0])])), p=pp0) # get option based on actual prob
                opt1 = None # no need another conditioned column
            # else:
            #     pp0 = probs[col_idx[i][0]] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
            #     opt0 = np.random.choice(np.arange(len(probs[col_idx[i][0]])), p=pp0) # get option based on actual prob
            #     opt1 = None
            option_list.append([opt0, opt1])
        
        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_constr_cond(self, probs, col_idx):
        option_list = []
        for i in range(len(col_idx)):
            if None in col_idx[i]:
                class_balance_col = list(self.class_balance.keys())[0]
                # if list(self.categorical_labels.keys())[col_idx[i][0]] == class_balance_col: # check if the columns is in constraint
                #     pp = self.class_balance[class_balance_col] # sampling basen on user`s constraint probability (class balance) for this columns
                #     opt0 = np.random.choice(np.arange(len(pp)), p=pp)
                #     opt1 = None
                if probs is not None:
                    pp = probs
                    opt0 = np.random.choice(np.arange(len(pp)), p=pp)
                    opt1 = None
                else:
                    pp = self.p_sampling[int(col_idx[i][0])]
                    opt0 = np.random.choice(np.arange(len(pp)), p=pp)
                    opt1 = None
            else:
                col_name_1 = list(self.categorical_labels.keys())[int(col_idx[i][0])]
                col_name_2 = list(self.categorical_labels.keys())[int(col_idx[i][1])]
                epsilon = np.random.uniform(0,1)
                if epsilon <= self.cond_ratio:
                    if col_name_1 == self.condition_list[0]["col1"]: # get the row under condition
                        opt0 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
                        opt1 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                    else:
                        opt0 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                        opt1 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
                else:
                    pp0 = self.p_sampling[int(col_idx[i][0])] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
                    opt0 = np.random.choice(np.arange(len(pp0)), p=pp0) # get option based on actual prob
                    opt1 = None # no need another conditioned column

            option_list.append([opt0, opt1])
        
        return np.array(option_list).reshape(col_idx.shape)
    
    def sample_train(self, batch, gen_probs=None): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None

        # cases: actual, constr, cond, constr_cond
        if self.case_name == "actual":
            # col and opt - 1
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical columns!
            
            ### case actual original probs (nothing with constraints or conditions)
            opt1prime = self.category_choice_actual_prob(self.p_sampling, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector

        elif  self.case_name == "constr_loss":
            # col and opt - 1
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch

            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical columns!
            
            ### probs from generator
            opt1prime = self.category_choice_generator(gen_probs, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector

        elif self.case_name == "constr" or self.case_name == "constr_cv":
            # col and opt - 1
            probs = np.full(self.n_col, 0.0 / (self.n_col - 1)) 
            probs[self.constr_col_ind] = 1.0
            col_idx = np.random.choice(np.arange(self.n_col), size=batch, p=probs)
            
            # mask is used in loss_generator
            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!
            
            ### case constr get column from constraint and sample from class_balance
            opt1prime = self.category_choice_constraint(self.p_sampling, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector

        elif self.case_name == "cond" or self.case_name == "constr_cond":
            # col and opt - 2
            col_idx = np.tile([self.cond_col1_ind, self.cond_col2_ind], (batch, 1))
            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

            ### case cond get columns from condtions and sample from condtions col and opt - 2
            opt1prime = self.category_choice_condition(self.p_sampling, col_idx) # category choice for current column - batch
            
            for i in np.arange(batch): 
                if opt1prime[i][1] == None:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    
                    mask[i, col_idx[i][0]] = 1 # set the first column
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1
                    
                    mask[i, col_idx[i][0]] = 1 # set the first column
                    mask[i, col_idx[i][1]] = 1  # set the second column

        return vec, mask, col_idx, opt1prime

    def sample(self, batch): # conditional vectors for testing / evaluation - using actual probabilities
        if self.n_col == 0:
            return None

        # col and opt - 2 
        if self.case_name == "cond" or self.case_name == "constr_cond":
            col_idx = np.array([np.random.choice(np.arange(self.n_col), size=2, replace=False) for _ in range(batch)])
            vec = np.zeros((batch, self.n_opt), dtype='float32')

            opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx, col_opt_single=False) # choice of the option in the current column based on actual probs, (not log)

            for i in np.arange(batch):
                if opt1prime[i][1] == None: # col and opt - 1
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1
                
        else:
            col_idx = np.random.choice(np.arange(self.n_col), batch) # choice of the column
            vec = np.zeros((batch, self.n_opt), dtype='float32')
            opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx, col_opt_single=True) # choice of the option in the current column based on actual probs, (not log)
            
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1
        return vec
    
    def sample_case(self, batch, gen_probs=None): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None

        # cases: actual, constr, cond, constr_cond
        if self.case_name == "actual":
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical columns!
            opt1prime = self.category_choice_actual_prob(self.p_sampling, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector
        
        elif self.case_name == "constr_loss": # cond vector does not guide, everything is uniform
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical columns!
            opt1prime = self.category_choice_generator(gen_probs, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector            
        
        elif self.case_name == "constr" or self.case_name == "constr_cv":
            probs = np.full(self.n_col, 0.0) 
            probs[self.constr_col_ind] = 1.0 # choose constr col
            col_idx = np.random.choice(np.arange(self.n_col), size=batch, p=probs)
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!
            
            opt1prime = self.category_choice_constraint(self.p_sampling, col_idx) # category choice for current column - batch
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1 # form conditional vector

        elif self.case_name == "cond" or self.case_name == "constr_cond":
            col_idx = np.tile([self.cond_col1_ind, self.cond_col2_ind], (batch, 1))
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!
            opt1prime = self.category_choice_condition(self.p_sampling, col_idx) # category choice for current column - batch
            
            for i in np.arange(batch): 
                if opt1prime[i][1] == None:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1
        
        return vec