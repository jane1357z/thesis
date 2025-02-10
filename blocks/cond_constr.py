import numpy as np
import torch

def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

# selects option for each sample in batch, cumulative sum maybe to change 
# def random_choice_prob_index(a, axis=1): # takes indices based on cumulative probability distributions in array of normalized log-probabilities - sampling from normalized log-probs 
#     r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis) # random value between 0 and 1 for each category in feature in batch
#     # returns the index of the first threshold where the cumulative sum exceeds the random number (r) - index of category for each sample in batch
#     return (a.cumsum(axis=axis) > r).argmax(axis=axis) # compares cumulative sum of probabilities with the random numbers

def random_choice_prob_index_sampling(probs,col_idx,col_opt_single): # sampling from actual probs, not log - selects option for each sample in batch
    option_list = []
    if col_opt_single == True:
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    else:
        for i in range(len(col_idx)):
            pp0 = probs[col_idx[i][0]] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
            # pp1 = probs[col_idx[i][1]] # get actual probs (no condition applied), col_idx[i][1] - to get second col of condition
            opt0 = np.random.choice(np.arange(len(probs[col_idx[i][0]])), p=pp0) # get option based on actual prob
            # opt1 = np.random.choice(np.arange(len(probs[col_idx[i][1]])), p=pp1) # get option based on actual prob
            opt1 = None
            option_list.append([opt0, opt1])
    
    return np.array(option_list).reshape(col_idx.shape)

class Cond_vector(object):
    def __init__(self, data, col_types, col_names, col_dims, categorical_labels, class_balance, condition_list):
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.categorical_labels = categorical_labels
        self.cat_col_dims = []
        self.class_balance = class_balance
        self.condition_list = condition_list

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
        
            tmp_prob = np.sum(self.data[:, st:ed], axis=0) # sum of 1s of each category in the column
            tmp_prob = tmp_prob / np.sum(tmp_prob) # probability of each category in the column
            self.p_sampling.append(tmp_prob)

            #### not needed
            tmp_occur = np.sum(self.data[:, st:ed], axis=0)  # occurrences of each category
            tmp_occur = np.log(tmp_occur + 1) # (log(x + 1)) to smooth out the counts and normalize them into probabilities, creates a less extreme distribution when sampling.
            tmp_log_prob = tmp_occur / np.sum(tmp_occur) # Normalized into probabilities log-transform occurrences  of each category in the column
            self.p_log[counter, :len(value)] = tmp_log_prob
            ####

            counter += 1
        
    def category_choice_actual_prob(self, probs, col_idx):
        option_list = []
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
        
        return np.array(option_list).reshape(col_idx.shape)

    
    def category_choice_constraint(self, probs, col_idx):
        option_list = []
        class_balance_col = list(self.class_balance.keys())[0]
        for i in col_idx:
            if list(self.categorical_labels.keys())[i] == class_balance_col:
                pp = self.class_balance[class_balance_col]
                option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
            else:
                pp = probs[i]
                option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_condition(self, probs, col_idx):
        option_list = []
        cond_ratio = 0.4 # threshold based on probability (how often we want to create this condition)
        for i in range(len(col_idx)):
            # check if the columns are conditioned
            col_name_1 = list(self.categorical_labels.keys())[col_idx[i][0]]
            col_name_2 = list(self.categorical_labels.keys())[col_idx[i][1]]
            if (col_name_1 in self.condition_list[0].values() and col_name_2 in self.condition_list[0].values()) == True:
                epsilon = np.random.uniform(0,1)
                if epsilon <= cond_ratio:
                    if col_name_1 == self.condition_list[0]["col1"]:
                        opt0 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
                        opt1 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                    else:
                        opt0 = np.where(self.categorical_labels[self.condition_list[0]["col2"]]==self.condition_list[0]["cat2"])[0][0]
                        opt1 = np.where(self.categorical_labels[self.condition_list[0]["col1"]]==self.condition_list[0]["cat1"])[0][0]
                else:
                    pp0 = probs[col_idx[i][0]] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
                    # pp1 = probs[col_idx[i][1]] # get actual probs (no condition applied), col_idx[i][1] - to get second col of condition
                    opt0 = np.random.choice(np.arange(len(probs[col_idx[i][0]])), p=pp0) # get option based on actual prob
                    # opt1 = np.random.choice(np.arange(len(probs[col_idx[i][1]])), p=pp1) # get option based on actual prob
                    opt1 = None
            else:
                pp0 = probs[col_idx[i][0]] # get actual probs (no condition applied), col_idx[i][0] - to get first col of condition
                # pp1 = probs[col_idx[i][1]] # get actual probs (no condition applied), col_idx[i][1] - to get second col of condition
                opt0 = np.random.choice(np.arange(len(probs[col_idx[i][0]])), p=pp0) # get option based on actual prob
                # opt1 = np.random.choice(np.arange(len(probs[col_idx[i][1]])), p=pp1) # get option based on actual prob
                opt1 = None
            option_list.append([opt0, opt1])
        
        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_constr_cond(self, probs, col_idx): # !!!
        option_list = []
        class_balance_col = list(self.class_balance.keys())[0] 
        for i in col_idx:
            if list(self.categorical_labels.keys())[i] == class_balance_col:
                pp = self.class_balance[class_balance_col] # takes probs from constraint
                option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
                # compose condition
            else:
                # compose condition
                pp = probs[i]
                option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

        return np.array(option_list).reshape(col_idx.shape)
    
    def sample_train(self, batch): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None
        
        # here only categorical columns

        # cases: actual, constr, cond, constr_cond
        case_name = "cond"
        if case_name == "actual":
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
            # mask is not used anywhere?
            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!
            
            ### case actual original probs (nothing with constraints or conditions)
            opt1prime = self.category_choice_actual_prob(self.p_sampling, col_idx)
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1

        elif case_name == "constr":
            col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
            # mask is used loss_generator
            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection
            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!
            
            ### case constr get column from constraint and sample from class_balance
            opt1prime = self.category_choice_constraint(self.p_sampling, col_idx)
            for i in np.arange(batch):
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1

        elif case_name == "cond":
            col_idx = np.array([np.random.choice(np.arange(self.n_col), size=2, replace=False) for _ in range(batch)])

            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx[:, 0]] = 1 # set the first column
            mask[np.arange(batch), col_idx[:, 1]] = 1  # set the second column

            # mask = np.zeros((batch, 2, 2), dtype='float32') # categorical column was selected for each sample
            # mask[np.arange(batch), col_idx[:, 0], 0] = 1  # set the first column
            # mask[np.arange(batch), col_idx[:, 1], 1] = 1  # set the second column

            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

            ### case cond get columns from condtions and sample from condtions col and opt - 2
            opt1prime = self.category_choice_condition(self.p_sampling, col_idx)
            
            for i in np.arange(batch): 
                # col and opt - 2
                if opt1prime[i][1] == None:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1

        elif case_name == "constr_cond":
            col_idx = np.array([np.random.choice(np.arange(self.n_col), size=2, replace=False) for _ in range(batch)])

            mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
            mask[np.arange(batch), col_idx[:, 0]] = 1 # set the first column
            mask[np.arange(batch), col_idx[:, 1]] = 1  # set the second column

            # mask = np.zeros((batch, 2, 2), dtype='float32') # categorical column was selected for each sample
            # mask[np.arange(batch), col_idx[:, 0], 0] = 1  # set the first column
            # mask[np.arange(batch), col_idx[:, 1], 1] = 1  # set the second column

            vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

            ### case constr_cond get column from constraint and sample from class_balance and conditions col and opt - 2
            opt1prime = self.category_choice_constr_cond(self.p_sampling, col_idx)
            for i in np.arange(batch): 
                # col and opt - 2
                if opt1prime[i][1] == None:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1

        #### not needed # initially
        # opt1prime = random_choice_prob_index(self.p_log[col_idx]) # p_log[col_idx] - batch with selected columns, select option in the current column

        return vec, mask, col_idx, opt1prime

    def sample(self, batch): # conditional vectors for testing / evaluation - using actual probabilities
        if self.n_col == 0:
            return None
        
        case_name = "cond"
        if case_name == "actual" or case_name == "constr":
            # col and opt - 1
            col_idx = np.random.choice(np.arange(self.n_col), batch) # choice of the column
            vec = np.zeros((batch, self.n_opt), dtype='float32')
            # col and opt - 1
            opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx, col_opt_single=False) # choice of the option in the current column based on actual probs, (not log)
            
            for i in np.arange(batch):
                # col and opt - 1
                vec[i, self.cat_col_dims[col_idx[i]][0] + opt1prime[i]] = 1
                
        elif case_name == "cond" or case_name == "constr_cond":
            # col and opt - 2
            col_idx = np.array([np.random.choice(np.arange(self.n_col), size=2, replace=False) for _ in range(batch)])
            vec = np.zeros((batch, self.n_opt), dtype='float32')

            opt1prime = random_choice_prob_index_sampling(self.p_sampling,col_idx, col_opt_single=False) # choice of the option in the current column based on actual probs, (not log)

            for i in np.arange(batch):
                # col and opt - 2
                if opt1prime[i][1] == None:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                else:
                    vec[i, self.cat_col_dims[col_idx[i][0]][0] + opt1prime[i][0]] = 1 # put first 1 
                    vec[i, self.cat_col_dims[col_idx[i][1]][0] + opt1prime[i][1]] = 1 # put second 1
        return vec

class Constraints(object):
    def __init__(self, col_types, col_names, col_dims, categorical_labels):
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.categorical_labels = categorical_labels

    def calc_constraint_penalty(self, data, batch_size, constraints):
        self.data = data.detach().numpy()
        self.data_len = batch_size
        self.constr_dict = constraints
        penalty = 0

        for col, perc in self.constr_dict.items():
            col_idx = self.col_names.index(col)
            st = self.col_dims[col_idx][0]
            ed = self.col_dims[col_idx][0] + self.col_dims[col_idx][1]
            col_data= np.transpose(self.data[:, st:ed])
            binary_col_data = (col_data >= 0.5).astype(int)
            result = np.array([sum(col) for col in binary_col_data]) # count 1s of one-hot encoded column
            res_percentages = result/result.sum() # probs of each category in each column of one-hot encoded categorical var
            penalty += np.mean((res_percentages - perc) ** 2) # mse for fake data balance class and desired
        lambda_coef = 100 #!!!
        penalty = lambda_coef*penalty
        penalty = torch.tensor(penalty, requires_grad=True)
        return penalty
    