import numpy as np
import torch

def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

# selects option for each sample in batch, cumulative sum maybe to change 
def random_choice_prob_index(a, axis=1): # takes indices based on cumulative probability distributions in array of normalized log-probabilities - sampling from normalized log-probs 
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis) # random value between 0 and 1 for each category in feature in batch
    # returns the index of the first threshold where the cumulative sum exceeds the random number (r) - index of category for each sample in batch
    return (a.cumsum(axis=axis) > r).argmax(axis=axis) # compares cumulative sum of probabilities with the random numbers

def random_choice_prob_index_sampling(probs,col_idx): # sampling from actual probs, not log - selects option for each sample in batch
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
    
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

    def category_choice_constr_cond(self, probs, col_idx): # !!!
        option_list = []
        class_balance_col = list(self.class_balance.keys())[0]
        for i in col_idx:
            if list(self.categorical_labels.keys())[i] == class_balance_col:
                pp = self.class_balance[class_balance_col]
                option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
                # compose condition
            else:
                # compose condition
                pp = probs[i]
                option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

        return np.array(option_list).reshape(col_idx.shape)
    
    def category_choice_condition(self, probs, col_idx):
        # compose condition
        option_list = []
        for i in col_idx:
            pp = probs[i]
            option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))
        
        return np.array(option_list).reshape(col_idx.shape)
    
    def sample_train(self, batch): # conditional vectors for training - using log probabilities
        if self.n_col == 0:
            return None
        # col and opt - 1
        col_idx = np.random.choice(np.arange(self.n_col), batch) # column selection for each sample in batch
        
        # col and opt - 2
        # col_idx = np.array([np.random.choice(np.arange(self.n_col), size=2, replace=False) for _ in range(batch)])

        # here only categorical columns
        vec = np.zeros((batch, self.n_opt), dtype='float32') # one-hot encoded conditional vectors only for categorical colimns!

        # col and opt - 1, mask is not used anywhere?
        mask = np.zeros((batch, self.n_col), dtype='float32') # categorical column was selected for each sample
        mask[np.arange(batch), col_idx] = 1  # one-hot encoded, with only one for the column selection

        # col and opt - 2
        # mask = np.zeros((batch, 2, 2), dtype='float32') # categorical column was selected for each sample
        # mask[np.arange(batch), col_idx[:, 0], 0] = 1  # set the first column
        # mask[np.arange(batch), col_idx[:, 1], 1] = 1  # set the second column

        #### not needed # initially
        # opt1prime = random_choice_prob_index(self.p_log[col_idx]) # p_log[col_idx] - batch with selected columns, select option in the current column
        
        ####### cases
        ### case 0 original probs (nothing with constraints or conditions)
        # opt1prime = self.category_choice_actual_prob(self.p_sampling, col_idx)
        ### case 1 get column from constraint and sample from class_balance
        # opt1prime = self.category_choice_constraint(self.p_sampling, col_idx)
        ### case 2 get column from constraint and sample from class_balance and conditions col and opt - 2
        # opt1prime = self.category_choice_constr_cond(self.p_sampling, col_idx)
        ### case 3 get columns from condtions and sample from condtions col and opt - 2
        opt1prime = self.category_choice_condition(self.p_sampling, col_idx)



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
            result = np.array([sum(col) for col in col_data]) # count 1s of one-hot encoded column
            res_percentages = result/result.sum() # probs of each category in each column of one-hot encoded categorical var
            penalty += np.mean((res_percentages - perc) ** 2) # mse for fake data balance class and desired
        lambda_coef = 10 #!!!
        penalty = lambda_coef*penalty
        penalty = torch.tensor(penalty, requires_grad=True)
        return penalty
    

# class Conditions(object):
#     def __init__(self, col1, col2, cond):
#         self.col1 = col1
#         self.col2 = col2
#         self.cond = cond

#     def check_condition(self, data, batch_size):
#         # indicator
#         pass

