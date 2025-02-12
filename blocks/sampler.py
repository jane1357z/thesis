import numpy as np


def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

class Sampler(object):
    def __init__(self, data, col_types, col_names, col_dims, categorical_labels, case_name):
        super(Sampler, self).__init__()
        self.data = np.array(data)
        self.col_types = col_types
        self.col_names = col_names
        self.col_dims = col_dims
        self.data_len = len(data)
        self.categorical_labels = categorical_labels
        self.cat_ones_row_idx = []
        self.case_name = case_name

        for key, value in self.col_types.items(): #categorical_labels
            if value=="categorical":
                st, ed = st_ed_col(key, self.col_names, self.col_dims)
                tmp = []
                for j in range(ed-st):
                    tmp.append(np.nonzero(np.array(data)[:, st+j]))
                self.cat_ones_row_idx.append(tmp) # row indeces where category is 1 for each column and its category
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
        # col and opt - 1
        if self.case_name == "actual" or self.case_name == "constr":
            for c, o in zip(col, opt):
                idx.append(np.random.choice(self.cat_ones_row_idx[c][o][0])) # get random row by the condition
                
        elif self.case_name == "cond" or self.case_name == "constr_cond":
        # col and opt - 2
            for c, o in zip(col, opt):
                if o[1] == None:
                    idx.append(np.random.choice(self.cat_ones_row_idx[c[0]][o[0]][0]))
                else:
                    cond1 = self.cat_ones_row_idx[c[0]][o[0]][0]  # rows by first col-opt
                    cond2 = self.cat_ones_row_idx[c[1]][o[1]][0]  # rows by second col-opt

                    intersection = np.intersect1d(cond1, cond2) # get random row by the condition, where two options in two columns intersect

                    # Make a random choice if intersection is not empty
                    if intersection.size > 0:
                        idx.append(np.random.choice(intersection))
                    else:
                        idx = np.random.choice(np.arange(self.data_len), 1) #!!!

        return self.data[idx]
    


