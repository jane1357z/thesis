import numpy as np


def st_ed_col(target_col, col_names, col_dims): # find indeces (starting and ending) of columns for labels
    col_idx = col_names.index(target_col)
    st = col_dims[col_idx][0]
    ed = col_dims[col_idx][0] + col_dims[col_idx][1]
    return st, ed

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
        return self.data[idx]
    


