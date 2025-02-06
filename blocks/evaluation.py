

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy  # for binary or multi-class classification
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch

class Model_evaluation(object):
    def __init__(self):
        self.penalties = []

    def penalty_measure(self, penalty):
        penlaty_np = penalty.detach().numpy()
        self.penalties.append(penlaty_np)

    def penalty_graph(self):
        plt.scatter(range(len(self.penalties)), self.penalties)
        plt.show()


    def check_timers(self):
        pass

class Data_evaluation(object):
    def __init__(self, real_data, fake_data, class_balance, condition_list, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes):
        self.real_data = real_data
        self.fake_data = fake_data
        self.class_balance = class_balance
        self.mixed_modes = mixed_modes
        self.condition_list = condition_list
        self.col_types = {} # col_name: type
        
        for col in general_cols:
            self.col_types[col] = "general"
        for col in continuous_cols:
            self.col_types[col] = "continuous"
        for col in mixed_cols:
            self.col_types[col] = "mixed"
        for col in categorical_cols:
            self.col_types[col] = "categorical"

    def calc_metrics(self): # !!!
        gen_max_min = {}
        gen_mean = {}
        cat_balance = {}
        con_max_min = {}
        con_mean = {}
        for key, value in self.col_types.items():
            if value == "general":
                max_min = [[self.real_data[key].max(),self.fake_data[key].max()], [self.real_data [key].min(), self.fake_data [key].min()]]
                gen_max_min[key] = max_min
                gen_mean[key] = [self.real_data[key].mean(), self.fake_data[key].mean()]
            elif value == "categorical":
                cat_balance[key] = [self.real_data[key].value_counts(), self.fake_data[key].value_counts()]
            elif value == "continuous":
                max_min = [[self.real_data[key].max(),self.fake_data[key].max()], [self.real_data[key].min(), self.fake_data[key].min()]]
                con_max_min[key] = max_min
                con_mean[key] = [self.real_data[key].mean(), self.fake_data[key].mean()]
            elif value== "mixed":
                pass
        return gen_max_min, gen_mean, cat_balance, con_max_min, con_mean
    
    def check_constraints(self): # !!!
        cat_balance = {}
        for key, value in self.col_types.items():
            if value == "categorical":
                bal_real =self.real_data[key].value_counts().to_numpy()
                bal_real = bal_real/bal_real.sum()
                bal_fake = self.fake_data[key].value_counts().reindex(list(self.real_data[key].value_counts().index), fill_value=0).to_numpy() # if some categories are missing
                bal_fake = bal_fake/bal_fake.sum()
                cat_balance[key] = [bal_real, bal_fake]
        return cat_balance

    def check_conditions(self):
        f_data_size = len(self.fake_data)
        result_cond = []
        for condition in self.condition_list:
            indicator = (self.fake_data[condition["col1"]] == condition["val1"]) & (self.fake_data[condition["col2"]] == condition["val2"])
            count = indicator.sum()
            result_cond.append(count/f_data_size) # count the fraction of conndition occuarences

        r_data_size = len(self.real_data)
        orig_result_cond = []
        for condition in self.condition_list:
            indicator = (self.real_data[condition["col1"]] == condition["val1"]) & (self.real_data[condition["col2"]] == condition["val2"])
            count = indicator.sum()
            orig_result_cond.append(count/r_data_size) # count the fraction of conndition occuarences
        return result_cond, orig_result_cond

    def get_graphs(self):
        pass
    