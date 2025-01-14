

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy  # for binary or multi-class classification
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


class Evaluation(object):
    def __init__(self):
        pass

    def eval_model(self):
        pass

    def fake_eval_init(self, real_data, fake_data, class_balance, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes):
        self.real_data = real_data
        self.fake_data = fake_data
        self.class_balance = class_balance
        self.mixed_modes = mixed_modes
        self.col_types = {} # col_name: type
        
        for col in general_cols:
            self.col_types[col] = "general"
        for col in continuous_cols:
            self.col_types[col] = "continuous"
        for col in mixed_cols:
            self.col_types[col] = "mixed"
        for col in categorical_cols:
            self.col_types[col] = "categorical"

    def calc_metrics(self):
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
                max_min = [[self.real_data[key].max(),self.fake_data[key].max()], [self.real_data [key].min(), self.fake_data [key].min()]]
                con_max_min[key] = max_min
                con_mean[key] = [self.real_data[key].mean(), self.fake_data[key].mean()]
            elif value== "mixed":
                pass
        return gen_max_min, gen_mean, cat_balance, con_max_min, con_mean
    
    def check_constraints(self):
        cat_balance = {}
        for key, value in self.col_types.items():
            if value == "categorical":
                bal_real =self.real_data[key].value_counts().to_numpy()
                bal_real = bal_real/bal_real.sum()
                bal_fake = self.fake_data[key].value_counts().to_numpy()
                bal_fake = bal_fake/bal_fake.sum()
                cat_balance[key] = [bal_real, bal_fake]
        return cat_balance

    def get_graphs(self):
        pass
    