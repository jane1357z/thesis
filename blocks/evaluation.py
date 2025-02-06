

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy  # for binary or multi-class classification
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import torch

from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from dython.nominal import theils_u, correlation_ratio
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier, MLPRegressor


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
        self.cat_cols = categorical_cols
        self.non_cat_cols = general_cols + continuous_cols + mixed_cols

        self.cat_real_data = self.real_data[self.cat_cols]
        self.cat_fake_data = self.fake_data[self.cat_cols]

        self.non_cat_real_data = self.real_data[self.non_cat_cols]
        self.non_cat_fake_data = self.real_data[self.non_cat_cols]

        self.col_types = {} # col_name: type
        
        for col in general_cols:
            self.col_types[col] = "general"
        for col in continuous_cols:
            self.col_types[col] = "continuous"
        for col in mixed_cols:
            self.col_types[col] = "mixed"
        for col in categorical_cols:
            self.col_types[col] = "categorical"

    def calc_average_jsd(self, features):
        jsd = []
        for col in features:
            real_pdf = dict(self.cat_real_data[col].value_counts()/self.cat_real_data[col].value_counts().sum())
            fake_pdf = dict(self.cat_fake_data[col].value_counts()/self.cat_fake_data[col].value_counts().sum())
            for key in real_pdf: # can be error, if prob=0 => no element
                fake_pdf.setdefault(key, 0)
            real_pdf_sorted = [v for k, v in sorted(real_pdf.items())]
            fake_pdf_sorted = [v for k, v in sorted(fake_pdf.items())] 
            jsd_score = distance.jensenshannon(real_pdf_sorted, fake_pdf_sorted,base=2)
            jsd.append(jsd_score)

        jsd_average = sum(jsd)/len(jsd)
        return jsd_average
        
    def calc_average_wd(self, features):
        wd = []
        for col in features:
            wd_score = wasserstein_distance(self.non_cat_real_data[col].to_numpy(), self.non_cat_fake_data[col].to_numpy())
            wd.append(wd_score)
        wd_average = sum(wd)/len(wd)
        return wd_average
        
    
    def calc_diff_corr_coef(self):
        real_corr = np.triu(self.non_cat_real_data.corr(method = 'spearman')) # pearson
        fake_corr = np.triu(self.non_cat_fake_data.corr(method = 'spearman'))
        result = np.sum(np.abs(real_corr - fake_corr)) #/np.size(real_matrix)
        return result
    
    def calc_diff_theils_u(self, features):
        real_theil = np.zeros([len(self.cat_cols), len(self.cat_cols)])
        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(self.cat_real_data[var1], self.cat_real_data[var2])
                real_theil[self.cat_cols.index(var1),self.cat_cols.index(var2)] = theils_coef

        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(self.cat_real_data[var2], self.cat_real_data[var1])
                real_theil[self.cat_cols.index(var2),self.cat_cols.index(var1)] = theils_coef

        fake_theil = np.zeros([len(self.cat_cols), len(self.cat_cols)])
        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(self.cat_fake_data[var1], self.cat_fake_data[var2])
                fake_theil[self.cat_cols.index(var1),self.cat_cols.index(var2)] = theils_coef


        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(self.cat_fake_data[var2], self.cat_fake_data[var1])
                fake_theil[self.cat_cols.index(var2),self.cat_cols.index(var1)] = theils_coef
        result = np.sum(np.abs(real_theil - fake_theil))
        return result
    
    def calc_diff_corr_ratio(self):
        real_corr_ratio = np.zeros([len(self.cat_cols), len(self.non_cat_cols)])

        for cat_var in self.cat_cols:
            for con_var in self.non_cat_cols:
                corr_ratio = correlation_ratio(self.cat_real_data[cat_var], self.non_cat_real_data[con_var])
                real_corr_ratio[self.cat_cols.index(cat_var), self.non_cat_cols.index(con_var)] = corr_ratio

        fake_corr_ratio = np.zeros([len(self.cat_cols), len(self.non_cat_cols)])
    
        for cat_var in self.cat_cols:
            for con_var in self.non_cat_cols:
                corr_ratio = correlation_ratio(self.cat_fake_data[cat_var], self.non_cat_fake_data[con_var])
                fake_corr_ratio[self.cat_cols.index(cat_var), self.non_cat_cols.index(con_var)] = corr_ratio
        result = np.sum(np.abs(real_corr_ratio - fake_corr_ratio))
        return result

    def perform_clustering(self):
        m_clusters = 4
        # real fake separately
        kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        labels_real = kmeans_cluster.fit_predict(self.real_data)
        centroids_real = kmeans_cluster.cluster_centers_ 

        kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        labels_fake = kmeans_cluster.fit_predict(self.fake_data)
        centroids_fake = kmeans_cluster.cluster_centers_ 
        
        centroids_diff_fake = np.sum(np.abs(centroids_real - centroids_fake))
        
        # log cluster, merged data
        merged_data = np.vstack((self.real_data, self.fake_data))
        labels_merged = np.array([1] * len(self.real_data) + [0] * len(self.fake_data))  # 1 for real, 0 for synthetic

        n_R = len(self.real_data)
        n_S = len(self.fake_data)
        c = n_R / (n_R + n_S)

        kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        labels_clustered = kmeans_cluster.fit_predict(merged_data)
        centroids_merged = kmeans_cluster.cluster_centers_ 
        centroids_diff_merged = np.sum(np.abs(centroids_real - centroids_merged))

        n_R_i = np.array([np.sum(labels_merged[labels_clustered == i]) for i in range(m_clusters)])
        n_i = np.array([np.sum(labels_clustered == i) for i in range(m_clusters)])
        log_cluster_merged = np.log(np.mean(((n_R_i / n_i) - c)**2))

        # log cluster , real data - gold value
        
        real_fake_data = self.real_data.copy()
        real_fake_data = real_fake_data.sample(frac=1).reset_index(drop=True)
        # labels_r_s = np.array([1] * len_r + [0] * len_s) 
        labels_r_s = np.random.choice([0, 1], size=len(self.real_data))
        n_R = np.count_nonzero(labels_r_s)
        n_S = labels_r_s.size - n_R
        c = n_R / (n_R + n_S)
        kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        labels_clustered = kmeans_cluster.fit_predict(real_fake_data)
        centroids_fake_real = kmeans_cluster.cluster_centers_ 

        n_R_i = np.array([np.sum(labels_r_s[labels_clustered == i]) for i in range(m_clusters)])
        n_i = np.array([np.sum(labels_clustered == i) for i in range(m_clusters)])
        log_cluster_fake_real = np.log(np.mean(((n_R_i / n_i) - c) ** 2))
        return centroids_diff_fake, centroids_diff_merged, centroids_fake_real, log_cluster_merged, log_cluster_fake_real
    

    def train_evaluate_algo_class(self, x_train, y_train, x_test, y_test, model_name):
        if model_name == "dt":
            model = tree.DecisionTreeClassifier(random_state=42)
        elif model_name == "mlp":
            model = MLPClassifier(random_state=42,max_iter=100)
        
        model.fit(x_train, y_train)
        pred_labels = model.predict(x_test)

        if len(np.unique(y_train))>2: # multi-class classification
            pred_prob = model.predict_proba(x_test)
            # y_score = np.zeros((len(y_test), len(labels_unique)))  # if classes are missing
            # for i, label in enumerate(pred_labels):
            #     class_index = np.where(labels_unique == label)[0][0]
            #     y_score[i, class_index] = 1 
            acc = metrics.accuracy_score(y_test,pred_labels)
            auc = metrics.roc_auc_score(y_test, pred_prob,average="weighted",multi_class="ovr") # ovr computes the AUC of each class against the rest (the number of true instances for each label)
            f1_score_ = metrics.f1_score(y_test, pred_labels,average="weighted") # == precision_recall_fscore_support(y_test, pred_labels)
        else: # binary classification
            pred_prob = model.predict_proba(x_test)[:,1]
            acc = metrics.accuracy_score(y_test,pred_labels)
            auc = metrics.roc_auc_score(y_test, pred_prob)
            f1_score_ = metrics.f1_score(y_test,pred_labels)
            
        if model_name == "dt":
            feature_importance = model.feature_importances_
            return acc, auc, f1_score_, feature_importance
        else:
            return acc, auc, f1_score_

    def train_evaluate_algo_regr(self, x_train, y_train, x_test, y_test, model_name):
        if model_name == "dt":
            model = tree.DecisionTreeRegressor(random_state=42)
        elif model_name == "mlp":
            model = MLPRegressor(random_state=42,max_iter=100)
        
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        mape = metrics.mean_absolute_percentage_error(y_test, pred)
        evs = metrics.explained_variance_score(y_test, pred)
        r2_score_ = metrics.r2_score(y_test, pred)
            
        if model_name == "dt":
            feature_importance = model.feature_importances_
            return mape, evs, r2_score_, feature_importance
        else:
            return mape, evs, r2_score_

    def ml_utility(self, task, target_col, model_name):
        # delete classes, which are too small
        ## assume, len(fake)<len(real)



        non_target_cols = [x for x in list(self.col_types.keys()) if x != target_col]
        fake_data_indep = self.fake_data.loc[:, non_target_cols]
        fake_data_dep = self.fake_data.loc[:, target_col]
        real_data_indep = self.real_data.iloc[:, non_target_cols].sample(len(fake_data_indep))
        real_data_dep = self.real_data.loc[:, target_col].sample(len(fake_data_indep))

        x_train_real, x_test_real, y_train_real, y_test_real = model_selection.train_test_split(real_data_indep ,real_data_dep, test_size=0.2,random_state=42) 
        x_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(fake_data_indep,fake_data_dep,test_size=0.2, random_state=42) 

        task = "class" # "regr"
        model_name = "dt" # mlp"
        if task == "class":
            if model_name == "dt":
                acc_real, auc_real, f1_score_real, feature_importance_real = self.train_evaluate_algo_class(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                acc_fake, auc_fake, f1_score_fake, feature_importance_fake = self.train_evaluate_algo_class(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                class_metrics_diff_dt = [abs(acc_real-acc_fake), abs(auc_real-auc_fake), abs(f1_score_real-f1_score_fake)]
                # Detects missing correlations: If some features become less important in the fake data, the GAN might not be learning their dependencies correctly.
                feature_importance_diff_class = feature_importance_real-feature_importance_fake
                
                # graph
                x_labels = non_target_cols
                x = np.arange(len(x_labels))

                plt.bar(x - 0.2, feature_importance_real, width=0.4, label='Real Data', alpha=0.8)
                plt.bar(x + 0.2, feature_importance_fake, width=0.4, label='Fake Data', alpha=0.8)

                plt.xticks(ticks=x, labels=x_labels)
                plt.ylabel('Feature Importance')
                plt.title('Comparison of Decision Tree Feature Importance (Real vs. Fake)')
                plt.legend()
                plt.show()
            elif model_name == "mlp":
                # scaler = StandardScaler() # for normal distr / k-means
                scaler = MinMaxScaler()
                scaler.fit(real_data_indep)
                x_train_real = scaler.transform(x_train_real)
                x_test_real = scaler.transform(x_test_real)

                # scaler = StandardScaler() # for normal distr / k-means
                scaler = MinMaxScaler()
                scaler.fit(fake_data_indep)
                x_train_fake = scaler.transform(x_train_fake)

                acc_real, auc_real, f1_score_real = self.train_evaluate_algo_class(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                acc_fake, auc_fake, f1_score_fake = self.train_evaluate_algo_class(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)

                class_metrics_diff_mlp = [abs(acc_real-acc_fake), abs(auc_real-auc_fake), abs(f1_score_real-f1_score_fake)]

        elif task == "regr":
            if model_name == "dt":
                mape_real, evs_real, r2_score_real, feature_importance_real = self.train_evaluate_algo_regr(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                mape_fake, evs_fake, r2_score_fake, feature_importance_fake = self.train_evaluate_algo_regr(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                regr_metrics_diff_dt = [abs(mape_real-mape_fake), abs(evs_real-evs_fake), abs(r2_score_real-r2_score_fake)]
                feature_importance_diff_regr = feature_importance_real-feature_importance_fake

                # graph
                x_labels = non_target_cols
                x = np.arange(len(x_labels))

                plt.bar(x - 0.2, feature_importance_real, width=0.4, label='Real Data', alpha=0.8)
                plt.bar(x + 0.2, feature_importance_fake, width=0.4, label='Fake Data', alpha=0.8)

                plt.xticks(ticks=x, labels=x_labels)
                plt.ylabel('Feature Importance')
                plt.title('Comparison of Decision Tree Feature Importance (Real vs. Fake)')
                plt.legend()
                plt.show()
            elif model_name =="mlp":
                # scaler = StandardScaler() # for normal distr / k-means
                scaler = MinMaxScaler()
                scaler.fit(real_data_indep)
                x_train_real = scaler.transform(x_train_real)
                x_test_real = scaler.transform(x_test_real)

                # scaler = StandardScaler() # for normal distr / k-means
                scaler = MinMaxScaler()
                scaler.fit(fake_data_indep)
                x_train_fake = scaler.transform(x_train_fake)

                mape_real, evs_real, r2_score_real = self.train_evaluate_algo_regr(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                mape_fake, evs_fake, r2_score_fake = self.train_evaluate_algo_regr(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                regr_metrics_diff_mlp = [abs(mape_real-mape_fake), abs(evs_real-evs_fake), abs(r2_score_real-r2_score_fake)]

    def calc_min_max(self):
        
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
    
    def evaluation_metrics(self):
        pass

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
            indicator = (self.fake_data[condition["col1"]] == condition["cat1"]) & (self.fake_data[condition["col2"]] == condition["cat2"])
            count = indicator.sum()
            result_cond.append(count/f_data_size) # count the fraction of conndition occuarences

        r_data_size = len(self.real_data)
        orig_result_cond = []
        for condition in self.condition_list:
            indicator = (self.real_data[condition["col1"]] == condition["cat1"]) & (self.real_data[condition["col2"]] == condition["cat2"])
            count = indicator.sum()
            orig_result_cond.append(count/r_data_size) # count the fraction of conndition occuarences
        return result_cond, orig_result_cond

    def get_graphs(self):
        pass
    