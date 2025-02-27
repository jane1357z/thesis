import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce

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
from sklearn.neural_network import MLPClassifier, MLPRegressor

class Model_evaluation(object):
    def __init__(self, epochs, steps_per_epoch, steps_d, case_name, data_name):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.steps_d = steps_d
        self.case_name = case_name
        self.data_name = data_name

    def losses_plot(self, d_loss_lst, g_loss_lst, g_loss_orig_lst, g_loss_gen_lst, g_loss_info_lst, g_loss_class_lst, g_loss_constr_lst=None):
        # plot each loss and save on pdf
        d_loss_lst = np.array(d_loss_lst)
        g_loss_lst = np.array(g_loss_lst)
        g_loss_orig_lst = np.array(g_loss_orig_lst)
        g_loss_gen_lst = np.array(g_loss_gen_lst)
        g_loss_info_lst = np.array(g_loss_info_lst)
        g_loss_class_lst = np.array(g_loss_class_lst)
        
        # get average per epoch
        d_loss = np.mean(d_loss_lst.reshape(-1, self.steps_per_epoch*self.steps_d), axis=1)
        g_loss = np.mean(g_loss_lst.reshape(-1, self.steps_per_epoch), axis=1)
        g_loss_orig = np.mean(g_loss_orig_lst.reshape(-1, self.steps_per_epoch), axis=1)
        g_loss_gen = np.mean(g_loss_gen_lst.reshape(-1, self.steps_per_epoch), axis=1)
        g_loss_info = np.mean(g_loss_info_lst.reshape(-1, self.steps_per_epoch), axis=1)
        g_loss_class = np.mean(g_loss_class_lst.reshape(-1, self.steps_per_epoch), axis=1)
        
        if g_loss_constr_lst != None: # depending on the case (with/without constr)
            g_loss_constr_lst = np.array(g_loss_constr_lst)
            g_loss_constr = np.mean(g_loss_constr_lst.reshape(-1, self.steps_per_epoch), axis=1)
        
            losses = [d_loss, g_loss, g_loss_orig, g_loss_gen, g_loss_info, g_loss_class, g_loss_constr]
            names_losses = ["d_loss", "g_loss", "g_loss_orig", "g_loss_gen", "g_loss_info", "g_loss_class", "g_loss_constr"]
            fig, axes = plt.subplots(2, 4, figsize=(19, 8))
            fig.delaxes(axes[1, 3])
        else:
            losses = [d_loss, g_loss, g_loss_orig, g_loss_gen, g_loss_info, g_loss_class]
            names_losses = ["d_loss", "g_loss","g_loss_orig", "g_loss_gen", "g_loss_info", "g_loss_class"]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        for i, ax in enumerate(axes.flat):  
            if i < len(losses):
                ax.plot(losses[i], ls='-')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(names_losses[i])
        fig.suptitle(f"{self.data_name}. Losses")
        fig.subplots_adjust(hspace=0.3)
        fig.savefig(f'results/{self.data_name}/model_eval/Losses_{self.case_name}.pdf', bbox_inches='tight')
        plt.close()

    def calc_metrics(self, time_epoch, d_auc_score):
        avg_train_epoch_time = sum(time_epoch)/len(time_epoch)

        d_auc_score = np.array(d_auc_score)
        d_auc_score_epoch = np.mean(d_auc_score.reshape(-1, self.steps_per_epoch*self.steps_d), axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(range(0, self.epochs), d_auc_score_epoch, ls='-', label=f"{self.case_name}.Discriminator AUC Score", color='b')
        plt.xlabel('epoch')
        plt.ylabel('d_auc_score')
        plt.savefig(f'results/{self.data_name}/model_eval/d_auc_score_{self.case_name}.pdf', bbox_inches='tight')
        plt.close()

        with open(f'results/{self.data_name}/model_eval/metrics.txt', "a") as f:
            f.write(f"Avg train epoch time: {avg_train_epoch_time}\n")
            f.write(f"d auc score last epoch: {d_auc_score_epoch[-1]}\n")

class Data_evaluation(object):
    def __init__(self, data_name, real_data, fake_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=None, condition_list=None, cond_ratio = None):
        self.data_name = data_name
        
        self.real_data = real_data
        self.fake_data = fake_data

        self.mixed_modes = mixed_modes
        self.cat_cols = categorical_cols
        self.non_cat_cols = general_cols + continuous_cols + mixed_cols

        self.task = task
        self.target_col = target_col

        # cases: actual, constr, cond, constr_cond
        # get real data based on constr or cond for comparison
        if class_balance == None and condition_list == None:
            case_name = "actual"
            self.row_case = "Actual"
            
        elif class_balance != None and condition_list == None:
            case_name = "constr"
            key_cat = list(class_balance.keys())[0]
            prob_dict = dict(zip(list(self.real_data[key_cat].value_counts().index), class_balance[key_cat]))
            self.real_data_constr = self.real_data.sample(n=len(self.fake_data), weights=self.real_data[key_cat].map(prob_dict))

            self.class_balance = class_balance
            
            self.row_case = "Constraints"

        elif class_balance == None and condition_list != None:
            case_name = "cond"
            self.condition_list = condition_list
            self.cond_ratio = cond_ratio


            n_samples = len(self.fake_data) # length of the real data for comparison   !!maybe other algo
            # find indices in real data for conditioned rows and not conditioned separately
            ind_cond = real_data.index[(real_data[condition_list[0]["col1"]] == condition_list[0]["cat1"]) & (real_data[condition_list[0]["col2"]] == condition_list[0]["cat2"])].to_numpy()
            ind_rest = real_data.index[~((real_data[condition_list[0]["col1"]] == condition_list[0]["cat1"]) & (real_data[condition_list[0]["col2"]] == condition_list[0]["cat2"]))].to_numpy()

            real_cond_count = len(ind_cond) # how many conditioned rows in real data
            self.real_cond_count = real_cond_count
            
            if real_cond_count < n_samples*cond_ratio: # if there are not enough conditioned rows in real data
                r_data_size = len(self.real_data)
                cond_ratio =self.real_cond_count/r_data_size

            choices = np.random.choice([0, 1], size=n_samples, p=[cond_ratio, 1-cond_ratio]) # get choice from which array to sample based on cond_ratio

            cond_sample = np.random.choice(ind_cond, size=np.sum(choices == 0), replace=False) # sample cond rows from indices
            rest_sample = np.random.choice(ind_rest, size=np.sum(choices == 1), replace=False) # sample rest rows from indices

            indices = np.empty(n_samples, dtype=int) # concatinate sampled indices
            indices[choices == 0] = cond_sample
            indices[choices == 1] = rest_sample

            self.real_data_cond = real_data.iloc[indices, :].reset_index(drop=True) # get conditioned dataframe

            self.row_case = "Conditions"

        elif class_balance != None and condition_list != None:
            case_name = "constr_cond"
            self.class_balance = class_balance
            self.condition_list = condition_list
            self.cond_ratio = cond_ratio

            self.row_case = "Constr & Cond"
        
        self.case_name = case_name


        self.col_types = {} # col_name: type
        
        for col in general_cols:
            self.col_types[col] = "general"
        for col in continuous_cols:
            self.col_types[col] = "continuous"
        for col in mixed_cols:
            self.col_types[col] = "mixed"
        for col in categorical_cols:
            self.col_types[col] = "categorical"

    def calc_average_jsd(self, cat_real, cat_fake):
        features = self.cat_cols
        jsd = []
        for col in features:
            real_pdf = dict(cat_real[col].value_counts()/cat_real[col].value_counts().sum())
            fake_pdf = dict(cat_fake[col].value_counts()/cat_fake[col].value_counts().sum())
            for key in real_pdf: # can be error, if prob=0 => no element
                fake_pdf.setdefault(key, 0)
            real_pdf_sorted = [v for k, v in sorted(real_pdf.items())]
            fake_pdf_sorted = [v for k, v in sorted(fake_pdf.items())] 
            jsd_score = distance.jensenshannon(real_pdf_sorted, fake_pdf_sorted,base=2)
            jsd.append(jsd_score)

        jsd_average = sum(jsd)/len(jsd)
        return jsd_average
        
    def calc_average_wd(self, non_cat_real, non_cat_fake):
        features = self.non_cat_cols
        wd = []

        scaler = MinMaxScaler() 

        for col in features:
            col_scaled_real = scaler.fit_transform(non_cat_real[col].to_numpy().reshape(-1, 1)).ravel()
            col_scaled_fake = scaler.transform(non_cat_fake[col].to_numpy().reshape(-1, 1)).ravel()
            wd_score = wasserstein_distance(col_scaled_real, col_scaled_fake) # wasserstein_distance(non_cat_real[col].to_numpy(), non_cat_fake[col].to_numpy())
            wd.append(wd_score)
        wd_average = sum(wd)/len(wd)
        return wd_average
         
    def calc_diff_corr_coef(self, non_cat_real, non_cat_fake):
        real_corr = np.triu(non_cat_real.corr(method = 'spearman')) # pearson
        fake_corr = np.triu(non_cat_fake.corr(method = 'spearman'))
        corr_coef_diff = np.sum(np.abs(fake_corr - real_corr)) #/np.size(real_matrix)
        return corr_coef_diff
    
    def calc_diff_theils_u(self, cat_real, cat_fake):
        features = self.cat_cols
        real_theils = np.zeros([len(self.cat_cols), len(self.cat_cols)])
        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(cat_real[var1], cat_real[var2])
                real_theils[self.cat_cols.index(var1),self.cat_cols.index(var2)] = theils_coef

        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(cat_real[var2], cat_real[var1])
                real_theils[self.cat_cols.index(var2),self.cat_cols.index(var1)] = theils_coef

        fake_theils = np.zeros([len(self.cat_cols), len(self.cat_cols)])
        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(cat_fake[var1], cat_fake[var2])
                fake_theils[self.cat_cols.index(var1),self.cat_cols.index(var2)] = theils_coef


        for i, var1 in enumerate(features):
            for var2 in features[i+1:]:
                theils_coef = theils_u(cat_fake[var2], cat_fake[var1])
                fake_theils[self.cat_cols.index(var2),self.cat_cols.index(var1)] = theils_coef
        theils_u_diff = np.sum(np.abs(fake_theils - real_theils))
        return theils_u_diff
    
    def calc_diff_corr_ratio(self, cat_real, cat_fake, non_cat_real, non_cat_fake):
        real_corr_ratio = np.zeros([len(self.cat_cols), len(self.non_cat_cols)])

        for cat_var in self.cat_cols:
            for con_var in self.non_cat_cols:
                corr_ratio = correlation_ratio(cat_real[cat_var], non_cat_real[con_var])
                real_corr_ratio[self.cat_cols.index(cat_var), self.non_cat_cols.index(con_var)] = corr_ratio

        fake_corr_ratio = np.zeros([len(self.cat_cols), len(self.non_cat_cols)])
    
        for cat_var in self.cat_cols:
            for con_var in self.non_cat_cols:
                corr_ratio = correlation_ratio(cat_fake[cat_var], non_cat_fake[con_var])
                fake_corr_ratio[self.cat_cols.index(cat_var), self.non_cat_cols.index(con_var)] = corr_ratio
        corr_ratio_diff = np.sum(np.abs(fake_corr_ratio - real_corr_ratio))
        return corr_ratio_diff

    def perform_clustering(self, real, fake):
        sil_scores = []
        for k in range(2, 7):
            kmeans = KMeans(init = "random", n_clusters = k)
            labels = kmeans.fit_predict(real)

            silhouette_avg = silhouette_score(real, labels)
            sil_scores.append(silhouette_avg)

        m_clusters = sil_scores.index(max(sil_scores))+2

        # # real fake separately
        # kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        # labels_real = kmeans_cluster.fit_predict(real)
        # centroids_real = kmeans_cluster.cluster_centers_ 

        # kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        # labels_fake = kmeans_cluster.fit_predict(fake)
        # centroids_fake = kmeans_cluster.cluster_centers_ 
        
        # centroids_diff_fake = np.sum(np.abs(centroids_real - centroids_fake))
        
        # log cluster, merged data
        merged_data = np.vstack((real, fake))
        labels_merged = np.array([1] * len(real) + [0] * len(fake))  # 1 for real, 0 for synthetic

        n_R = len(real)
        n_S = len(fake)
        c = n_R / (n_R + n_S)

        kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        labels_clustered = kmeans_cluster.fit_predict(merged_data)
        # centroids_merged = kmeans_cluster.cluster_centers_ 
        # centroids_diff_merged = np.sum(np.abs(centroids_real - centroids_merged))

        n_R_i = np.array([np.sum(labels_merged[labels_clustered == i]) for i in range(m_clusters)])
        n_i = np.array([np.sum(labels_clustered == i) for i in range(m_clusters)])
        log_cluster_merged = np.log(np.mean(((n_R_i / n_i) - c)**2))

        # log cluster , real data - gold value
        # real_fake_data = real.copy()
        # real_fake_data = real_fake_data.sample(frac=1).reset_index(drop=True)
        # # labels_r_s = np.array([1] * len_r + [0] * len_s) 
        # labels_r_s = np.random.choice([0, 1], size=len(real))
        # n_R = np.count_nonzero(labels_r_s)
        # n_S = labels_r_s.size - n_R
        # c = n_R / (n_R + n_S)
        # kmeans_cluster = KMeans(init = "random", n_clusters = m_clusters)
        # labels_clustered = kmeans_cluster.fit_predict(real_fake_data)
        # centroids_fake_real = kmeans_cluster.cluster_centers_ 

        # n_R_i = np.array([np.sum(labels_r_s[labels_clustered == i]) for i in range(m_clusters)])
        # n_i = np.array([np.sum(labels_clustered == i) for i in range(m_clusters)])
        # log_cluster_fake_real = np.log(np.mean(((n_R_i / n_i) - c) ** 2))

        return log_cluster_merged
    
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

    def ml_utility(self, real, fake, task, target_col, model_name):
        # delete classes, which are too small
        ## assume, len(fake)<len(real)

        non_target_cols = [x for x in list(self.col_types.keys()) if x != target_col]

        if task == "class":
            if model_name == "dt":
                fake_data_indep = fake.loc[:, non_target_cols]
                fake_data_dep = fake[[target_col]]
                real_data_indep = real.loc[:, non_target_cols].sample(len(fake))
                real_data_dep = real[[target_col]].sample(len(fake))

                x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(real_data_indep ,real_data_dep, test_size=0.2,random_state=42) 
                x_train_fake, _ , y_train_fake, _ = train_test_split(fake_data_indep,fake_data_dep,test_size=0.2, random_state=42) 

                acc_real, auc_real, f1_score_real, feature_importance_real = self.train_evaluate_algo_class(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                acc_fake, auc_fake, f1_score_fake, feature_importance_fake = self.train_evaluate_algo_class(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                class_metrics_diff_dt = [abs(acc_fake-acc_real), abs(auc_fake-auc_real), abs(f1_score_fake-f1_score_real)]
                # Detects missing correlations: If some features become less important in the fake data, the GAN might not be learning their dependencies correctly.
                feature_importance_diff_class = feature_importance_fake - feature_importance_real
                
                # feature importance graph
                x = np.arange(len(non_target_cols))

                plt.bar(x - 0.15, feature_importance_real, width=0.3, label='Real Data', alpha=0.9)
                plt.bar(x + 0.15, feature_importance_fake, width=0.3, label='Fake Data', alpha=0.9)

                plt.xticks(ticks=x, labels=non_target_cols, rotation=60)
                plt.ylabel('Feature Importance')
                plt.title(f'{self.case_name}.DT Feature Importance')
                plt.legend()
                plt.savefig(f"results/{self.data_name}/DT Feature Importance_{self.case_name}.pdf", dpi=300, bbox_inches='tight')
                plt.close()

                return class_metrics_diff_dt, feature_importance_diff_class
            
            elif model_name == "mlp":

                #### data scaling / encoding
                fake_data_indep_init = fake.loc[:, non_target_cols]
                real_data_indep_init = real.loc[:, non_target_cols].sample(len(fake))

                cat_cols_mlp = self.cat_cols.copy()
                non_cat_cols_mlp = self.non_cat_cols
                if target_col in cat_cols_mlp:
                    cat_cols_mlp.remove(target_col)
                if target_col in non_cat_cols_mlp:
                    non_cat_cols_mlp.remove(target_col)

                encoder=ce.OneHotEncoder(cols=cat_cols_mlp,handle_unknown='return_nan',return_df=True,use_cat_names=True)
                cat_feature_transformed = encoder.fit_transform(real_data_indep_init.loc[:, cat_cols_mlp])

                scaler = MinMaxScaler()
                non_cat_feature_transformed = scaler.fit_transform(real_data_indep_init.loc[:, non_cat_cols_mlp])
                real_data_indep = np.hstack((cat_feature_transformed, non_cat_feature_transformed)) 

                real_data_dep = pd.DataFrame()
                real_data_dep = real[target_col].astype('category').cat.codes.replace(-1, np.nan).sample(len(fake)).values.ravel()

                cat_feature_transformed = encoder.transform(fake_data_indep_init.loc[:, cat_cols_mlp])
                non_cat_feature_transformed = scaler.transform(fake_data_indep_init.loc[:, non_cat_cols_mlp])
                fake_data_indep = np.hstack((cat_feature_transformed, non_cat_feature_transformed)) 

                fake_data_dep = pd.DataFrame()
                fake_data_dep = fake[target_col].astype('category').cat.codes.replace(-1, np.nan).values.ravel()
                ####

                x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(real_data_indep,real_data_dep, test_size=0.2,random_state=42) 
                
                x_train_fake, _ , y_train_fake, _ = train_test_split(fake_data_indep,fake_data_dep,test_size=0.2, random_state=42) 

                acc_real, auc_real, f1_score_real = self.train_evaluate_algo_class(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                acc_fake, auc_fake, f1_score_fake = self.train_evaluate_algo_class(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)

                class_metrics_diff_mlp = [abs(acc_fake-acc_real), abs(auc_fake-auc_real), abs(f1_score_fake-f1_score_real)]
                return class_metrics_diff_mlp
            
        elif task == "regr":
            if model_name == "dt":
                fake_data_indep = fake.loc[:, non_target_cols]
                fake_data_dep = fake[[target_col]]
                real_data_indep = real.loc[:, non_target_cols].sample(len(fake))
                real_data_dep = real[[target_col]].sample(len(fake))

                x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(real_data_indep,real_data_dep, test_size=0.2,random_state=42) 
                x_train_fake, _ , y_train_fake, _ = train_test_split(fake_data_indep,fake_data_dep,test_size=0.2, random_state=42) 

                mape_real, evs_real, r2_score_real, feature_importance_real = self.train_evaluate_algo_regr(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                mape_fake, evs_fake, r2_score_fake, feature_importance_fake = self.train_evaluate_algo_regr(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                regr_metrics_diff_dt = [abs(mape_fake-mape_real), abs(evs_fake-evs_real), abs(r2_score_fake-r2_score_real)]
                feature_importance_diff_regr = feature_importance_fake-feature_importance_real

                # feature importance graph
                x = np.arange(len(non_target_cols))

                plt.bar(x - 0.15, feature_importance_real, width=0.3, label='Real Data', alpha=0.9)
                plt.bar(x + 0.15, feature_importance_fake, width=0.3, label='Fake Data', alpha=0.9)

                plt.xticks(ticks=x, labels=non_target_cols, rotation=60)
                plt.ylabel('Feature Importance')
                plt.title(f'{self.case_name}.DT Feature Importance')
                plt.legend()
                plt.savefig(f"results/{self.data_name}/DT Feature Importance_{self.case_name}.pdf", dpi=300, bbox_inches='tight')
                plt.close()

                return regr_metrics_diff_dt, feature_importance_diff_regr
            
            elif model_name =="mlp":

                #### data scaling / encoding
                fake_data_indep_init = fake.loc[:, non_target_cols]
                fake_data_dep = fake[[target_col]].values.ravel()
                real_data_indep_init = real.loc[:, non_target_cols].sample(len(fake))
                real_data_dep = real[[target_col]].sample(len(fake)).values.ravel()

                cat_cols_mlp = self.cat_cols.copy()
                non_cat_cols_mlp = self.non_cat_cols
                if target_col in cat_cols_mlp:
                    cat_cols_mlp.remove(target_col)
                if target_col in non_cat_cols_mlp:
                    non_cat_cols_mlp.remove(target_col)

                encoder=ce.OneHotEncoder(cols=cat_cols_mlp,handle_unknown='return_nan',return_df=True,use_cat_names=True)
                cat_feature_transformed = encoder.fit_transform(real_data_indep_init.loc[:, cat_cols_mlp])
                scaler = MinMaxScaler()
                non_cat_feature_transformed = scaler.fit_transform(real_data_indep_init.loc[:, non_cat_cols_mlp])
                real_data_indep = np.hstack((cat_feature_transformed, non_cat_feature_transformed)) 

                cat_feature_transformed = encoder.transform(fake_data_indep_init.loc[:, cat_cols_mlp])
                non_cat_feature_transformed = scaler.transform(fake_data_indep_init.loc[:, non_cat_cols_mlp])
                fake_data_indep = np.hstack((cat_feature_transformed, non_cat_feature_transformed)) 
                ####

                x_train_real, x_test_real, y_train_real, y_test_real = train_test_split(real_data_indep,real_data_dep, test_size=0.2,random_state=42) 

                x_train_fake, _ , y_train_fake, _ = train_test_split(fake_data_indep,fake_data_dep,test_size=0.2, random_state=42) 

                mape_real, evs_real, r2_score_real = self.train_evaluate_algo_regr(x_train_real, y_train_real, x_test_real, y_test_real, model_name)
                mape_fake, evs_fake, r2_score_fake = self.train_evaluate_algo_regr(x_train_fake, y_train_fake, x_test_real, y_test_real, model_name)
                regr_metrics_diff_mlp = [abs(mape_fake-mape_real), abs(evs_fake-evs_real), abs(r2_score_fake-r2_score_real)]
                return regr_metrics_diff_mlp

    def evaluation_metrics(self, real):
        fake = self.fake_data
        cat_real = real[self.cat_cols] # only categorical data columns
        cat_fake = fake[self.cat_cols] # only categorical data columns

        non_cat_real = real[self.non_cat_cols] # only non categorical data columns
        non_cat_fake = fake[self.non_cat_cols] # only non categorical data columns

        jsd_average = self.calc_average_jsd(cat_real, cat_fake)
        wd_average = self.calc_average_wd(non_cat_real, non_cat_fake)
        corr_coef_diff = self.calc_diff_corr_coef(non_cat_real, non_cat_fake)
        theils_u_diff = self.calc_diff_theils_u(cat_real, cat_fake)
        corr_ratio_diff = self.calc_diff_corr_ratio(cat_real, cat_fake, non_cat_real, non_cat_fake)

        log_cluster_merged = self.perform_clustering(real, fake)
        
        for model_name in ["dt", "mlp"]:
            if model_name == "dt":
                metrics_diff_dt, feature_importance_diff = self.ml_utility(real, fake, self.task, self.target_col, model_name)
            else:
                metrics_diff_mlp = self.ml_utility(real, fake, self.task, self.target_col, model_name)


        lst_metrics = [jsd_average, wd_average, corr_coef_diff, theils_u_diff, corr_ratio_diff, log_cluster_merged]

        lst_ml_utility = [metrics_diff_dt[0], metrics_diff_dt[1], metrics_diff_dt[2], metrics_diff_mlp[0], metrics_diff_mlp[1], metrics_diff_mlp[2]]

        return lst_metrics, lst_ml_utility

    def check_constraints(self):
        fake = self.fake_data
        # cat_balance_diff = {}
        for key, value in self.col_types.items():
            # if value == "categorical":
            #     bal_real =self.real_data[key].value_counts().to_numpy()
            #     bal_real = bal_real/bal_real.sum()
            #     bal_fake = fake[key].value_counts().reindex(list(self.real_data[key].value_counts().index), fill_value=0).to_numpy() # if some categories are missing
            #     bal_fake = bal_fake/bal_fake.sum()
            #     cat_balance_diff[key] = bal_real - bal_fake
            if key in self.class_balance.keys():
                bal_fake = fake[key].value_counts().reindex(list(self.real_data[key].value_counts().index), fill_value=0).to_numpy() # if some categories are missing
                bal_fake = bal_fake/bal_fake.sum()
                class_balance_diff = np.mean((bal_fake - self.class_balance[key]) ** 2) # MSE
        return class_balance_diff

    def check_conditions(self):
        fake = self.fake_data
        f_data_size = len(fake)
        fake_prob_cond = []
        for condition in self.condition_list:
            indicator = (fake[condition["col1"]] == condition["cat1"]) & (fake[condition["col2"]] == condition["cat2"])
            count = indicator.sum()
            fake_prob_cond.append(count/f_data_size) # count the fraction of condition occurances
        
        cond_prob_diff = fake_prob_cond[0] - self.cond_ratio
        
        r_data_size = len(self.real_data)
        real_prob_cond = []
        real_prob_cond.append(self.real_cond_count/r_data_size) # count the fraction of condition occurances in real data

        cond_prob_diff = [self.cond_ratio, fake_prob_cond, real_prob_cond]
        return cond_prob_diff

    def get_graphs(self):
        pass
    

    def evaluate_data(self):
        # df_metrics_act, df_ml_utility_act = self.evaluation_metrics(self.real_data)
        # real data depends on the case, fake data is the same
        
        if self.case_name == "actual":
            lst_metrics, lst_ml_utility = self.evaluation_metrics(self.real_data)
            add_diff = None
        elif self.case_name == "constr":
            cat_balance_diff = self.check_constraints()
            lst_metrics, lst_ml_utility = self.evaluation_metrics(self.real_data_constr)
            add_diff = cat_balance_diff
        elif self.case_name == "cond":
            cond_prob_diff= self.check_conditions()
            lst_metrics, lst_ml_utility = self.evaluation_metrics(self.real_data_cond)
            add_diff = cond_prob_diff
        elif self.case_name == "constr_cond":
            pass
        return lst_metrics, lst_ml_utility, add_diff