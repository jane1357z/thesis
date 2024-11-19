import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.mixture import BayesianGaussianMixture

class DataPrep:
    def __init__(self, row_df: pd.DataFrame, categorical_cols: list, continuous_cols: list, mixed_cols: list, general_cols: list, clusters_numbers: dict, mode_threshold: float, mixed_modes: dict):
        self.row_df = row_df
        self.clusters_numbers = clusters_numbers
        self.mixed_modes = mixed_modes
        self.mode_threshold = mode_threshold

        self.col_types = {} # col_name: type
        self.categrical_labels = {} # col_name: labels list (only for categorical)

        for col in general_cols:
            self.col_types[col] = "general"
        for col in continuous_cols:
            self.col_types[col] = "continuous"
        for col in mixed_cols:
            self.col_types[col] = "mixed"
        for col in categorical_cols:
            self.col_types[col] = "categorical"
            # categories_temp = row_df[col].unique()
            # one_hot_dict = {cat: list(np.eye(len(categories_temp))[idx]) for idx, cat in enumerate(categories_temp)}
            self.categrical_labels[col] = row_df[col].unique() #one_hot_dict 
        

        self.cols_mapping = {"general": [], "continuous": [], "mixed": [], "categorical": []} # for each type - col_name, # of columns in transformed
        self.vector_repres = {"general": [], "continuous": [], "mixed": [], "categorical": []} # for each type - col_name, transformed array
        self.models_cont_mixed= {"means": {}, "stds":{}, "components": {}} # used in inverse, model outputs for cont and mixed data
        self.gen_min_max = {"min": {}, "max": {}} # used in inverse, min and max values for general transf



    def transform(self):
        for key, value in self.col_types.items():
            current = self.row_df[key]
            if value == "general":
                # get min, max values
                max_v = self.row_df[key].max()
                min_v = self.row_df[key].min()
                self.gen_min_max["min"][key] = min_v
                self.gen_min_max["max"][key] = max_v
                # transform
                feature_transformed = 2*(self.row_df[key]-min_v)/(max_v-min_v)-1
                feature_transformed = feature_transformed.to_numpy().reshape(-1, 1)

                self.cols_mapping["general"].append([key, 1])
                self.vector_repres["general"].append(feature_transformed.tolist())

            elif value == "categorical":
                current = self.row_df[key]
                encoder=ce.OneHotEncoder(cols=key,handle_unknown='return_nan',return_df=True,use_cat_names=True)



                feature_transformed = encoder.fit_transform(current)
                feature_transformed = feature_transformed.to_numpy()

                self.cols_mapping["categorical"].append([key, feature_transformed.shape[1]])
                self.vector_repres["categorical"].append(feature_transformed.tolist())

            elif value == "continuous":
                current = current.to_numpy()
                n_clusters = self.clusters_numbers[key]
                
                # fit model
                gm = BayesianGaussianMixture(
                n_components = n_clusters, 
                weight_concentration_prior=0.001,  
                random_state=42)

                gm.fit(current.reshape([-1, 1]))
                mode_freq = (pd.Series(gm.predict(current.reshape([-1, 1]))).value_counts().keys()) # mode frequency descending order
                old_comp = gm.weights_ > self.mode_threshold # boolean array old_comp; modes with weights below this threshold are considered inactive or insignificant
                signif_modes_bool = []

                for i in range(n_clusters):
                    if (i in (mode_freq)) & old_comp[i]: # if both conditions are met (the mode is active and its weight is significant)
                        signif_modes_bool.append(True)
                    else:
                        signif_modes_bool.append(False)
                self.models_cont_mixed["components"][key] = signif_modes_bool

                # transform
                means = gm.means_.reshape([-1])
                stds = np.sqrt(gm.covariances_).reshape([-1])
                self.models_cont_mixed["means"][key] = means
                self.models_cont_mixed["stds"][key] = stds

                probs = gm.predict_proba(current.reshape([-1, 1])) # predicts the probability of current value belonging to each cluster
                probs = probs[:, signif_modes_bool] # only significant modes
                n_opts = sum(signif_modes_bool) # number of significant modes

                opt_sel = np.zeros(len(current), dtype='int')
                for i in range(len(current)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)

                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp) # sample from probability to choose cluster 
                    
                idx = np.arange((len(current)))
                features = np.empty(shape=(len(current.reshape([-1, 1])), n_clusters))
                features = np.abs(current.reshape([-1, 1]) - means) / (4 * stds) # calculate alphas
                features = features[:, signif_modes_bool]
                feature_transformed = features[idx, opt_sel].reshape([-1, 1])
                feature_transformed = np.clip(feature_transformed, -.99, .99)
                mean_probs_onehot = np.zeros_like(probs)
                mean_probs_onehot[np.arange(len(probs)), opt_sel] = 1 # one-hot encoding of cluster

                feature_transformed = np.concatenate([feature_transformed, mean_probs_onehot], axis = 1) # final result. alpha and one-hot encoding of cluster

                ###### change the order, starting from the biggest 
                # col_sums = mean_probs_onehot.sum(axis=0)
                # largest_indices = np.argsort(-1*col_sums)[:mean_probs_onehot.shape[1]] # -1 for descending order
                # re_ordered_means = np.zeros_like(mean_probs_onehot)
                # for id,val in enumerate(largest_indices):
                #     re_ordered_means[:,id] = mean_probs_onehot[:,val]
                # # ordering = []
                # # ordering.append(largest_indices)
                # vector_repres_cont = []
                # vector_repres_cont += [feature_transformed, re_ordered_means]
                ######

                self.cols_mapping["continuous"].append([key, feature_transformed.shape[1]])
                self.vector_repres["continuous"].append(feature_transformed.tolist())

            elif value== "mixed":
                current = current.to_numpy()
                filter_cont = ~np.isin(current, self.mixed_modes[key]) # true for categorical part (modes)
                n_clusters = self.clusters_numbers[key]

                gm = BayesianGaussianMixture(
                    n_components = n_clusters, 
                    weight_concentration_prior=0.001,  
                    random_state=42)

                gm.fit(current.reshape([-1, 1])) # Fits the model on all values
                mode_freq = (pd.Series(gm.predict(current.reshape([-1, 1]))).value_counts().keys()) # mode frequency descending order

                old_comp = gm.weights_ > self.mode_threshold # boolean array old_comp; modes with weights below this threshold are considered inactive or insignificant

                signif_modes_bool = []

                for i in range(n_clusters):
                    if (i in (mode_freq)) & old_comp[i]: # if both conditions are met (the mode is active and its weight is significant)
                        signif_modes_bool.append(True)
                    else:
                        signif_modes_bool.append(False)

                self.models_cont_mixed["components"][key] = signif_modes_bool

                # transform
                means = gm.means_.reshape([-1])
                stds = np.sqrt(gm.covariances_).reshape([-1])
                self.models_cont_mixed["means"][key] = means
                self.models_cont_mixed["stds"][key] = stds

                zero_std_list = [] # index of mean for modes

                for i in range(len(self.mixed_modes[key])): # define the mode to mean index
                    mode = self.mixed_modes[key][i]
                    dist = []
                    for idx, val in enumerate(list(means.flatten())):
                        dist.append(abs(mode - val))
                    index_min = np.argmin(np.array(dist))
                    self.mixed_modes[key][i] = [index_min, mode] # mode index and mode value
                    zero_std_list.append(index_min)

                current = current.reshape([-1, 1])
                current_cont = current[filter_cont] # get only continuous values
                probs = gm.predict_proba(current_cont.reshape([-1, 1])) # predicts the probability of current value belonging to each cluster
                probs = probs[:, signif_modes_bool] # only significant modes
                n_opts = sum(signif_modes_bool) # number of significant modes

                opt_sel = np.zeros(len(current_cont), dtype='int')
                for i in range(len(current_cont)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)

                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp) # sample from probability to choose cluster

                idx = np.arange((len(current_cont)))
                features = np.empty(shape=(len(current_cont.reshape([-1, 1])), n_clusters))
                features = np.abs(current_cont.reshape([-1, 1]) - means) / (4 * stds) # calculate alphas
                features = features[:, signif_modes_bool]
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(features, -.99, .99)
                mean_probs_onehot = np.zeros_like(probs)
                mean_probs_onehot[np.arange(len(probs)), opt_sel] = 1 # one-hot encoding of cluster

                feature_transformed = np.zeros([len(current), 1+mean_probs_onehot.shape[1]])
                col_modes = {pair[1]: pair[0] for pair in self.mixed_modes[key]} # get dict from mixed_modes[key] list to find category index
                features_list_counter = 0
                for idx, val in enumerate(current): # transform each value separately
                    if val.item() in col_modes.keys():
                        category_index = col_modes.get(val.item())
                        feature_transformed[idx, 0] = 0 # alpha for categorical part is 0
                        feature_transformed[idx, (zero_std_list[category_index]+1)] = 1    
                    else:
                        feature_transformed[idx, 0] = features[features_list_counter] # alpha 
                        feature_transformed[idx, 1:] = mean_probs_onehot[features_list_counter]
                        features_list_counter = features_list_counter + 1

            
                ###### reordering
                # just_onehot = feature_transformed[:,1:]
                # re_ordered_jhot= np.zeros_like(just_onehot)
                # n = just_onehot.shape[1]
                # col_sums = just_onehot.sum(axis=0)
                # largest_indices = np.argsort(-1*col_sums)[:n]
                # for id,val in enumerate(largest_indices):
                #         re_ordered_jhot[:,id] = just_onehot[:,val]
                # feature_transformed = feature_transformed[:,0].reshape([-1, 1])
                # vector_repres_mixed = []
                # vector_repres_mixed += [feature_transformed]
                ######

                self.cols_mapping["mixed"].append([key, feature_transformed.shape[1]])
                self.vector_repres["mixed"].append(feature_transformed.tolist())
        
        # combine together all types
        transformed_data_arrays = [] # transformed data
        transformed_col_names = [] # track columns names
        transformed_col_dims = [] # [number of starting column, number of columns]

        for key, value in self.cols_mapping.items():
            if not transformed_col_dims:
                counter = 0
            else:
                counter = transformed_col_dims[len(transformed_col_dims)-1][0]+transformed_col_dims[len(transformed_col_dims)-1][1]
            if key=="general":
                for i in range(len(self.vector_repres[key])):
                    transformed_data_arrays.append(self.vector_repres[key][i])
                    transformed_col_names.append(self.cols_mapping[key][i][0])
                    transformed_col_dims.append([counter,self.cols_mapping[key][i][1]])
                    counter += self.cols_mapping[key][i][1]
            elif key=="categorical":
                for i in range(len(self.vector_repres[key])):
                    transformed_data_arrays.append(self.vector_repres[key][i])
                    transformed_col_names.append(self.cols_mapping[key][i][0])
                    transformed_col_dims.append([counter, self.cols_mapping[key][i][1]])
                    counter += self.cols_mapping[key][i][1]
            elif key=="continuous":
                for i in range(len(self.vector_repres[key])):
                    transformed_data_arrays.append(self.vector_repres[key][i])
                    transformed_col_names.append(self.cols_mapping[key][i][0])
                    transformed_col_dims.append([counter, self.cols_mapping[key][i][1]])
                    counter += self.cols_mapping[key][i][1]
            elif key=="mixed":
                for i in range(len(self.vector_repres[key])):
                    transformed_data_arrays.append(self.vector_repres[key][i])
                    transformed_col_names.append(self.cols_mapping[key][i][0])
                    transformed_col_dims.append([counter, self.cols_mapping[key][i][1]])
                    counter += self.cols_mapping[key][i][1]
        tranposed_data = list(map(list, zip(*transformed_data_arrays)))
        transformed_data = [sum(inner, []) for inner in tranposed_data] # flatten arrays for each row

        self.transformed_col_names = transformed_col_names # for inverse
        # self.transformed_data_arrays = transformed_data_arrays
        self.transformed_col_dims = transformed_col_dims # for inverse
        return transformed_data
    
    def inverse_transform(self,generated_data):
        generated_data_arrays = []
        generated_data = np.array(generated_data)
        for elem in self.transformed_col_dims:
            generated_data_arrays.append(generated_data[:, elem[0]:elem[0]+elem[1]])
        df_inverse = pd.DataFrame()
        for elem in self.transformed_col_names:
            current = generated_data_arrays[self.transformed_col_names.index(elem)]
            if self.col_types[elem]=="general":
                u = np.array(current).flatten()
                u = (u + 1) / 2
                u = np.clip(u, 0, 1)
                min_v = self.gen_min_max["min"][elem]
                max_v = self.gen_min_max["max"][elem]
                u = u * (max_v - min_v) + min_v
                df_inverse[elem] = u
            
            elif self.col_types[elem]=="continuous":
                n_clusters = self.clusters_numbers[elem]
                current = np.array(current)
                u = np.clip(current[:, 0], -1, 1)
                v_t = np.ones((len(u), n_clusters)) * -100
                v=current[:, 1:]
                v_t[:, self.models_cont_mixed["components"][elem]] = v
                v = v_t

                p_argmax = np.argmax(v, axis=1)  # identify cluster index for each row
                means = self.models_cont_mixed["means"][elem]
                stds = self.models_cont_mixed["stds"][elem]
                std_t = stds.reshape([-1])[p_argmax]
                mean_t = means.reshape([-1])[p_argmax]
                tmp = u * 4 * std_t + mean_t

                # for idx, val in enumerate(tmp):
                #     if (val < info["min"]) | (val > info['max']):
                #         invalid_ids.append(idx)  # mark invalid data points

                tmp = np.round(tmp)
                df_inverse[elem] = tmp

            elif self.col_types[elem]=="mixed":
                n_clusters = self.clusters_numbers[elem]
                means = self.models_cont_mixed["means"][elem]
                stds = self.models_cont_mixed["stds"][elem]
                current = np.array(current)
                p_argmax = np.argmax(current[:,1:], axis=1)
                std_t = stds.reshape([-1])[p_argmax]
                mean_t = means.reshape([-1])[p_argmax]

                tmp = []
                for i in range(len(current)):
                    if current[i][0]==0:
                        col_modes = dict(self.mixed_modes[elem])
                        val = col_modes.get(p_argmax[i])
                        tmp.append(val) # get the mode from dict
                    else:
                        val = current[i][0] * 4 * std_t[i] + mean_t[i]
                        tmp.append(val)
                tmp = np.round(tmp)
                df_inverse[elem] = tmp
            elif self.col_types[elem]=="categorical":
                labels = self.categrical_labels[elem] #labels = list(self.categrical_labels.keys())[elem]
                idx = np.argmax(current, axis=1)
                df_inverse[elem] = [labels[i] for i in idx]
        return df_inverse