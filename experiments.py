from model import Synthesizer
from blocks.evaluation import Data_evaluation
import pandas as pd
import numpy as np
import time



def param_search():
    pass

def experiments():
    pass

def reset_user_info():
    categorical_cols = ['Family', 'Education', 'Personal Loan', 'Securities Account', 'CD Account', 'Online', 'CreditCard']
    general_cols = ["Age", "Experience", "CCAvg", "Income", "ZIP Code"]
    continuous_cols = []
    mixed_cols = ["Mortgage"]
    components_numbers = {"Mortgage": 2}
    mixed_modes = {"Mortgage": [0]}


    class_balance = {"CreditCard": [0.6,0.4]} # constraint
    condition_list = [{"col1":"Family", "cat1": 2, "col2": "Education", "cat2":2}]
    cond_ratio = 0.15

    target_col = 'Personal Loan'
    data_name = "loan"
    task = "class" # "regr"
    return categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task

df_loan = pd.read_excel("data\\Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)
df_loan = df_loan.drop("ID", axis=1)
train_data = df_loan.reset_index(drop=True)  # !!!

def add_to_df(lst_metrics, lst_ml_utility, row_case):
    df_metrics.loc[row_case] = lst_metrics
    df_ml_utility.loc[row_case] = lst_ml_utility
    return df_metrics, df_ml_utility

cases=["Actual", "Constraints", "Conditions", "Constr & Cond"]

df_metrics = pd.DataFrame(columns=["JSD", "WD", "Corr.coef", "Theil.coef", "Corr.ratio", "Log-cluster"], index=cases)
df_metrics.index.name = "Statistical\nsimilarity"


categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()

if task == "class":
    ml_task = "(classification)"
    ml_columns = ["DT.Acc", "DT.AUC", "DT.f1 score", "MLP.Acc", "MLP.AUC", "MLP.f1 score"]
elif task == "regr":
    ml_task = "(regression)"
    ml_columns = ["DT.MAPE", "DT.EVS", "DT.R2 score", "MLP.MAPE", "MLP.EVS", "MLP.R2 score"]

df_ml_utility = pd.DataFrame(columns=ml_columns, index=["Actual", "Constraints", "Conditions", "Constr & Cond"])
df_ml_utility.index.name = f"ML utility\n{ml_task}"



def model_test(df_metrics, df_ml_utility, epochs, step_size_g, gamma_g, lr_g, step_size_d, gamma_d, lr_d, i_exp):
    categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
    model = Synthesizer(epochs=epochs,
                    step_size_g=step_size_g,
                    gamma_g=gamma_g, 
                    lr_g=lr_g,
                    step_size_d=step_size_d,
                    gamma_d=gamma_d, 
                    lr_d=lr_d,
                    i_exp=i_exp)
    model.fit(data_name=data_name,
        raw_data=train_data,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        mixed_cols=mixed_cols,
        general_cols=general_cols,
        components_numbers=components_numbers,
        mixed_modes=mixed_modes,
        target_col=target_col)

    synth_data = model.sample(1000)

    evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col)
    # # act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
    lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

    df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[0])


    del model
    return df_metrics, df_ml_utility

## experiments
# gamma_d_lst = [0.4,0.5,0.6,0.7,0.8,0.9]
# step_size_g_lst = [20,30,40,50]
lr_g_lst = [5e-3, 2e-3, 2e-4, 2e-5]
for i in range(len(lr_g_lst)):
    print(i)
    step_size_g=100
    gamma_g=0.8
    lr_g=lr_g_lst[i]
    lr_d=2e-4
    step_size_d=10
    gamma_d=0.8
    i_exp=i
    epochs=500
    df_metrics, df_ml_utility= model_test(df_metrics, df_ml_utility, epochs, step_size_g, gamma_g, lr_g, step_size_d, gamma_d, lr_d, i_exp)

    df_metrics = df_metrics.replace(pd.NA, np.nan).round(3)
    df_ml_utility = df_ml_utility.replace(pd.NA, np.nan).round(3)

    if i_exp==0:
        with pd.ExcelWriter("results/evaluation.xlsx", engine="openpyxl", mode="w") as writer: # for the 1st time, creates file
            df_metrics.to_excel(writer, sheet_name=f"{i_exp}", startrow=0, startcol=0)
            df_ml_utility.to_excel(writer, sheet_name=f"{i_exp}", startrow=6, startcol=0)
    else:
        with pd.ExcelWriter("results/evaluation.xlsx", engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer: # adds sheets in existing file
            df_metrics.to_excel(writer, sheet_name=f"{i_exp}", startrow=0, startcol=0)
            df_ml_utility.to_excel(writer, sheet_name=f"{i_exp}", startrow=6, startcol=0)



# with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
#     f.write(f"Experiment time: {exp_time/60} mins\n\n")
#     f.write(f"{cases[1]}:\n")
    
    
# print(cases[1])
# time_start = time.perf_counter()
# model = Synthesizer()
# model.fit(data_name=data_name,
#     raw_data=train_data,
#     categorical_cols=categorical_cols,
#     continuous_cols=continuous_cols,
#     mixed_cols=mixed_cols,
#     general_cols=general_cols,
#     components_numbers=components_numbers,
#     mixed_modes=mixed_modes,
#     target_col=target_col,
#     class_balance=class_balance)

# synth_data = model.sample(200)
# synth_data.to_csv(f"results/{data_name}/synth_data/constr.csv", sep=',', index=False)

# evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=class_balance)
# # act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
# lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

# time_end = time.perf_counter()
# exp_time = time_end - time_start

# df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[1])

# with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
#     f.write(f"Experiment time: {exp_time/60} mins\n\n")
#     f.write("Cond:\n")

# del model
# categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()

# print(cases[2])
# time_start = time.perf_counter()
# model = Synthesizer()
# model.fit(data_name=data_name,
#     raw_data=train_data,
#     categorical_cols=categorical_cols,
#     continuous_cols=continuous_cols,
#     mixed_cols=mixed_cols,
#     general_cols=general_cols,
#     components_numbers=components_numbers,
#     mixed_modes=mixed_modes,
#     target_col=target_col,
#     condition_list=condition_list,
#     cond_ratio=cond_ratio)

# synth_data = model.sample(200)
# synth_data.to_csv(f"results/{data_name}/synth_data/cond.csv", sep=',', index=False)

# evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, condition_list=condition_list, cond_ratio=cond_ratio)
# # act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
# lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

# time_end = time.perf_counter()
# exp_time = time_end - time_start

# df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[2])

# with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
#     f.write(f"Experiment time: {exp_time/60} mins\n\n")
#     f.write("Constr cond:\n")

# del model
# categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()

# print(cases[3])
# time_start = time.perf_counter()

# model = Synthesizer()
# model.fit(data_name=data_name,
#     raw_data=train_data,
#     categorical_cols=categorical_cols,
#     continuous_cols=continuous_cols,
#     mixed_cols=mixed_cols,
#     general_cols=general_cols,
#     components_numbers=components_numbers,
#     mixed_modes=mixed_modes,
#     target_col=target_col,
#     class_balance=class_balance,
#     condition_list=condition_list,
#     cond_ratio=cond_ratio)

# synth_data = model.sample(200)
# synth_data.to_csv(f"results/{data_name}/synth_data/constr_cond.csv", sep=',', index=False)

# evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, class_balance, condition_list, cond_ratio)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
# lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

# df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[3])

# time_end = time.perf_counter()
# exp_time = time_end - time_start

# with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
#     f.write(f"Experiment time: {exp_time/60} mins \n\n")












# my_data = pd.read_csv("synth_data.csv",sep=',')
# print(my_data)