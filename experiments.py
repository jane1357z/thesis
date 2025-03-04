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
    categorical_cols = ['sex', 'children', 'smoker', 'region']
    general_cols = ['age', 'bmi', 'charges']
    continuous_cols = []
    mixed_cols = []
    components_numbers = {}
    mixed_modes = {}

    class_balance = {"smoker": [0.5,0.5]} # constraint
    condition_list = [{"col1":"smoker", "cat1": "no", "col2": "children", "cat2": 0}]
    cond_ratio = 0.5

    target_col = "charges"
    data_name = "insurance"
    task = "regr" # "regr" "class"
    return categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task

df_insurance = pd.read_csv("data\\insurance.csv")

df_insurance.reset_index(drop=True, inplace=True)  # !!!

train_data = df_insurance.copy()

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

with open(f'results/{data_name}/model_eval/metrics.txt', "w") as f:
    f.write(f"{cases[0]}:\n")

## experiments
print(cases[0])
time_start = time.perf_counter()
model = Synthesizer()
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
synth_data.to_csv(f"results/{data_name}/synth_data/actual.csv", sep=',', index=False)


evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col)
# # act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[0])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n\n")
    f.write(f"{cases[1]}:\n")
    

del model
del evaluation_data
categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_insurance.copy()

print(cases[1])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    class_balance=class_balance)

synth_data = model.sample(200)
synth_data.to_csv(f"results/{data_name}/synth_data/constr.csv", sep=',', index=False)

evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=class_balance)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[1])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n\n")
    f.write("Cond:\n")

del model
del evaluation_data
categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_insurance.copy()

print(cases[2])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    condition_list=condition_list,
    cond_ratio=cond_ratio)

synth_data = model.sample(200)
synth_data.to_csv(f"results/{data_name}/synth_data/cond.csv", sep=',', index=False)

evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, condition_list=condition_list, cond_ratio=cond_ratio)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[2])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n\n")
    f.write("Constr cond:\n")

# del model
# del evaluation_data
# categorical_cols, general_cols, continuous_cols, mixed_cols, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
# train_data = df_insurance.copy()

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

df_metrics = df_metrics.replace(pd.NA, np.nan).round(3)
df_ml_utility = df_ml_utility.replace(pd.NA, np.nan).round(3)

with pd.ExcelWriter("results/evaluation.xlsx", engine="openpyxl", mode="w") as writer: # for the 1st time, creates file
    df_metrics.to_excel(writer, sheet_name=f"{data_name}", startrow=0, startcol=0)
    df_ml_utility.to_excel(writer, sheet_name=f"{data_name}", startrow=6, startcol=0)








# with pd.ExcelWriter("check_loop.xlsx", engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer: # adds sheets in existing file
#     df_metrics.to_excel(writer, sheet_name=f"{data_name}", startrow=0, startcol=0)
#     df_ml_utility.to_excel(writer, sheet_name=f"{data_name}", startrow=6, startcol=0)




# my_data = pd.read_csv("synth_data.csv",sep=',')
# print(my_data)