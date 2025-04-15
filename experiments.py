from model import Synthesizer
from blocks.evaluation import Data_evaluation
import pandas as pd
import numpy as np
import time
from ucimlrepo import fetch_ucirepo 


def get_data():

    # df_loan = pd.read_excel("data\\Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)
    # df_loan = df_loan.drop("ID", axis=1)
    # df_loan.loc[(df_loan["ZIP Code"] < 80000)]
    # df_loan.reset_index(drop=True, inplace=True)
    # df_init = df_loan.copy()

    # df_insurance = pd.read_csv("data\\insurance.csv")
    # df_insurance.reset_index(drop=True, inplace=True)
    # df_init = df_insurance.copy()

    # df_house = pd.read_csv("data\\kc_house_data.csv")
    # df_house = df_house.drop("id", axis=1)
    # df_house = df_house.drop("date", axis=1)
    # df_house.reset_index(drop=True, inplace=True)
    # df_init = df_house.copy()

    adult = fetch_ucirepo(id=2) 
    x = adult.data.features
    y = adult.data.targets
    df_adult = pd.concat([x, y], axis=1)  
    df_adult = df_adult.drop("education-num", axis=1)
    df_adult.dropna(inplace=True)
    df_adult.reset_index(drop=True, inplace=True)
    df_init = df_adult.copy()

    return df_init

def add_to_df(lst_metrics, lst_ml_utility, row_case):
    df_metrics.loc[row_case] = lst_metrics
    df_ml_utility.loc[row_case] = lst_ml_utility
    return df_metrics, df_ml_utility

def reset_user_info():
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
    general_cols = ['age']
    continuous_cols = ['fnlwgt']
    mixed_cols = ['capital-gain','capital-loss','hours-per-week']
    components_numbers = {'fnlwgt': 7,'capital-gain': 12, 'capital-loss': 2, 'hours-per-week':15}
    mixed_modes = {'capital-gain': [0], 'capital-loss': [0],'hours-per-week': [40]}
    log_transf = ['age','fnlwgt']

    class_balance = {"race": [0.6,0.2,0.1,0.05,0.05]} # constraint
    condition_list = [{"col1":"workclass", "cat1": "Private", "col2": "relationship", "cat2": "Wife"}] # condition
    cond_ratio = 0.2

    target_col = "income"
    data_name = "adult"
    task = "class" # "regr" "class"

    return categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task

def initalize_dfs():
    cases=["Actual", "Constraints", "Constr cv", "Constr loss", "Conditions", "Constr & Cond"]

    df_metrics = pd.DataFrame(columns=["JSD", "WD", "Corr.coef", "Theil.coef", "Corr.ratio", "Log-cluster"], index=cases)
    df_metrics.index.name = "Statistical\nsimilarity"


    _, _, _, _, _, _, _, _, _, _, _, _, task = reset_user_info()

    if task == "class":
        ml_task = "(classification)"
        ml_columns = ["DT.Acc", "DT.AUC", "DT.f1 score", "MLP.Acc", "MLP.AUC", "MLP.f1 score"]
    elif task == "regr":
        ml_task = "(regression)"
        ml_columns = ["DT.MAPE", "DT.EVS", "DT.R2 score", "MLP.MAPE", "MLP.EVS", "MLP.R2 score"]

    df_ml_utility = pd.DataFrame(columns=ml_columns, index=cases)
    df_ml_utility.index.name = f"ML utility\n{ml_task}"


    return df_metrics, df_ml_utility, cases

###########

df_metrics, df_ml_utility, cases = initalize_dfs()

df_init = get_data()

## experiments


model_cases = ['actual', 'constr', 'constr_cv', 'constr_loss', 'cond', 'constr_cond']

num_samples = 2000

###################################################################### case actual
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

with open(f'results/{data_name}/model_eval/metrics.txt', "w") as f:
    f.write(f"{cases[0]}:\n")

case_name = model_cases[0]
print(cases[0])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/actual.csv", sep=',', index=False)

print("Data evaluation")
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

###################################################################### case constr
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

case_name = model_cases[1]
print(df_metrics, df_ml_utility)
print(cases[1])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name,
    class_balance=class_balance)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/constr.csv", sep=',', index=False)

print("Data evaluation")
evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=class_balance)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[1])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n")
    f.write(f"add_diff: {add_diff}\n\n")
    f.write(f"{cases[2]}:\n")

del model
del evaluation_data

# ###################################################################### case constr_cv
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

case_name = model_cases[2]
print(df_metrics, df_ml_utility)
print(cases[2])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name,
    class_balance=class_balance)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/constr_cv.csv", sep=',', index=False)

print("Data evaluation")
evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=class_balance)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[2])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n")
    f.write(f"add_diff: {add_diff}\n\n")
    f.write(f"{cases[3]}:\n")

del model
del evaluation_data

# # ###################################################################### case constr_loss
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

case_name = model_cases[3]
print(df_metrics, df_ml_utility)
print(cases[3])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name,
    class_balance=class_balance)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/constr_loss.csv", sep=',', index=False)

print("Data evaluation")
evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance=class_balance)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[3])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins\n")
    f.write(f"add_diff: {add_diff}\n\n")
    f.write(f"{cases[4]}:\n")

del model
del evaluation_data

# ###################################################################### case cond
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

case_name = model_cases[4]
print(df_metrics, df_ml_utility)
print(cases[4])
time_start = time.perf_counter()
model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name,
    condition_list=condition_list,
    cond_ratio=cond_ratio)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/cond.csv", sep=',', index=False)

print("Data evaluation")
evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, condition_list=condition_list, cond_ratio=cond_ratio)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[4])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"{cases[4]}:\n")
    f.write(f"Experiment time: {exp_time/60} mins\n")
    f.write(f"add_diff: {add_diff}\n\n")
    f.write(f"{cases[5]}:\n")

del model
del evaluation_data

# ###################################################################### case constr_cond
categorical_cols, general_cols, continuous_cols, mixed_cols, log_transf, components_numbers, mixed_modes, class_balance, condition_list, cond_ratio, target_col, data_name, task = reset_user_info()
train_data = df_init.copy()

case_name = model_cases[5]
print(df_metrics, df_ml_utility)
print(cases[5])
time_start = time.perf_counter()

model = Synthesizer()
model.fit(data_name=data_name,
    raw_data=train_data,
    categorical_cols=categorical_cols,
    continuous_cols=continuous_cols,
    mixed_cols=mixed_cols,
    general_cols=general_cols,
    log_transf=log_transf,
    components_numbers=components_numbers,
    mixed_modes=mixed_modes,
    target_col=target_col,
    case_name=case_name,
    class_balance=class_balance,
    condition_list=condition_list,
    cond_ratio=cond_ratio)

synth_data = model.sample(num_samples)
synth_data.to_csv(f"results/{data_name}/synth_data/constr_cond.csv", sep=',', index=False)

print("Data evaluation")
evaluation_data = Data_evaluation(data_name, train_data, synth_data, categorical_cols, general_cols, continuous_cols, mixed_cols, mixed_modes, task, target_col, class_balance, condition_list, cond_ratio)
# act - compare with actual data, without - compare with conditioned or constrined data, add_diff - class_balance/cond_ratio
lst_metrics, lst_ml_utility, add_diff = evaluation_data.evaluate_data()

time_end = time.perf_counter()
exp_time = time_end - time_start

df_metrics, df_ml_utility = add_to_df(lst_metrics, lst_ml_utility, cases[5])

with open(f'results/{data_name}/model_eval/metrics.txt', "a") as f:
    f.write(f"Experiment time: {exp_time/60} mins \n")
    f.write(f"add_diff: {add_diff}")
######################################################################

print(df_metrics, df_ml_utility)

df_metrics = df_metrics.replace(pd.NA, np.nan).round(3)
df_ml_utility = df_ml_utility.replace(pd.NA, np.nan).round(3)


with pd.ExcelWriter("results/evaluation.xlsx", engine="openpyxl", mode="w") as writer:
    df_metrics.to_excel(writer, sheet_name=f"{data_name}", startrow=0, startcol=0)
    df_ml_utility.to_excel(writer, sheet_name=f"{data_name}", startrow=8, startcol=0)

