import numpy as np
import pandas as pd
import joblib
import yaml
from read_data import read_data, split_input_output, split_data
import preprocess_data as ppr
import training
import testing

#Open yaml
f = open("src/params/params_2.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

#Splitting Data
print("Running on splitting...")
data_house = read_data(params['DATA_PATH'])
output_df, input_df = split_input_output(
                            data_house,
                            params['TARGET_COLUMN'])

X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(input_df,
                                                                output_df,
                                                                True,
                                                                params['TEST_SIZE'])

temp = ['TRAIN','VALID','TEST']

#Feature Engineering

for subgroup in temp:
    print(f"Running on feature engineering {subgroup}...")
    xpath = params[f'X_PATH_{subgroup}']
    ypath = params[f'Y_PATH_{subgroup}']
    dump_path = params[f'DUMP_{subgroup}']

    if subgroup == 'TRAIN':
        state = 'fit'
    else:
        state = 'transform'
    ppr.run(params, xpath, ypath, dump_path, state)

    
#Training and Tuning
print(f"Running on training and hyperparameter tuning...")
training.main(params)

#Predicting and Last Evaluation
print(f"Last evaluation on test data")
testing.main(params)