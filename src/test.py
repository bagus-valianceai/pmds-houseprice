import joblib
import yaml
import os
import pandas as pd
import numpy as np
import preprocess_data as prep
from feature_engineering import main as add_feature

# Open yaml
f = open("src/params/params_2.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

# model_name = joblib.load(params['MODEL_NAME'])
# main_model = joblib.load(params['BEST_MODEL'])

# def test_1():
#     x = 5
#     assert(x == 5)

def set_dtypes(data_input, params):
    '''
    Check data input datatypes consistency with predefined DTYPES
    Set data datatypes as DTYPE
    
    Parameters
    ----------
    data_input: pd.DataFrame
        DaraFrame for modeling
    
    Returns
    -------
    data: pd.DataFrame
        Checked dataset for columns consistency
    '''
    data = data_input.astype(params["PREDICT_COLUMN_TYPE"])
    return data

def construct_df(params, data_to_predict, file=None):
    if file == 'csv':
        df_to_predict = pd.read_csv(data_to_predict, sep = ';')
    elif file == 'excel':
        df_to_predict = pd.read_excel(data_to_predict)
    else:
        df_to_predict = pd.DataFrame(data=data_to_predict)
        df_to_predict = set_dtypes(df_to_predict, params)
        COLUMN = set(params['PREDICT_COLUMN'])
        column_in_data = set(df_to_predict.columns)
        remain_columns = list(COLUMN-column_in_data)
        df_to_predict[remain_columns] = np.NaN
    return df_to_predict

def feature_engineering_predict(data_to_predict):
    # this process is similah with feature engineering in training stage
    state = 'transform'
    dump_path = params[f'DUMP_PREDICT']
    data_to_predict = data_to_predict.copy()
    house_numerical = data_to_predict[params['PREDICT_COLUMN']]
    df_add_feature = add_feature(house_numerical, state="predict")
    df_numerical_imputed = prep.numerical_imputer(df_add_feature, state=state)
    x_predict = prep.normalization(df_add_feature, state=state)
    #joblib.dump(x_predict, dump_path)
    return x_predict

def test_predict():
    # load model name and model estimator with best param
    model_name = joblib.load(params['MODEL_NAME'])
    main_model = joblib.load(params['BEST_MODEL'])

    # construct dictionary as log file
    predict_dict = {
    'model': [main_model],
    'model_name': [model_name],
    'predicted': []
    }

    data_predict = [{
    "OverallQual" : 7,
    "GrLivArea" : 1710,
    "TotalBsmtSF" : 856,
    "FirstFlrSF" : 1,
    "GarageCars" : 2,
    "GarageArea" : 548
    }]

    # Make input data to DataFrame
    x_input = construct_df(params, data_predict)

    # Feature engineering on input DataFrame
    x_predict = feature_engineering_predict(x_input)

    # Make prediction
    y_predicted = main_model.predict(x_predict)

    # Dump log prediction result
    predict_dict['predicted'].append(y_predicted)

    # Show the result of price prediction
    print(f"Model: {predict_dict['model_name']},\n Predicted: {predict_dict['predicted']}\n")

    assert(205000 < y_predicted < 206000)
