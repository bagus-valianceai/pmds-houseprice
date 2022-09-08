import numpy as np
import pandas as pd
import model_lib
import time
import joblib
import yaml
import preprocess_data as prep
from feature_engineering import main as add_feature


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
    joblib.dump(x_predict, dump_path)
    return x_predict

if __name__ == "__main__":
    '''
    Main function of prediction.
    1. Load the model with best parameters in pickle
    2. Get input data (manual user input)
    3. Construct DataFrame of input data (including defense in data type)
    4. Preprocessed and Feature engineering (load estimator in pickle file)
    5. Make prediction
    '''
    # Open yaml
    f = open("src/params/params.yaml", "r")
    params = yaml.load(f, Loader=yaml.SafeLoader)
    f.close()
    
    # load model name and model estimator with best param
    model_name = joblib.load(params['MODEL_NAME'])
    main_model = joblib.load(params['BEST_MODEL'])
    
    print(f"Working on predict data with {model_name} model\n")
    
    # construct dictionary as log file
    predict_dict = {'model': [main_model],
                  'model_name': [model_name],
                  'predicted': []}
    
    # input data to predict
    
    # through file
    # data_predict = 'data/2data.csv'
    # file = 'csv'
    
    # through input
    n_data = int(input(f"Input data (enter int value): "))
    data_predict = {}
    for i in range(n_data):
        for i in params["PREDICT_COLUMN"]:
            if i in data_predict:
                data_predict[i].append(input(f"Input {i}: "))
            else:
                data_predict[i] = [input(f"Input {i}: ")]
    
    # Make input data to DataFrame
    x_input = construct_df(params, data_predict)
    
    # Feature engineering on input DataFrame
    print(f"Running on feature engineering...\n")
    x_predict = feature_engineering_predict(x_input)
    
    # Make prediction
    print(f"Running on prediction...\n")
    y_predicted = main_model.predict(x_predict)
    
    # Dump log prediction result
    predict_dict['predicted'].append(y_predicted)
    joblib.dump(predict_dict, 'output/predict/predict_log.pkl')
    
    # Show the result of price prediction
    print(f"Model: {predict_dict['model_name']},\n Predicted: {predict_dict['predicted']}\n")
    
    for i in range(len(x_predict)):
        print(f"{i+1}. Data with overall quality : {x_input['OverallQual'][i]}, First Floor: {x_input['FirstFlrSF'][i]} square feet, were predict to have sale price {y_predicted[i]}\n")
    