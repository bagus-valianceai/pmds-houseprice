import numpy as np
import pandas as pd
import model_lib
import joblib
import yaml

def read_data_test(params):
    x_test = joblib.load(params['DUMP_TEST'])
    y_test = joblib.load(params['Y_PATH_TEST'])

    return x_test, y_test

def main(params):
    model_name = joblib.load(params['MODEL_NAME'])
    print(f"Working on test data with {model_name} model")
    
    main_model = joblib.load(params['BEST_MODEL'])
    
    test_log_dict = {'model': [main_model],
                  'model_name': [model_name],
                  'model_score': []}

    x_test, y_test  = read_data_test(params)
    score = model_lib.validation_score(x_test, y_test, main_model)
    test_log_dict['model_score'].append(score)
    joblib.dump(test_log_dict, 'output/model/test/test_log.pkl')
    print(
        f"Model: {test_log_dict['model_name']},\n Score: {test_log_dict['model_score']},\n Model's parameter: {test_log_dict['model']}")
    