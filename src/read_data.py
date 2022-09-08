import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


def read_data(path, 
              save_file = True,
              return_file = True):
    '''
    Read data from data folder in csv format.
    
    Parameters
    ----------
    path: str
          path to data
    
    '''
    
    data = pd.read_csv(path)
    
    if save_file:
        joblib.dump(data, "output/read_data/data_original.pkl")
    
    if return_file:
        return data

    
    
    
def split_input_output(dataset,
                       target_column,
                       save_file = True,
                       return_file = True):
    
    output_df = dataset[target_column]
    input_df = dataset.drop([target_column],
                            axis = 1)
    
    if save_file:
        joblib.dump(output_df, "output/read_data/output_df.pkl")
        joblib.dump(input_df, "output/read_data/input_df.pkl")
    
    if return_file:
        return output_df, input_df

    
def split_train_test(x, y, TEST_SIZE):
    # Do not forget to stratify if classification
    x_train, x_test,\
        y_train, y_test = train_test_split(x,
                                           y,
                                           test_size=TEST_SIZE,
                                           random_state=123)

    return x_train, x_test, y_train, y_test


def split_data(data_input, data_ouput, return_file=False, TEST_SIZE=0.2):

    x_train, x_test, \
        y_train, y_test = split_train_test(
            data_input,
            data_ouput,
            TEST_SIZE)

    x_train, x_valid, \
        y_train, y_valid = split_train_test(
            x_train,
            y_train,
            TEST_SIZE)

    joblib.dump(x_train, "output/read_data/x_train_v1.1.pkl")
    joblib.dump(y_train, "output/read_data/y_train_v1.1.pkl")
    joblib.dump(x_valid, "output/read_data/x_valid_v1.1.pkl")
    joblib.dump(y_valid, "output/read_data/y_valid_v1.1.pkl")
    joblib.dump(x_test, "output/read_data/x_test_v1.1.pkl")
    joblib.dump(y_test, "output/read_data/y_test_v1.1.pkl")

    if return_file:
        return x_train, y_train, \
            x_valid, y_valid, \
            x_test, y_test
