import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from feature_engineering import main as add_feature
import yaml


def numerical_imputer(numerical,
                    state = 'transform'):
    
    index = numerical.index
    cols = numerical.columns
    
    if state == 'fit':
        imputer = SimpleImputer(
            missing_values=np.nan,
            strategy="mean")

        imputer.fit(numerical)
        joblib.dump(imputer,
                    "output/preprocess_data/estimator/numerical_imputer.pkl")
    elif state == 'transform':
        imputer = joblib.load("output/preprocess_data/estimator/numerical_imputer.pkl")
        
    imputed = imputer.transform(numerical)
    imputed = pd.DataFrame(imputed)
    imputed.index = index
    imputed.columns = cols
    return imputed


def categorical_imputer(df_categorical):
    df = df_categorical.copy()
    df.fillna(value = 'KOSONG', inplace=True)
    return df


def one_hot_encoder(x_cat,
                    state='fit'):
    df = x_cat.copy()
    index = x_cat.index
    col = x_cat.columns
    
    if state == 'fit':
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoder.fit(x_cat)
        joblib.dump(encoder,
                    "output/preprocess_data/estimator/onehotencoder.pkl")
        
    elif state == 'transform':
        encoder = joblib.load("output/preprocess_data/estimator/onehotencoder.pkl")
    
    encoded = encoder.transform(x_cat)
    feat_names = encoder.get_feature_names_out(col)
    encoded = pd.DataFrame(encoded)
    encoded.index = index
    encoded.columns = feat_names
    return encoded


def normalization(x_all,
                  state = 'fit'):
    index = x_all.index
    cols = x_all.columns
    

    if state == 'fit':
        normalizer = StandardScaler()
        normalizer.fit(x_all)
        joblib.dump(normalizer,
                    "output/preprocess_data/estimator/normalizer.pkl")

    elif state == 'transform':
        normalizer = joblib.load("output/preprocess_data/estimator/normalizer.pkl")
        
    normalized = normalizer.transform(x_all)
    normalized = pd.DataFrame(normalized)
    normalized.index = index
    normalized.columns = cols
    return normalized

def run(params, xpath, ypath, dump_path, state='fit'):
    '''
    Main function of wrangling and feature engineering.
    This function will applied in data training, testing and validation.
    
    Parameters
    ----------
    params: .yaml file
        File containing necessary variables as constant variable such as location file and features name 
        - PREDICT_COLUMN(str) : list of features to be used   
    xpath: string
        Location of features pickle file

    ypath: string
        Location of target pickle file

    dump_path: string
        Location to save the result of preprocessing

    state: string
        Data state for leakage handling. fit for training data, transform for validation and testing data

    '''
    
    # Load variables and target pickle file
    house_variables = joblib.load(xpath)
    house_target = joblib.load(ypath)
    
    # Due to simplicity, we just use six features with highest correlation to target class
    house_numerical = house_variables[params['TRAIN_COLUMN']]
    
    # Add a representative feature
    df_add_feature = add_feature(house_numerical, state=state)
    
    # Handling missing value
    df_numerical_imputed = numerical_imputer(df_add_feature, state=state)
    
    # Normalization
    df_normalized = normalization(df_numerical_imputed, state=state)
    joblib.dump(df_normalized, dump_path)
