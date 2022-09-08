import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics


def read_data(params):
    x_train = joblib.load(params['DUMP_TRAIN'])
    y_train = joblib.load(params['Y_PATH_TRAIN'])
    x_valid = joblib.load(params['DUMP_VALID'])
    y_valid = joblib.load(params['Y_PATH_VALID'])

    return x_train, y_train, x_valid, y_valid


def model_lasso():
    param_dist = {'alpha': np.random.uniform(0.01,1,3)}
    base_model = Lasso(random_state=42, selection='random')
    return param_dist, base_model


def model_rf():
    param_dist = {"n_estimators": [100, 250, 500, 1000]}
    base_model = RandomForestRegressor(random_state=0, n_jobs=-1)
    return param_dist, base_model


def model_svr():
    param_dist = {'C': [0.25, 0.5, 1, 1.25]}
    base_model = LinearSVR(loss = 'squared_epsilon_insensitive', dual=False, max_iter=10000)
    return param_dist, base_model


def random_search_cv(model, param, scoring, n_iter, x, y, verbosity=0):
    random_fit = RandomizedSearchCV(estimator=model,
                                    param_distributions=param,
                                    scoring=scoring,
                                    n_iter=n_iter,
                                    cv=5,
                                    random_state=0,
                                    verbose=verbosity, refit=scoring[0])
    random_fit.fit(x, y)
    return random_fit


def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    mape = metrics.mean_absolute_percentage_error(true, predicted)
    exp_var = metrics.explained_variance_score(true, predicted)
    return mae, mse, rmse, r2_square, mape, exp_var


def fit(x_train, y_train, model, model_param, general_params):
    """
    Fit model

    Args:
        - model(callable): Sklearn / imblearn model
        - model_param(dict): sklearn's RandomizedSearchCV param_distribution
        - general_params(dict):x general parameters for the function
            - target(str) : y column to be used   
            - scoring(str) : sklearn cross-val scoring scheme
            - n_iter_search : RandomizedSearchCV number of iteration
    """
    #print( general_params['scoring'])

    model_fitted = random_search_cv(model, model_param,
                                    general_params['scoring'],
                                    general_params['n_iter_search'],
                                    x_train, y_train,
                                    general_params['verbosity'])

    
    print(
        f'Model: {model_fitted.best_estimator_}, {general_params["scoring"][0]}: {model_fitted.best_score_}')

    return model_fitted, model_fitted.best_estimator_


def validation_score(x_valid, y_valid, model_fitted):
    
    # Report default
    y_predicted = model_fitted.predict(x_valid)
    mae, mse, rmse, r2_square, mape, exp_var = evaluate(y_valid, y_predicted)
    score = {'mae':mae, 'mse':mse, 'rmse':rmse, 'r2': r2_square, 'mape': mape, 'exp_var': exp_var}

    return score

def select_model(train_log_dict):
    temp = []
    for score in train_log_dict['model_score']:
        temp.append(score['rmse'])
    best_model = train_log_dict['model_fit'][temp.index(min(temp))]
    best_parameter = train_log_dict['model_report'][temp.index(min(temp))]
    best_report = train_log_dict['model_score'][temp.index(min(temp))]
    
    return best_model, best_parameter, best_report