from fastapi import FastAPI
from pydantic import BaseModel
from test import api_prerdict
import yaml
import joblib

f = open("params/params_2.yaml", "r")
params = yaml.load(f, Loader=yaml.SafeLoader)
f.close()

# load model name and model estimator with best param
model_name = joblib.load("../" + params['MODEL_NAME'])
main_model = joblib.load("../" + params['BEST_MODEL'])

class api_data(BaseModel):
    OverallQual : int
    GrLivArea : int
    TotalBsmtSF : int
    FirstFlrSF : int
    GarageCars : int
    GarageArea : int
    
app = FastAPI()

@app.get("/")
def root():
    return "Hello PMDS v1!"

@app.post("/predict_v1/")
def root(item: api_data):
    data_predict = dict()

    for i, value in enumerate(item):
        data_predict[value[0]] = value[1]

    res = api_prerdict(data_predict, main_model, model_name)
    print(res[0])
    return {"res" : res[0]}