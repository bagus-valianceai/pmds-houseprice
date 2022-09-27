from fastapi import FastAPI, Header, HTTPException
import starlette.responses as response
from pydantic import BaseModel
from test import api_predict
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
    return response.RedirectResponse("/redoc")

@app.post("/predict_v1/")
def root(item: api_data, deploy_token: str = Header()):
    if deploy_token != "pacmannpmds" :
        raise HTTPException(status_code = 401, detail = "Invalid token.")


    data_predict = dict()

    for i, value in enumerate(item):
        data_predict[value[0]] = value[1]

    res = api_predict(data_predict, main_model, model_name)
    print(res[0])
    return {"res" : res[0]}