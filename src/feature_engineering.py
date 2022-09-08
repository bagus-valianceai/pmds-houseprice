import pandas as pd
import numpy as np 
import joblib

def clean(df):
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
    }, inplace=True,
    )
    return df
    
def mathematical_transforms(df):
    df["_OQuGLA"] = df.OverallQual * df.GrLivArea
    return df

def main(x, state):
    df = x.copy()
    if (state=="predict"):
        df_transform = mathematical_transforms(df)
    else:
        df_clean = clean(df)
        df_transform = mathematical_transforms(df_clean)
    return df_transform
