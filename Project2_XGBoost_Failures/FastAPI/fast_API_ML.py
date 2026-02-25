import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Input
class RoboXGboostInput(BaseModel):
    Torque_Nm: float
    Rotational_speed_rpm: float 
    Tool_wear_min: int
    Temp_Diff: float

app = FastAPI(title="RoboXGBoost Failure Predictor")

try:
    model = joblib.load('robotics_xgb_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict") 
def predict_failure(input_data: RoboXGboostInput):

    input_array = np.array([[
        input_data.Torque_Nm, 
        input_data.Rotational_speed_rpm, 
        input_data.Tool_wear_min, 
        input_data.Temp_Diff
    ]])
    
    # predict
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)
    
    return {
        "result": "Failure" if int(prediction[0]) == 1 else "No Failure",
        "probability_of_failure": round(float(probability[0][1]), 4)
    }

@app.get("/")
def read_root():
    return {"message": "API is online. Go to /docs for testing."}