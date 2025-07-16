# ============================================
# 🚀 RUL Prediction API (FastAPI + XGBoost)
# ============================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import uvicorn
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import List

# Load trained model and feature columns
xgb_model = joblib.load("models/xgb_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")

# Initialize FastAPI
app = FastAPI(title="RUL Prediction API")

# Define input schema
class SensorInput(BaseModel):
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

class PredictionOutput(BaseModel):
    rul_cycles: float
    rul_days: float
    rul_months: float
    rul_years: float
    status: str
    model: str

# Helper function for status

def get_status(days):
    if days < 1:
        return "☠️ FAILED"
    elif days <= 90:
        return "⚠️ DANGER"
    else:
        return "🟢 OK"

# Route to predict RUL using specified model
@app.post("/predict", response_model=PredictionOutput)
def predict_rul(data: SensorInput, model: str = "xgb"):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df[feature_cols]  # Align column order

    if model == "xgb":
        pred = xgb_model.predict(input_df)[0]
    elif model == "rf":
        pred = rf_model.predict(input_df)[0]
    else:
        raise HTTPException(status_code=400, detail="Model must be 'xgb' or 'rf'")

    cycle_to_day_ratio = 1
    rul_days = pred * cycle_to_day_ratio
    return PredictionOutput(
        rul_cycles=round(pred, 2),
        rul_days=round(rul_days, 2),
        rul_months=round(rul_days / 30, 2),
        rul_years=round(rul_days / 365, 3),
        status=get_status(rul_days),
        model=model.upper()
    )

# Run using: uvicorn rul_api:app --reload
if __name__ == "__main__":
    uvicorn.run("rul_api:app", host="0.0.0.0", port=8000, reload=True)
