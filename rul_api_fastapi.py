# ======================================
# 🧠 RUL Prediction + Training Switcher API
# ======================================
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from fastapi.middleware.cors import CORSMiddleware
import logging

app = FastAPI(title="RUL Dynamic Training + Prediction API")

# Enable CORS for frontend (Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ========== 📥 Input Schema ==========
class SensorInput(BaseModel):
    features: dict
    model: Optional[str] = "xgb"

class BatchInput(BaseModel):
    batch: List[SensorInput]

# ========== 🔧 Helper Functions ==========
def compute_rul(df):
    df.columns = ["unit", "time"] + [f"op_setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
    df['RUL'] = df.groupby("unit")["time"].transform("max") - df["time"]
    return df

def select_features(df, sensor_cols):
    var_thresh = 1e-6
    selector = VarianceThreshold(threshold=var_thresh)
    selector.fit(df[sensor_cols])
    flat_mask = selector.get_support()
    filtered = list(np.array(sensor_cols)[flat_mask])
    corr = df[filtered + ['RUL']].corr()['RUL'].drop('RUL')
    strong = corr[abs(corr) >= 0.01].index.tolist()
    return [col for col in filtered if col in strong]

def train_models(df):
    op_cols = [col for col in df.columns if col.startswith("op_setting")]
    sensor_cols = [col for col in df.columns if col.startswith("sensor")]
    final_sensor_cols = select_features(df, sensor_cols)
    final_features = op_cols + final_sensor_cols
    X = df[final_features]
    y = df['RUL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    joblib.dump(rf, f"{MODELS_DIR}/rf_model.pkl")
    joblib.dump(xgb, f"{MODELS_DIR}/xgb_model.pkl")
    joblib.dump(final_features, f"{MODELS_DIR}/feature_columns.pkl")

    logging.info("✅ Models and features saved.")
    return final_features

# ========== 🔁 Upload New Training Data ==========
@app.post("/upload")
def upload_train_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, sep="\s+", header=None)
        df = compute_rul(df)
        train_models(df)
        return {"message": f"✅ Model trained and updated from file: {file.filename}"}
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")

# ========== 🔍 Predict ==========
@app.post("/predict")
def predict_rul(input_data: SensorInput):
    try:
        model_type = input_data.model.lower()
        features = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
        model = joblib.load(f"{MODELS_DIR}/{model_type}_model.pkl")

        input_df = pd.DataFrame([input_data.features])[features]
        pred = model.predict(input_df)[0]
        days = pred
        return {
            "rul_cycles": round(pred, 2),
            "rul_days": round(days, 2),
            "rul_months": round(days / 30, 2),
            "rul_years": round(days / 365, 3),
            "status": "☠️ FAILED" if days < 1 else "⚠️ DANGER" if days <= 90 else "🟢 OK",
            "model": model_type.upper()
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# ========== 📦 Batch Prediction ==========
@app.post("/predict/batch")
def batch_predict(batch_input: BatchInput):
    try:
        results = []
        for item in batch_input.batch:
            res = predict_rul(item)
            results.append(res)
        return {"predictions": results}
    except Exception as e:
        logging.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

# ========== 📊 Info Route ==========
@app.get("/info")
def model_info():
    try:
        features = joblib.load(f"{MODELS_DIR}/feature_columns.pkl")
        return {
            "model_versions": ["xgb_model.pkl", "rf_model.pkl"],
            "feature_count": len(features),
            "features": features
        }
    except:
        raise HTTPException(status_code=404, detail="No trained model found.")

# ========== 🐳 Docker Healthcheck Route ==========
@app.get("/")
def root():
    return {"message": "Welcome to the RUL Prediction API, RUL API is running!"}