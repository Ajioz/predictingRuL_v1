import joblib
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns

sns.set(style='whitegrid')

def infer_column_schema(df: pd.DataFrame):
    """
    Infers the number of operational setting columns and sensor columns
    based on heuristic patterns:
    - First 2 columns: unit and time
    - Next N columns with relatively low variance => op-settings
    - Remaining columns => sensors
    """
    fixed_cols = ["unit", "time"]
    data = df.iloc[:, 2:]

    variances = data.var()
    threshold = 1e2

    # Assume op-settings are contiguous and start from the first column after "time"
    op_count = 0
    for i, v in enumerate(variances):
        if v < threshold:
            op_count += 1
        else:
            break  # Stop once sensor-level variance begins

    op_cols = [f"op_setting_{i}" for i in range(1, op_count + 1)]
    sensor_cols = [f"sensor_{i}" for i in range(1, data.shape[1] - op_count + 1)]

    return fixed_cols + op_cols + sensor_cols, op_cols, sensor_cols


def run_training_pipeline(dataset_id: str):
    print(f"\n🚀 Training pipeline for {dataset_id}")

    # === Step 1: Load Data ===
    def load_dataset(file_path):
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        cols, op_cols, sensor_cols = infer_column_schema(df)
        df.columns = cols
        return df, op_cols, sensor_cols


    train_fp = Path("data") / f"train_{dataset_id}.txt"
    test_fp = Path("data") / f"test_{dataset_id}.txt"
    train_df, op_cols, sensor_cols = load_dataset(train_fp)
    test_df, _, _ = load_dataset(test_fp)

    # Compute RUL
    train_df["RUL"] = train_df.groupby("unit")["time"].transform("max") - train_df["time"]
    test_df["RUL"] = test_df.groupby("unit")["time"].transform("max") - test_df["time"]

    # === Step 2: Feature Selection ===
    selector = VarianceThreshold(1e-6)
    selector.fit(train_df[sensor_cols])
    selected_sensors = list(np.array(sensor_cols)[selector.get_support()])

    corr = train_df[selected_sensors + ['RUL']].corr()
    weak_corr = corr['RUL'].abs() < 0.01
    weak_sensors = corr['RUL'][weak_corr].index.tolist()
    final_sensors = [s for s in selected_sensors if s not in weak_sensors]
    features = op_cols + final_sensors
    print(f"✅ Features selected: {len(features)}")

    # === Step 3: Train Models ===
    X = train_df[features]
    y = train_df["RUL"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42)

    print("🏋️ Training RandomForest and XGBoost...")
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    print("✅ Models trained.")

    # === Step 4: Evaluate Models ===
    X_test = test_df[features]
    y_test = test_df["RUL"]
    rf_preds = rf.predict(X_test)
    xgb_preds = xgb.predict(X_test)

    def evaluate(y_true, y_pred):
        return {
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }

    rf_eval = evaluate(y_test, rf_preds)
    xgb_eval = evaluate(y_test, xgb_preds)

    print("📊 Evaluation:")
    for k, v in rf_eval.items():
        print(f"   RandomForest {k}: {v:.4f}")
    for k, v in xgb_eval.items():
        print(f"   XGBoost      {k}: {v:.4f}")

    # === Step 5: Save Artifacts ===
    model_dir = Path("models")
    inference_dir = Path("inference")
    report_dir = Path("reports")
    for d in [model_dir, inference_dir, report_dir]:
        d.mkdir(exist_ok=True)

    joblib.dump(xgb, model_dir / f"{dataset_id.lower()}_xgb_model.pkl")
    joblib.dump(rf, model_dir / f"{dataset_id.lower()}_rf_model.pkl")
    joblib.dump(features, model_dir / f"{dataset_id.lower()}_feature_columns.pkl")

    metrics = {
        "RandomForest": rf_eval,
        "XGBoost": xgb_eval,
        "generated": datetime.now().isoformat()
    }
    with open(inference_dir / f"{dataset_id.lower()}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    predictions = pd.DataFrame({
        "True_RUL": y_test,
        "RF_Predicted_RUL": rf_preds,
        "XGB_Predicted_RUL": xgb_preds
    })
    predictions.to_csv(inference_dir / f"{dataset_id.lower()}_predictions.csv", index=False)

    # === Step 6: SHAP Visuals ===
    try:
        print("🔍 Generating SHAP explanations...")
        explainer = shap.Explainer(xgb)
        shap_values = explainer(X_train)

        # SHAP bar
        shap_bar = report_dir / f"{dataset_id.lower()}_shap_bar.png"
        shap.plots.bar(shap_values, max_display=10, show=False)
        plt.title(f"{dataset_id} SHAP Top Features")
        plt.savefig(shap_bar, bbox_inches="tight")
        plt.close()

        # SHAP beeswarm
        shap_swarm = report_dir / f"{dataset_id.lower()}_shap_beeswarm.png"
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        plt.title(f"{dataset_id} SHAP Beeswarm")
        plt.savefig(shap_swarm, bbox_inches="tight")
        plt.close()

        print(f"📈 SHAP plots saved: {shap_bar.name}, {shap_swarm.name}")

    except Exception as e:
        print(f"⚠️ SHAP generation skipped: {e}")

    return {
        "dataset": dataset_id,
        "rf_metrics": rf_eval,
        "xgb_metrics": xgb_eval,
        "features_used": features
    }

if __name__ == "__main__":
    for dataset in ["FD001", "FD002", "FD003", "FD004"]:
        run_training_pipeline(dataset)
