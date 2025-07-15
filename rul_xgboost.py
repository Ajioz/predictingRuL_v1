# ====================================
# 📦 Step 1: Imports
# ====================================
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
import shap          


sns.set(style='whitegrid')

# Set path to your train file (adjust as needed)
DATA_TRAIN_PATH = "data/train_FD001.txt"

# NASA C-MAPSS column structure
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
cols = ["unit", "time"] + op_cols + sensor_cols

# Load dataset
sensor_df = pd.read_csv(DATA_TRAIN_PATH, sep="\s+", header=None, names=cols)

# Compute RUL from max cycle
rul_per_unit = sensor_df.groupby("unit")["time"].transform("max")
sensor_df["RUL"] = rul_per_unit - sensor_df["time"]

print("✅ Data loaded and RUL computed:")
print(sensor_df.head())

# ====================================
# 📊 Step 2: Exploratory Data Analysis
# ====================================
# Assume `sensor_df` already exists and includes ['unit', 'time', 'op_setting_*', 'sensor_*', 'RUL']

# Display basic info
print("📊 Dataset Shape:", sensor_df.shape)
print("🧾 Columns:", sensor_df.columns.tolist())
print("\n📌 Sample Data:")
print(sensor_df.head())

# Sensor and operational columns
sensor_cols = [col for col in sensor_df.columns if col.startswith("sensor_")]
op_cols = [col for col in sensor_df.columns if col.startswith("op_setting")]

# Plot sensor distributions
plt.figure(figsize=(15, 8))
for i, sensor in enumerate(sensor_cols[:5]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(sensor_df[sensor], bins=50, kde=True)
    plt.title(f'Distribution of {sensor}')
plt.tight_layout()
plt.show()

# RUL vs. Time for a few engines
plt.figure(figsize=(12, 6))
sample_units = sensor_df['unit'].unique()[:5]
for unit_id in sample_units:
    unit_data = sensor_df[sensor_df['unit'] == unit_id]
    plt.plot(unit_data['time'], unit_data['RUL'], label=f'Unit {unit_id}')
plt.xlabel('Cycle')
plt.ylabel('RUL')
plt.title('RUL Degradation over Time (Sample Units)')
plt.legend()
plt.grid(True)
plt.show()

# Sensor trends over time
sensor_to_plot = 'sensor_2'
plt.figure(figsize=(12, 6))
for unit_id in sample_units:
    unit_data = sensor_df[sensor_df['unit'] == unit_id]
    plt.plot(unit_data['time'], unit_data[sensor_to_plot], label=f'Unit {unit_id}')
plt.xlabel('Cycle')
plt.ylabel(sensor_to_plot)
plt.title(f'{sensor_to_plot} over Time (Sample Units)')
plt.legend()
plt.grid(True)
plt.show()

# ====================================
# 🔍 Step 3: Feature Selection
# ====================================
# Correlation Heatmap
sensor_corr_df = sensor_df[sensor_cols + ['RUL']].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(sensor_corr_df, cmap='coolwarm', annot=False, cbar=True)
plt.title("🔍 Sensor & RUL Correlation Heatmap")
plt.tight_layout()
plt.show()

# Drop low-variance (flat) sensors
var_thresh = 1e-4
selector = VarianceThreshold(threshold=var_thresh)
selector.fit(sensor_df[sensor_cols])
flat_mask = selector.get_support()
filtered_sensor_cols = list(np.array(sensor_cols)[flat_mask])
flat_dropped = list(set(sensor_cols) - set(filtered_sensor_cols))
print(f"🛑 Dropped {len(flat_dropped)} flat sensors:", flat_dropped)

# Drop low-correlation (irrelevant) sensors
rul_corr = sensor_corr_df['RUL'].drop('RUL')
weak_corr_sensors = rul_corr[rul_corr.abs() < 0.05].index.tolist()
print(f"⚠️ Dropped {len(weak_corr_sensors)} low-correlation sensors:", weak_corr_sensors)

# Final feature list
final_sensor_cols = [col for col in filtered_sensor_cols if col not in weak_corr_sensors]
final_feature_cols = op_cols + final_sensor_cols
print(f"✅ Final features used for modeling: {len(final_feature_cols)}")

# ====================================
# 🧠 Step 4: Model Training + Evaluation
# ====================================
# Take last cycle per unit as prediction point
latest_df = sensor_df.groupby('unit').last().reset_index()
X = latest_df[final_feature_cols]
y = latest_df['RUL']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predict
rf_preds = rf.predict(X_test)
xgb_preds = xgb.predict(X_test)

# Evaluate
def evaluate(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }

rf_results = evaluate(y_test, rf_preds)
xgb_results = evaluate(y_test, xgb_preds)

print("\n📊 Random Forest Results:", rf_results)
print("📊 XGBoost Results:", xgb_results)

# ====================================
# 📈 Step 5: Plot Predictions
# ====================================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, rf_preds, alpha=0.7, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest: True vs. Predicted RUL')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')

plt.subplot(1, 2, 2)
plt.scatter(y_test, xgb_preds, alpha=0.7, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('XGBoost: True vs. Predicted RUL')
plt.xlabel('True RUL')
plt.ylabel('Predicted RUL')

plt.tight_layout()
plt.show()


# ====================================
# 🧮 Step 6: SHAP Explanation for XGBoost
# ====================================
print("\n🔍  SHAP Explanations for XGBoost")

# 6‑1 ▸ Build a SHAP explainer on the training set
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train)

# 6‑2 ▸ Global feature‑importance summary (beeswarm)
plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)   # bar chart
plt.title("XGBoost – Top Features (SHAP Importance)")
plt.tight_layout()
plt.show()



# 6‑3 ▸ Detailed beeswarm (density) plot
shap.summary_plot(shap_values, X_train, show=True)   # opens interactive in notebooks

# 6‑4 ▸ Optional: Explain a single prediction
idx = 0                                               # first test sample
shap.force_plot(
    explainer.expected_value,
    explainer.shap_values(X_test.iloc[idx]),
    X_test.iloc[idx],
    matplotlib=True
)