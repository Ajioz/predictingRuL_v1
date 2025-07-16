# ====================================
# 📦 Step 1: Imports
# ====================================
import os
from pathlib import Path
from datetime import datetime

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

# ====================================
# 📁 Step 2: Load and Prepare Data
# ====================================
data = 'train_FD002.txt'
DATA_TRAIN_PATH = f"data/{data}"

sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
cols = ["unit", "time"] + op_cols + sensor_cols

sensor_df = pd.read_csv(DATA_TRAIN_PATH, sep="\s+", header=None, names=cols)
rul_per_unit = sensor_df.groupby("unit")["time"].transform("max")
sensor_df["RUL"] = rul_per_unit - sensor_df["time"]

print("✅ Data loaded and RUL computed:")
print(sensor_df.head())

# ====================================
# 📊 Step 3: Exploratory Data Analysis
# ====================================
print("📊 Dataset Shape:", sensor_df.shape)
print("🧾 Columns:", sensor_df.columns.tolist())

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

# Correlation Heatmap
sensor_corr_df = sensor_df[sensor_cols + ['RUL']].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(sensor_corr_df, cmap='coolwarm', annot=False, cbar=True)
plt.title("🔍 Sensor & RUL Correlation Heatmap")
plt.tight_layout()
plt.show()

# ====================================
# 🔍 Step 4: Feature Selection
# ====================================
var_thresh = 1e-6
selector = VarianceThreshold(threshold=var_thresh)
selector.fit(sensor_df[sensor_cols])
flat_mask = selector.get_support()
filtered_sensor_cols = list(np.array(sensor_cols)[flat_mask])

rul_corr = sensor_corr_df['RUL'].drop('RUL')
weak_corr_sensors = rul_corr[rul_corr.abs() < 0.01].index.tolist()

final_sensor_cols = [col for col in filtered_sensor_cols if col not in weak_corr_sensors]
final_feature_cols = op_cols + final_sensor_cols

print(f"✅ Final features used for modeling: {len(final_feature_cols)}")

# ====================================
# 🧠 Step 5: Model Training + Evaluation
# ====================================
X = sensor_df[final_feature_cols]
y = sensor_df['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

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
# 🕒 Step 6: Human-Readable RUL + Report
# ====================================
cycle_to_day_ratio = 1
rul_days = xgb_preds * cycle_to_day_ratio
rul_months = rul_days / 30
rul_years = rul_days / 365

rul_summary = pd.DataFrame({
    'True_RUL (cycles)': y_test,
    'Predicted_RUL (cycles)': xgb_preds,
    'RUL_Days': rul_days.round(1),
    'RUL_Months': rul_months.round(2),
    'RUL_Years': rul_years.round(3)
})

conditions = [
    rul_summary['RUL_Days'] < 1,
    rul_summary['RUL_Days'] <= 90
]
choices = ['☠️ FAILED', '⚠️ DANGER']
rul_summary['Status'] = np.select(conditions, choices, default='🟢 OK')

print("\n🕒 Human-Readable RUL Predictions:")
print(rul_summary.head(10))

# Save report
os.makedirs("report", exist_ok=True)
report_lines = [
    "Remaining Useful Life (RUL) Prediction Summary",
    "=" * 50,
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "Top 10 RUL Predictions:",
    "-" * 50
]
for idx, row in rul_summary.head(10).iterrows():
    line = (f"Index {idx:<3} | True RUL: {int(row['True_RUL (cycles)']):<3} | "
            f"Predicted: {row['Predicted_RUL (cycles)']:.1f} | "
            f"Days: {row['RUL_Days']:.1f} | Months: {row['RUL_Months']:.2f} | "
            f"Years: {row['RUL_Years']:.3f} | Status: {row['Status']}")
    report_lines.append(line)

report_lines.append("\nModel Performance:")
report_lines.append("-" * 50)
report_lines.append(f"Random Forest → RMSE: {rf_results['RMSE']:.3f}, MAE: {rf_results['MAE']:.3f}, R²: {rf_results['R²']:.3f}")
report_lines.append(f"XGBoost       → RMSE: {xgb_results['RMSE']:.3f}, MAE: {xgb_results['MAE']:.3f}, R²: {xgb_results['R²']:.3f}")

with open("report/summary_rul_report.txt", "w", encoding="utf-8") as f:
    for line in report_lines:
        f.write(line + "\n")

# ====================================
# 📊 Step 7: Plot Predictions
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

# --- Plot: Predicted RUL in Days/Months ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(rul_summary['RUL_Days'], bins=20, kde=True)
plt.title("⏳ Predicted RUL (Days)")
plt.xlabel("Remaining Useful Life (Days)")

plt.subplot(1, 2, 2)
sns.histplot(rul_summary['RUL_Months'], bins=20, kde=True, color='orange')
plt.title("📅 Predicted RUL (Months)")
plt.xlabel("Remaining Useful Life (Months)")

plt.tight_layout()
plt.show()

# ====================================
# 🧮 Step 8: SHAP Explanation (XGBoost)
# ====================================
print("\n🔍 SHAP Explanations for XGBoost")
explainer = shap.TreeExplainer(xgb.get_booster())
shap_values = explainer(X_train)

shap.plots.bar(shap_values, max_display=10)
shap.plots.beeswarm(shap_values, max_display=10)

idx = 0
sample = X_test.iloc[[idx]]
sample_sv = explainer(sample)
shap.plots.force(sample_sv)