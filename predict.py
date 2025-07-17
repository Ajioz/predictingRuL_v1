import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt


def prepare_input_data(source, engine_type: str, condition: str = "standard", row_index: int = 0) -> dict:
    """
    Prepares input data for inference from various formats.

    Args:
        source: str (file path), pd.DataFrame, or dict
        engine_type: C-MAPSS engine type (e.g., 'FD001')
        condition: operational condition name (default: "standard")
        row_index: row to extract (for batch input)

    Returns:
        dict: Cleaned input ready for prediction
    """
    model_dir = Path("models")
    feature_path = model_dir / f"{engine_type.lower()}_feature_columns.pkl"

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    feature_cols = joblib.load(feature_path)

    if isinstance(source, dict):
        input_data = source.copy()

    elif isinstance(source, pd.DataFrame):
        input_data = source.iloc[row_index].to_dict()

    elif isinstance(source, str) and source.endswith(".txt"):
        df_raw = pd.read_csv(source, sep=r"\s+", header=None)
        from training import infer_column_schema  # must exist in same project
        col_names, _, _ = infer_column_schema(df_raw)
        df_raw.columns = col_names
        input_data = df_raw.iloc[row_index][feature_cols].to_dict()

    elif isinstance(source, str) and source.endswith(".csv"):
        df = pd.read_csv(source)
        input_data = df.iloc[row_index][feature_cols].to_dict()

    else:
        raise ValueError("Unsupported input type. Provide a dict, DataFrame, or file path.")

    input_data["engine_type"] = engine_type
    input_data["condition"] = condition
    return input_data


def predict_rul(input_data: dict) -> dict:
    """
    Predict Remaining Useful Life (RUL) and explain using SHAP.

    Args:
        input_data (dict): Must include 'engine_type' and sensor/setting values.

    Returns:
        dict: Prediction result with SHAP explanation.
    """
    if "engine_type" not in input_data:
        raise ValueError("Missing 'engine_type' in input data.")

    engine_type = input_data.pop("engine_type")
    condition = input_data.pop("condition", "standard")

    model_dir = Path("models")
    model_path = model_dir / f"{engine_type.lower()}_xgb_model.pkl"
    features_path = model_dir / f"{engine_type.lower()}_feature_columns.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"Missing model or feature file for {engine_type}.")

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    feature_cols = [c for c in feature_cols if c not in ["unit", "time"]]

    input_df = pd.DataFrame([input_data], columns=feature_cols)
    if input_df.isnull().values.any():
        missing = input_df.columns[input_df.isnull().any()].tolist()
        raise ValueError(f"Missing required features: {missing}")

    predicted_rul = model.predict(input_df)[0]
    rul_days, rul_months, rul_years = predicted_rul, predicted_rul / 30, predicted_rul / 365

    status = (
        "☠️ CRITICAL" if rul_days <= 10 else
        "⚠️ DANGER" if rul_days <= 90 else
        "🟢 OK"
    )

    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)

    return {
        "prediction": {
            "rul_cycles": round(predicted_rul, 2),
            "rul_days": round(rul_days, 2),
            "rul_months": round(rul_months, 2),
            "rul_years": round(rul_years, 3),
            "status": status,
        },
        "explanation": {
            "shap_values_array": shap_values.values[0].tolist(),
            "expected_value": shap_values.base_values[0],
            "feature_names": feature_cols,
            "raw_input": input_df.values[0].tolist()
        }
    }


def save_shap_force_plot(explanation: dict, save_path: str = "reports/shap_force_plot.png"):
    """
    Saves SHAP force plot as PNG.

    Args:
        explanation (dict): SHAP explanation data from predict_rul
        save_path (str): Path to save image
    """
    Path(save_path).parent.mkdir(exist_ok=True)

    shap_values = shap.Explanation(
        values=np.array(explanation["shap_values_array"]),
        base_values=np.array(explanation["expected_value"]),
        data=np.array([explanation["raw_input"]]),
        feature_names=explanation["feature_names"]
    )
    shap.plots.force(shap_values, matplotlib=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"📈 SHAP force plot saved to '{save_path}'")


if __name__ == "__main__":
    try:
        # Example 1: From C-MAPSS test .txt file
        row_txt = prepare_input_data("data/test_FD002.txt", engine_type="FD002", row_index=0)
        result_txt = predict_rul(row_txt)
        save_shap_force_plot(result_txt["explanation"], save_path="reports/shap_force_plot_txt.png")

        # Example 2: From CSV (e.g., user uploaded batch)
        row_csv = prepare_input_data("uploads/user_input.csv", engine_type="FD002", row_index=1)
        result_csv = predict_rul(row_csv)
        save_shap_force_plot(result_csv["explanation"], save_path="reports/shap_force_plot_csv.png")

        # Example 3: From UI Form (dictionary input)
        ui_input = {
            'op_setting_1': 25.0, 'op_setting_2': 0.62, 'op_setting_3': 60.0,
            'sensor_2': 643.0, 'sensor_3': 1592.0, 'sensor_4': 1408.0,
            'sensor_7': 552.0, 'sensor_8': 2388.0, 'sensor_9': 9058.0,
            'sensor_11': 47.3, 'sensor_12': 521.0, 'sensor_13': 2388.0,
            'sensor_14': 8128.0, 'sensor_15': 8.4, 'sensor_17': 393.0,
            'sensor_20': 38.8, 'sensor_21': 23.3
        }
        row_form = prepare_input_data(ui_input, engine_type="FD002")
        result_form = predict_rul(row_form)
        save_shap_force_plot(result_form["explanation"], save_path="reports/shap_force_plot_form.png")

        # Example 4: From live streaming or batch dataframe (simulate with pandas)
        df_stream = pd.read_csv("data/test_FD002.txt", sep=r"\s+", header=None)
        from training import infer_column_schema
        cols, _, _ = infer_column_schema(df_stream)
        df_stream.columns = cols
        row_stream = prepare_input_data(df_stream, engine_type="FD002", row_index=5)
        result_stream = predict_rul(row_stream)
        save_shap_force_plot(result_stream["explanation"], save_path="reports/shap_force_plot_stream.png")

        # Print one of the summaries
        pred = result_txt["prediction"]
        print("\n--- Example: RUL Prediction from .txt ---")
        print(f"  Predicted RUL: {pred['rul_cycles']} cycles")
        print(f"  Status:        {pred['status']}")
        print(f"  Time Left:     ~{pred['rul_days']} days / ~{pred['rul_months']} months / ~{pred['rul_years']} years\n")

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
