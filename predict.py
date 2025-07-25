import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shap
import matplotlib.pyplot as plt


def prepare_input_data(source, engine_type: str, condition: str = "standard", row_index: int = 0) -> dict:
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
        from training import infer_column_schema
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


def detect_input_type(source):
    if isinstance(source, dict):
        return "form"
    elif isinstance(source, pd.DataFrame):
        return "stream"
    elif isinstance(source, str) and source.endswith(".csv"):
        return "csv"
    elif isinstance(source, str) and source.endswith(".txt"):
        return "txt"
    else:
        raise ValueError("Unsupported input source type")


if __name__ == "__main__":
    try:
        source = "data/test_FD002.txt"  # or dict, CSV, or DataFrame
        input_type = detect_input_type(source)

        if input_type == "form":
            input_row = prepare_input_data(source, engine_type="FD002")
        elif input_type == "csv":
            input_row = prepare_input_data(source, engine_type="FD002", row_index=0)
        elif input_type == "txt":
            input_row = prepare_input_data(source, engine_type="FD002", row_index=0)
        elif input_type == "stream":
            input_row = prepare_input_data(source, engine_type="FD002", row_index=0)

        result = predict_rul(input_row)

        pred = result["prediction"]
        print("\n--- RUL Prediction Summary ---")
        print(f"  Predicted RUL: {pred['rul_cycles']} cycles")
        print(f"  Status:        {pred['status']}")
        print(f"  Time Left:     ~{pred['rul_days']} days / ~{pred['rul_months']} months / ~{pred['rul_years']} years\n")

        save_shap_force_plot(result["explanation"], save_path="reports/shap_force_plot_detected.png")

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
