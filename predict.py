import joblib
import pandas as pd
from pathlib import Path
import numpy as np
import shap  # Import the shap library


def predict_rul(input_data: dict):
    """
    Loads the appropriate pre-trained model from a repository and predicts RUL.

    Args:
        input_data (dict): A dictionary containing engine metadata and sensor data.
                           Must include 'dataset' to identify the model (e.g., 'FD001').

    Returns:
        dict: A dictionary containing the raw prediction, human-readable metrics,
              and SHAP values for explanation.
    """
    if "engine_type" not in input_data or "condition" not in input_data:
        raise ValueError("Input data must contain 'engine_type' and 'condition' metadata.")

    engine_type = input_data.pop("engine_type")
    condition = input_data.pop("condition")

    model_dir = Path("models") / f"engine_type_{engine_type}" / f"condition_{condition}"
    model_path = model_dir / "xgb_model.pkl"
    features_path = model_dir / "feature_columns.pkl"

    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"Model or feature columns not found for engine type '{engine_type}' and condition '{condition}'.")

    # Load the trained model and the list of feature columns
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)

    # Ensure "unit" and "time" are not in feature_cols as they're not used for prediction
    # and could cause errors during DataFrame creation.
    feature_cols = [col for col in feature_cols if col not in ["unit", "time"]]

    # Create a DataFrame from the input data, ensuring columns are in the correct order
    input_df = pd.DataFrame([input_data], columns=feature_cols)
    if input_df.isnull().values.any():
        missing_cols = input_df.columns[input_df.isnull().any()].tolist()
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

    # Make the prediction (this is very fast)
    predicted_rul = model.predict(input_df)[0]

    # --- Generate Human-Readable Report ---
    cycle_to_day_ratio = 1  # This can be adjusted based on domain knowledge
    rul_days = predicted_rul * cycle_to_day_ratio
    rul_months = rul_days / 30
    rul_years = rul_days / 365

    # Determine status based on predicted RUL in days
    if rul_days <= 10:
        status = '☠️ CRITICAL'
    elif rul_days <= 90:
        status = '⚠️ DANGER'
    else:
        status = '🟢 OK'

    # --- Generate SHAP explanation data ---
    explainer = shap.TreeExplainer(model)
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
            "shap_values": shap_values,
            "feature_names": feature_cols,
        },
    }


if __name__ == "__main__":
    # Example: Simulating a prediction request
    request_data = {
        "engine_type": "FD002",  # Corresponds to the model to load
        "condition": "standard", # Placeholder for now
        'op_setting_1': 25.0, 'op_setting_2': 0.62, 'op_setting_3': 60.0,
        'sensor_2': 643.0, 'sensor_3': 1592.0, 'sensor_4': 1408.0,
        'sensor_7': 552.0, 'sensor_8': 2388.0, 'sensor_9': 9058.0,
        'sensor_11': 47.3, 'sensor_12': 521.0, 'sensor_13': 2388.0,
        'sensor_14': 8128.0, 'sensor_15': 8.4, 'sensor_17': 393.0,
        'sensor_20': 38.8, 'sensor_21': 23.3
    }

    try:
        result = predict_rul(request_data.copy()) # Pass a copy to avoid modification
        pred = result["prediction"]
        print("\n--- RUL Prediction Summary ---")
        print(f"  Predicted RUL: {pred['rul_cycles']} cycles")
        print(f"  Status:        {pred['status']}")
        print(f"  Time Left:     ~{pred['rul_days']} days / ~{pred['rul_months']} months / ~{pred['rul_years']} years")
        print("----------------------------\n")

        # The calling application can now decide to render the SHAP plot
        print("🔍 Generating SHAP force plot for this specific prediction...")
        shap.initjs() # Necessary for notebooks/some environments
        display(shap.plots.force(result["explanation"]["shap_values"]))

    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")