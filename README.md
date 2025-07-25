# Predicting Remaining Useful Life (RUL) of Turbofan Engines

This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. It employs machine learning models, specifically `RandomForestRegressor` and `XGBRegressor`, to forecast when an engine will require maintenance. The analysis includes comprehensive Exploratory Data Analysis (EDA), feature selection, model training, and explainability using SHAP to understand the model's predictions.

## 🚀 Features

- **Data Preprocessing**: Loads the C-MAPSS data and calculates RUL for each engine cycle.
- **Exploratory Data Analysis (EDA)**: Visualizes sensor distributions, RUL degradation, and sensor trends over time.
- **Feature Selection**: Selects the most relevant features by filtering out sensors with low variance and low correlation to the RUL.
- **Machine Learning Models**: Trains and evaluates RandomForest and XGBoost regressors.
- **Model Explainability**: Uses SHAP (SHapley Additive exPlanations) to interpret the XGBoost model's predictions and identify key features.

## 💿 Dataset

The project utilizes the **Turbofan Engine Degradation Simulation Data Set** from the NASA Ames Prognostics Data Repository.

Specifically, the `FD001` training dataset (`train_FD001.txt`) is used for this analysis. You will need to download this file and place it in the `data/` directory.

- **Link**: NASA Prognostics Data Repository

## 📁 Project Structure

```
.
├── data/
│   └── train_FD001.txt
├── model/
│   └── rul_xgboost.py
└── README.md
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone 'https://github.com/Ajioz/predictingRuL_v1'
    cd `predictingRuL_v1`
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
    ```

4.  **Download the data:**
    Download the `train_FD001.txt` file from the NASA data repository and place it in a `data/` directory at the project root.

## ▶️ Usage

To run the complete analysis, training, and evaluation pipeline, execute the main script from the root directory:

```bash
python model/rul_xgboost.py
```

The script will:
- Load and preprocess the data.
- Display several plots for EDA, feature selection, and results.
- Print the model evaluation metrics to the console.
- Generate and display SHAP plots for model explainability.

## 🛠️ Workflow

The script follows a structured machine learning workflow:

1.  **Data Loading & RUL Calculation**: The `train_FD001.txt` data is loaded into a pandas DataFrame. The Remaining Useful Life (RUL) is calculated for each engine unit by subtracting the current cycle from the maximum cycle of that unit.
2.  **Exploratory Data Analysis (EDA)**: Visualizations are generated to understand the data, including:
    - Distributions of sensor readings.
    - RUL degradation curves over time for sample engines.
    - Trends of individual sensors over time.
3.  **Feature Selection**: To improve model performance and reduce noise, features are selected based on two criteria:
    - **Low Variance**: Sensors with nearly constant (flat) values are removed using `VarianceThreshold`.
    - **Low Correlation**: Sensors that have a weak correlation with the target variable (RUL) are dropped.
4.  **Model Training**: Two regression models are trained to predict RUL: `RandomForestRegressor` and `XGBRegressor`.
5.  **Model Evaluation**: The models are evaluated on a held-out test set using **RMSE**, **MAE**, and **R²** metrics.
6.  **Model Explainability with SHAP**: SHAP is used to interpret the predictions of the XGBoost model, helping to understand which features are most influential for its decisions.

## 📊 Results & Discussion

The script evaluates the models and prints their performance metrics.

### Current Implementation Issue

The current implementation trains and tests the models on the **final operational cycle** of each engine. By definition, the RUL for this last cycle is always zero (`max_cycle - max_cycle = 0`). Consequently, the model is trained to predict a constant value of zero.

This leads to misleadingly perfect evaluation scores on the test set, as the target variable is also exclusively zero.

```shell
# Console Output from the script
📊 Random Forest Results: {'RMSE': 0.0, 'MAE': 0.0, 'R²': 1.0}
📊 XGBoost Results: {'RMSE': 0.0, 'MAE': 0.0, 'R²': 1.0}
```

The prediction plots generated by the script will show all points lying on a horizontal line at `Predicted RUL = 0`, confirming this behavior.

While the target variable is flawed, the SHAP analysis still offers some insight into which features the model found most discriminative for identifying a final-cycle data point.

## 💡 Future Improvements

To build a truly predictive model, the following improvements are recommended:

- **Correct Training Data**: Modify the training pipeline to use data from **all** operational cycles, not just the last one. This will train the model to predict RUL at any point in an engine's lifespan. The target `y` should be the `RUL` column for the entire dataset.

  ```python
  # Suggested change in Step 4
  X = sensor_df[final_feature_cols]
  y = sensor_df['RUL']
  
  # Split data while keeping engine units intact in train/test sets
  # to prevent data leakage.
  # X_train, X_test, y_train, y_test = ...
  ```

- **Time-Series Features**: Engineer features that capture trends over time (e.g., moving averages, slopes of sensor readings) to provide the model with more context.

- **Robust Cross-Validation**: Implement a more robust cross-validation strategy, such as time-based splitting or group-based splitting (by engine `unit`), to prevent data leakage between training and testing sets.

- **Hyperparameter Tuning**: Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for the models.

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.