"""
Train all models and save them to models/saved/.

Models trained:
  1. Logistic Regression (classification baseline)
  2. Polynomial Regression (regression baseline)
  3. Linear Regression
  3. XGBoost Classifier + XGBoost Regressor (primary model)

All scalers are fit on training data only and saved alongside the models.
Data is split chronologically (80/10/10) — never shuffled.

All training runs are logged with MLflow.
"""

import os
import sys
import json

import pandas as pd
import yaml
import numpy as np
import skops.io as sio

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

import mlflow
import mlflow.sklearn as mlflow_sklearn
import mlflow.xgboost as mlflow_xgboost

from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths & MLflow setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")
_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "cache", "training_dataset.csv")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Set MLflow tracking URI (can be overridden by env)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "stock-predictor")
_MODEL_NAME_PREFIX = os.getenv("MLFLOW_MODEL_NAME_PREFIX", "stock-predictor")


def _registry_name(suffix: str) -> str:
    return f"{_MODEL_NAME_PREFIX}-{suffix}"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

# Columns that are NOT features
_NON_FEATURE_COLS = {"Ticker", "Direction", "Next_Close", "Pct_Change"}


def load_dataset() -> pd.DataFrame:
    """Load the training dataset CSV."""
    if not os.path.exists(_DATASET_PATH):
        raise FileNotFoundError(
            f"Training dataset not found at {_DATASET_PATH}.\n"
            "Run  python features/build_dataset.py  first."
        )
    df = pd.read_csv(_DATASET_PATH, index_col="Date", parse_dates=True)
    df.sort_index(inplace=True)

    # Handle missing values by dropping them from the dataset
    df.dropna(inplace=True)

    # perform analysis and data visualizations
    data_analysis_visualizations(df)
    return df

def data_analysis_visualizations(df: pd.DataFrame):
    print("Data Analysis and Visualizations")
    print(df.describe())
    df2 = df.copy()
    df2 = df2.reset_index()

    # a boxplot visualizing outliers in the main features
    vis_df = df2[['Date', 'Open', 'High', 'Low', 'Close', 'Direction', 'Pct_Change']]
    plt.boxplot(vis_df[['Open', 'High', 'Low', 'Close', 'Direction', 'Pct_Change']], tick_labels=['Open', 'High', 'Low', 'Close', 'Direction', 'Pct_Change'])
    plt.title('Boxplot of Main Features of the Dataset')
    plt.ylabel('Values')
    plt.xlabel('Features')
    plt.show()

    # Histogram
    # Open
    plt.hist(vis_df['Open'])
    plt.title('Frequency histogram for Open column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # High
    plt.hist(vis_df['High'])
    plt.title('Frequency histogram for High column')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    # Line Graph
    # Low
    plt.plot(vis_df['Date'],vis_df['Low'])
    plt.title('Trend for Low column over time (1990 - 2026)')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()

    # Close
    plt.plot(vis_df['Date'],vis_df['Close'])
    plt.title('Trend for Close column over time (1990 - 2026)')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()

    # Scatter plot
    # Correlation between Direction and Close
    plt.scatter(x=vis_df['Open'], y=vis_df["Close"])
    plt.title('Correlation between Open and Close columns')
    plt.xlabel('Open')
    plt.ylabel('Close')
    plt.show()

    print()

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes targets & Ticker)."""
    return [column for column in df.columns if column not in _NON_FEATURE_COLS]


def chronological_split(df: pd.DataFrame, config: dict):
    """
    Split into train / val / test **chronologically** (no shuffle).

    Returns
    -------
    train_df, val_df, test_df
    """
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"[train] Split: train={len(train_df)}, val={len(val_df)}, "
          f"test={len(test_df)}  (total={n})")
    return train_df, val_df, test_df


def _save_feature_columns(feature_columns: list[str], save_dir: str) -> str:
    """Save feature columns to YAML for evaluate/predict pipelines."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "feature_columns.yaml")
    with open(path, "w") as file:
        yaml.safe_dump(feature_columns, file, sort_keys=False)
    print(f"  Saved → {path}")
    return path


def _save_run_ids(run_ids: dict, save_dir: str) -> str:
    """Persist MLflow run IDs so evaluation can attach test metrics to exact runs."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "mlflow_run_ids.json")
    with open(path, "w") as file:
        json.dump(run_ids, file, indent=2)
    print(f"  Saved → {path}")
    return path


def _save_skops(obj, filename: str, save_dir: str) -> str:
    """Serialize an sklearn-compatible object using skops."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    sio.dump(obj, path)
    print(f"  Saved → {path}")
    return path


def _save_xgb_model(obj, filename: str, save_dir: str) -> str:
    """Serialize an XGBoost model using its native JSON format."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    obj.save_model(path)
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Model 1 — Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, config, save_dir):
    """
    Train Logistic Regression with hyperparameter tuning.
    Logs everything to MLflow and saves the best model locally.
    """
    print("\n" + "=" * 60)
    print("[train] Model 1 — Logistic Regression")
    print("=" * 60)

    # Scale features using standardization
    # so features are weighed more equally and converge to a better solution faster (https://medium.com/@jazeem.lk/why-standardization-is-important-in-machine-learning-9b55a9e03d58)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    logistic_config = config.get('logistic_regression', {})

    # According to (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 'C' is the inverse of regularization strength (smaller value equals greater regularization strength);
    # for this we will try values 0.01, 1, and 10. Note that 1 is the default for 'C' if we do not specify.
    # The default solver for LogisticRegression is lbfgs. For l1_ratio we are 0 is equivalent to L2 regularization, 1 is L1 regularization, and 0.5 is both L1 and L2 regularization.
    parameters = {
        'C': logistic_config.get('C', [0.01, 1, 10]),
        'l1_ratio': logistic_config.get('l1_ratio', [0, 0.5, 1])
    }

    # Default max_iter value is 100 increasing this value allows more attempts (iterations) for the lbfgs solver to converge.
    base_logistic_regression = LogisticRegression(
        max_iter=logistic_config.get('max_iter', 1000),
        random_state=logistic_config.get('random_state', 42),
    )

    # Use TimeSeriesSplit for chronological cross‑validation
    # Like KFold cross validation but it maintains the ordering of dates.
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(base_logistic_regression, parameters, cv=tscv, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)

    # Capture best hyperparameter values
    best_params = grid.best_params_

    # Capture cross validation accuracy score
    cv_accuracy = grid.best_score_

    # Instantiate and fit the final model using best hyperparameter values.
    best_logistic_regression = LogisticRegression(
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        max_iter=logistic_config.get('max_iter', 1000),
        random_state=logistic_config.get('random_state', 42),
    )
    best_logistic_regression.fit(X_train_scaled, y_train)

    # Check model performance using the evaluation set
    y_predict_val = best_logistic_regression.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_predict_val)
    val_precision = precision_score(y_val, y_predict_val, zero_division=0)
    val_recall = recall_score(y_val, y_predict_val, zero_division=0)
    val_f1 = f1_score(y_val, y_predict_val, zero_division=0)

    # Check model performance using the test set
    X_test_scaled = scaler.transform(X_test)
    y_predict_test = best_logistic_regression.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_predict_test)
    test_precision = precision_score(y_test, y_predict_test, zero_division=0)
    test_recall = recall_score(y_test, y_predict_test, zero_division=0)
    test_f1 = f1_score(y_test, y_predict_test, zero_division=0)

    # Print to the console the best parameters cv_accuracy, validation and test accuracy
    print(f"  Best params: {best_params}")
    print(f"  CV accuracy: {cv_accuracy:.4f}")
    print(f"  Val accuracy: {val_accuracy:.4f}")
    print(f"  Val accuracy: {test_accuracy:.4f}")

    # Log Logistic Regression Model to MLFlow
    with mlflow.start_run(run_name="LogisticRegression", nested=True) as run:
        # Log Best Tuned Values For Hyperparameters
        mlflow.log_params(best_params)

        # Log Metrics
        mlflow.log_metrics({
            "cv_accuracy": cv_accuracy,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        })

        # Serialize and save the logistic regression model to the save directory
        logistic_model_path = _save_skops(best_logistic_regression, "logistic_model.skops", save_dir)
        # Upload the serialized model artifact to MLFlow
        mlflow.log_artifact(logistic_model_path)

        # Serialize and save the StandardScaler fitted to the training data for later predictions
        logistic_scaler_path = _save_skops(scaler, "logistic_scaler.skops", save_dir)
        # Upload the serialized StandardScaler artifact to MLFlow
        mlflow.log_artifact(logistic_scaler_path)

        # Upload The Same Logistic Regression Model to MLFlow Using MLFlow format to allow MLFlow loading/serving
        mlflow_sklearn.log_model(
            sk_model=best_logistic_regression,
            name="logistic_regression_model",
            registered_model_name=_registry_name("logistic"),
        )

        # Capture the run ID to associate the model with it
        run_id = run.info.run_id

    return {
        "model": best_logistic_regression,
        "scaler": scaler,
        "best_params": best_params,
        "run_id": run_id,
        "cv_accuracy": cv_accuracy,
        "val_acc": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
    }


# ---------------------------------------------------------------------------
# Model 2 — Polynomial Regression
# ---------------------------------------------------------------------------

def train_polynomial_regression(X_train, y_train, X_val, y_val, X_test, y_test, config, save_dir):
    """
    Train Polynomial Regression for percentage change prediction.
    Logs everything to MLflow and saves the best model locally.
    """
    print("\n" + "=" * 60)
    print("[train] Model 2 — Polynomial Regression")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    poly_config = config.get("polynomial_regression", {})
    degrees = poly_config.get("degrees", [1, 2])

    polynomial_regression = Pipeline([
        ("poly", PolynomialFeatures(include_bias=False)),
        ("linear", LinearRegression()),
    ])

    parameters = {"poly__degree": degrees}
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        polynomial_regression,
        param_grid=parameters,
        cv=tscv,
        scoring="neg_mean_squared_error",
    )
    grid.fit(X_train_scaled, y_train)

    best_polynomial_regression = grid.best_estimator_
    best_params = grid.best_params_
    cv_rmse = np.sqrt(-grid.best_score_)

    y_predict_val = best_polynomial_regression.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_predict_val))
    val_r2 = r2_score(y_val, y_predict_val)

    X_test_scaled = scaler.transform(X_test)
    y_predict_test = best_polynomial_regression.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_predict_test))
    test_r2 = r2_score(y_test, y_predict_test)

    print(f"  Best params: {best_params}")
    print(f"  CV RMSE: {cv_rmse:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")

    with mlflow.start_run(run_name="PolynomialRegression", nested=True) as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_rmse": cv_rmse,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })

        polynomial_model_path = _save_skops(best_polynomial_regression, "polynomial_regression.skops", save_dir)
        polynomial_scaler_path = _save_skops(scaler, "polynomial_scaler.skops", save_dir)
        mlflow.log_artifact(polynomial_model_path)
        mlflow.log_artifact(polynomial_scaler_path)
        mlflow_sklearn.log_model(
            sk_model=best_polynomial_regression,
            name="polynomial_regression_model",
            registered_model_name=_registry_name("polynomial"),
        )

        run_id = run.info.run_id

    return {
        "model": best_polynomial_regression,
        "scaler": scaler,
        "best_params": best_params,
        "run_id": run_id,
        "cv_rmse": cv_rmse,
        "val_rmse": val_rmse,
        "val_r2": val_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
    }



# ---------------------------------------------------------------------------
# Model 3 — XGBoost Classifier + Regressor
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train_classification, y_train_regression,
                  X_val, y_val_classification, y_val_regression,
                  X_test, y_test_classification, y_test_regression,
                  config, save_dir) -> dict:
    print("\n" + "=" * 60)
    print("[train] Model 4 — XGBoost (Classifier + Regressor)")
    print("=" * 60)

    xgb_configuration = config["xgboost"]
    classifier_grid_search = xgb_configuration["classifier_grid_search"]

    # Scale features using standardization recommended
    # for XGBoost to avoid numerical instability and slower convergence (https://medium.com/@indrajeetswain/8-common-xgboost-mistakes-every-data-scientist-should-avoid-0d9985e37968)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Compute scale_pos_weight for class imbalance
    n_down = int((y_train_classification == 0).sum())
    n_up = int((y_train_classification == 1).sum())
    spw = n_down / max(n_up, 1)
    print(f"  Class balance — DOWN: {n_down}, UP: {n_up}, "
          f"scale_pos_weight: {spw:.4f}")

    # --- Classifier with GridSearchCV + TimeSeriesSplit ---
    base_classifier = XGBClassifier(
        random_state=xgb_configuration["random_state"],
        eval_metric="logloss",
        scale_pos_weight=spw,
        use_label_encoder=False
    )
    parameters = {
        "n_estimators": classifier_grid_search["n_estimators"],
        "max_depth": classifier_grid_search["max_depth"],
        "learning_rate": classifier_grid_search["learning_rate"],
    }
    tscv = TimeSeriesSplit(n_splits=xgb_configuration["cv_folds"])
    grid = GridSearchCV(
        base_classifier, param_grid=parameters,
        cv=tscv,
        scoring="accuracy",
    )
    grid.fit(X_train_scaled, y_train_classification)

    best_classifier = grid.best_estimator_
    best_params = grid.best_params_
    cv_accuracy = grid.best_score_

    # Evaluate classifier on validation set
    y_predict_val_cls = best_classifier.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val_classification, y_predict_val_cls)
    val_precision = precision_score(y_val_classification, y_predict_val_cls, zero_division=0)
    val_recall = recall_score(y_val_classification, y_predict_val_cls, zero_division=0)
    val_f1 = f1_score(y_val_classification, y_predict_val_cls, zero_division=0)

    y_predict_test_cls = best_classifier.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_classification, y_predict_test_cls)
    test_precision = precision_score(y_test_classification, y_predict_test_cls, zero_division=0)
    test_recall = recall_score(y_test_classification, y_predict_test_cls, zero_division=0)
    test_f1 = f1_score(y_test_classification, y_predict_test_cls, zero_division=0)

    print(f"  Best params: {best_params}")
    print(f"  Classifier CV accuracy: {cv_accuracy:.4f}")
    print(f"  Classifier val accuracy: {val_accuracy:.4f}")

    classifier_path = _save_xgb_model(best_classifier, "xgboost_classifier.json", save_dir)
    scaler_path = _save_skops(scaler, "xgboost_scaler.skops", save_dir)

    # Log XGBoost classifier to MLFlow
    with mlflow.start_run(run_name="XGBoostClassifier", nested=True) as run:
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_accuracy": cv_accuracy,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        })

        mlflow.log_artifact(classifier_path)
        mlflow.log_artifact(scaler_path)
        mlflow_xgboost.log_model(
            xgb_model=best_classifier,
            name="xgb_classifier_model",
            registered_model_name=_registry_name("xgboost-classifier"),
        )

        classifier_run_id = run.info.run_id

    # --- Regressor with its own GridSearchCV + TimeSeriesSplit ---
    print("  Training XGBRegressor…")
    regressor_grid_search = xgb_configuration.get("regressor_grid_search", classifier_grid_search)
    regressor_parameters = {
        "n_estimators": regressor_grid_search["n_estimators"],
        "max_depth": regressor_grid_search["max_depth"],
        "learning_rate": regressor_grid_search["learning_rate"],
    }

    base_regressor = XGBRegressor(
        random_state=xgb_configuration["random_state"],
    )

    reg_grid = GridSearchCV(
        base_regressor,
        param_grid=regressor_parameters,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
    )
    reg_grid.fit(X_train_scaled, y_train_regression)
    xgb_regressor = reg_grid.best_estimator_
    reg_best_params = reg_grid.best_params_
    cv_rmse_reg = -reg_grid.best_score_

    # Fit final selected estimator on full training split.
    xgb_regressor.fit(X_train_scaled, y_train_regression)

    # Evaluate regressor on validation set (same target as polynomial model: Pct_Change)
    y_predict_val_reg = xgb_regressor.predict(X_val_scaled)
    val_rmse = np.sqrt(mean_squared_error(y_val_regression, y_predict_val_reg))
    val_r2 = r2_score(y_val_regression, y_predict_val_reg)

    y_predict_test_reg = xgb_regressor.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test_regression, y_predict_test_reg))
    test_r2 = r2_score(y_test_regression, y_predict_test_reg)

    print(f"  Regressor best params: {reg_best_params}")
    print(f"  Regressor val RMSE: {val_rmse:.4f}")
    print(f"  Regressor val R2:   {val_r2:.4f}")
    print(f"  Regressor CV RMSE:  {cv_rmse_reg:.4f}")

    regressor_path = _save_xgb_model(xgb_regressor, "xgboost_regressor.json", save_dir)

    # Log XGBoost regressor to MLFlow
    with mlflow.start_run(run_name="XGBoostRegressor", nested=True) as run:
        mlflow.log_params({f"reg_{k}": v for k, v in reg_best_params.items()})
        mlflow.log_metrics({
            "cv_rmse": cv_rmse_reg,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })

        mlflow.log_artifact(regressor_path)
        mlflow.log_artifact(scaler_path)
        mlflow_xgboost.log_model(
            xgb_model=xgb_regressor,
            name="xgb_regressor_model",
            registered_model_name=_registry_name("xgboost-regressor"),
        )

        regressor_run_id = run.info.run_id

    return {
        "classifier": {
            "model": best_classifier,
            "scaler": scaler,
            "best_params": best_params,
            "run_id": classifier_run_id,
            "cv_accuracy": cv_accuracy,
            "val_acc": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        },
        "regressor": {
            "model": xgb_regressor,
            "scaler": scaler,
            "best_params": reg_best_params,
            "run_id": regressor_run_id,
            "cv_rmse": cv_rmse_reg,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        },
    }


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train_all() -> dict:
    """Train all models and return a results dict."""
    config = _load_config()
    save_dir = os.path.join(_PROJECT_ROOT, config["output"]["model_save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    print("[train] Loading training dataset…")
    df = load_dataset()
    feature_columns = get_feature_columns(df)

    print(f"[train] Features ({len(feature_columns)}): {feature_columns[:5]}… "
          f"(+{len(feature_columns)-5} more)")

    # --- Chronological split ---
    train_df, val_df, test_df = chronological_split(df, config)

    X_train = train_df[feature_columns].values
    y_train_classification = train_df["Direction"].values
    y_train_regression = train_df["Pct_Change"].values

    X_val = val_df[feature_columns].values
    y_val_classification = val_df["Direction"].values
    y_val_regression = val_df["Pct_Change"].values

    X_test = test_df[feature_columns].values
    y_test_classification = test_df["Direction"].values
    y_test_regression = test_df["Pct_Change"].values

    # Save feature columns and sets for evaluate.py
    _save_feature_columns(feature_columns, save_dir)
    test_df.to_csv(os.path.join(save_dir, "test_set.csv"))
    val_df.to_csv(os.path.join(save_dir, "val_set.csv"))
    print("  Saved feature_columns.yaml, test_set.csv, and val_set.csv")

    results = {}

    # Create/select experiment explicitly to avoid reliance on default experiment ID.
    mlflow.set_experiment(_EXPERIMENT_NAME)

    # Start a parent MLflow run for the whole training process
    with mlflow.start_run(run_name="Training_Pipeline"):

        # --- Model 1: Logistic Regression ---
        results["logistic"] = train_logistic_regression(
            X_train, y_train_classification, X_val, y_val_classification,
            X_test, y_test_classification,
            config, save_dir
        )

        # --- Model 2: Polynomial Regression ---
        results["polynomial"] = train_polynomial_regression(
            X_train, y_train_regression, X_val, y_val_regression,
            X_test, y_test_regression,
            config, save_dir
        )

        # --- Model 3: XGBoost ---
        results["xgboost"] = train_xgboost(
            X_train, y_train_classification, y_train_regression,
            X_val, y_val_classification, y_val_regression,
            X_test, y_test_classification, y_test_regression,
            config, save_dir
        )

    _save_run_ids(
        {
            "logistic": results["logistic"]["run_id"],
            "polynomial": results["polynomial"]["run_id"],
            "xgboost_classifier": results["xgboost"]["classifier"]["run_id"],
            "xgboost_regressor": results["xgboost"]["regressor"]["run_id"],
        },
        save_dir,
    )

    # --- Classification comparison: Logistic vs XGBoost Classifier ---
    if "logistic" in results and "xgboost" in results:
        logistic = results["logistic"]
        xgb = results["xgboost"]["classifier"]
        print("\n" + "=" * 60)
        print("  CLASSIFICATION COMPARISON")
        print("=" * 60)
        print(f"  {'Model':<25} {'CV Acc':>10} {'Val Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(
            f"  {'Logistic Regression':<25} "
            f"{logistic.get('cv_accuracy', float('nan')):>10.4f} "
            f"{logistic.get('val_acc', float('nan')):>10.4f} "
            f"{logistic.get('val_precision', float('nan')):>10.4f} "
            f"{logistic.get('val_recall', float('nan')):>10.4f} "
            f"{logistic.get('val_f1', float('nan')):>10.4f}"
        )
        print(
            f"  {'XGBoost Classifier':<25} "
            f"{xgb.get('cv_accuracy', float('nan')):>10.4f} "
            f"{xgb.get('val_acc', float('nan')):>10.4f} "
            f"{xgb.get('val_precision', float('nan')):>10.4f} "
            f"{xgb.get('val_recall', float('nan')):>10.4f} "
            f"{xgb.get('val_f1', float('nan')):>10.4f}"
        )
        print("=" * 60)

    # --- Regression comparison: Polynomial vs XGBoost Regressor ---
    if "polynomial" in results and "xgboost" in results:
        poly = results["polynomial"]
        xgb = results["xgboost"]["regressor"]
        print("\n" + "=" * 60)
        print("  REGRESSION COMPARISON (Pct_Change)")
        print("=" * 60)
        print(f"  {'Model':<25} {'CV RMSE':>10} {'Val RMSE':>10} {'Val MAE':>10} {'Val R2':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        print(
            f"  {'Polynomial Regression':<25} "
            f"{poly.get('cv_rmse', float('nan')):>10.4f} "
            f"{poly.get('val_rmse', float('nan')):>10.4f} "
            f"{poly.get('val_r2', float('nan')):>10.4f}"
        )
        print(
            f"  {'XGBoost Regressor':<25} "
            f"{xgb.get('cv_rmse', float('nan')):>10.4f} "
            f"{xgb.get('val_rmse', float('nan')):>10.4f} "
            f"{xgb.get('val_r2', float('nan')):>10.4f}"
        )
        print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_all()
