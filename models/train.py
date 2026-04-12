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
import warnings
import pickle
import joblib

import pandas as pd
import yaml
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import PolynomialFeatures

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models.signature import infer_signature

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)

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
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))


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
    return df


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


def _save_artifact(obj, filename: str, save_dir: str) -> str:
    """Pickle an object to save_dir/filename. Returns the full path."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "wb") as file:
        pickle.dump(obj, file)
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Model 1 — Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, X_val, y_val,
                              feature_cols, config, save_dir):
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

    # Hyperparameters to try
    # C represents the inverse of regularization strength (Smaller c = stronger regularization = greater penalty for larger weights)
    # Solvers are optimization algorithms.
    # Penalty represents the method of regularization, in our case we are using ridge regression 
    parameters = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }

    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Use TimeSeriesSplit for chronological cross‑validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(logistic_regression, parameters, cv=tscv, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)

    best_logistic_regression = grid.best_estimator_
    best_params = grid.best_params_
    cv_accuracy = grid.best_score_

    # Evaluate on validation set
    y_predict_val = best_logistic_regression.predict(X_val_scaled)
    val_accuracy = accuracy_score(y_val, y_predict_val)
    val_precision = precision_score(y_val, y_predict_val, zero_division=0)
    val_recall = recall_score(y_val, y_predict_val, zero_division=0)
    val_f1 = f1_score(y_val, y_predict_val, zero_division=0)

    print(f"  Best params: {best_params}")
    print(f"  CV accuracy: {cv_accuracy:.4f}")
    print(f"  Val accuracy: {val_accuracy:.4f}")

    # Log Logistic Regression Model to MLFlow
    with mlflow.start_run(run_name="LogisticRegression", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_accuracy": cv_accuracy,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        # Log the scaler as an artifact
        scaler_path = os.path.join(save_dir, "logistic_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        # Log the model with signature
        signature = infer_signature(X_train_scaled, best_logistic_regression.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            best_logistic_regression,
            "logistic_model",
            signature=signature,
            input_example=X_train_scaled[:5]
        )

    # Save locally for compatibility
    joblib.dump(best_logistic_regression, os.path.join(save_dir, "logistic_regression.pkl"))
    _save_artifact(scaler, "logistic_scaler.pkl", save_dir)

    return {
        "model": best_logistic_regression,
        "scaler": scaler,
        "best_params": best_params,
        "val_acc": val_accuracy
    }


# ---------------------------------------------------------------------------
# Model 2 — Polynomial Regression
# ---------------------------------------------------------------------------

def train_polynomial_regression(X_train, y_train, X_val, y_val,
                                feature_cols, config, save_dir):
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
    degrees = poly_config.get("degrees", [2, 3])

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
    val_mae = mean_absolute_error(y_val, y_predict_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_predict_val))
    val_r2 = r2_score(y_val, y_predict_val)

    print(f"  Best params: {best_params}")
    print(f"  CV RMSE: {cv_rmse:.4f}")
    print(f"  Val RMSE: {val_rmse:.4f}")

    with mlflow.start_run(run_name="PolynomialRegression", nested=True):
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_rmse": cv_rmse,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2
        })

        scaler_path = os.path.join(save_dir, "polynomial_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        signature = infer_signature(
            X_train_scaled,
            best_polynomial_regression.predict(X_train_scaled)
        )
        mlflow.sklearn.log_model(
            best_polynomial_regression,
            "polynomial_model",
            signature=signature,
            input_example=X_train_scaled[:5]
        )

    joblib.dump(
        best_polynomial_regression,
        os.path.join(save_dir, "polynomial_regression.pkl")
    )
    _save_artifact(scaler, "polynomial_scaler.pkl", save_dir)

    return {
        "model": best_polynomial_regression,
        "scaler": scaler,
        "best_params": best_params,
        "val_rmse": val_rmse
    }



# ---------------------------------------------------------------------------
# Model 3 —  Linear Regression 
# ---------------------------------------------------------------------------
def train_linear_regression(X_train, y_train_regression,
                             X_val, y_val_regression,
                             feature_cols, config, save_dir) -> dict:
    """
    Train a Linear Regression model to predict next-day percentage change.
    Mirrors the XGBoost regressor: target is Pct_Change, output is converted
    to a dollar price at inference time.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import mlflow.sklearn

    print("\n" + "=" * 60)
    print("[train] Model 3 — Linear Regression")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train_regression)

    y_pred_val = model.predict(X_val_scaled)
    val_mae  = mean_absolute_error(y_val_regression, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val_regression, y_pred_val))
    val_r2   = r2_score(y_val_regression, y_pred_val)

    print(f"  Val MAE  : {val_mae:.4f}")
    print(f"  Val RMSE : {val_rmse:.4f}")
    print(f"  Val R²   : {val_r2:.4f}")

    # Log to MLflow
    with mlflow.start_run(run_name="LinearRegression", nested=True):
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics({
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        })
        scaler_path = os.path.join(save_dir, "linear_regression_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))
        mlflow.sklearn.log_model(
            model,
            "linear_regression_model",
            signature=signature,
            input_example=X_train_scaled[:5]
        )

    # Save locally (same pattern as XGBoost)
    _save_artifact(model,  "linear_regression.pkl",        save_dir)
    _save_artifact(scaler, "linear_regression_scaler.pkl", save_dir)

    return {
        "model":    model,
        "scaler":   scaler,
        "val_mae":  val_mae,
        "val_rmse": val_rmse,
        "val_r2":   val_r2,
    }


# ---------------------------------------------------------------------------
# Model 4 — XGBoost Classifier + Regressor
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train_classification, y_train_regression,
                  X_val, y_val_classification, y_val_regression,
                  feature_cols, config, save_dir) -> dict:
    print("\n" + "=" * 60)
    print("[train] Model 4 — XGBoost (Classifier + Regressor)")
    print("=" * 60)

    xgb_configuration = config["xgboost"]
    grid_search = xgb_configuration["grid_search"]

    # Scale features using standardization recommended
    # for XGBoost to avoid numerical instability and slower convergence (https://medium.com/@indrajeetswain/8-common-xgboost-mistakes-every-data-scientist-should-avoid-0d9985e37968)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

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
        "n_estimators": grid_search["n_estimators"],
        "max_depth": grid_search["max_depth"],
        "learning_rate": grid_search["learning_rate"],
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

    print(f"  Best params: {best_params}")
    print(f"  Classifier CV accuracy: {cv_accuracy:.4f}")
    print(f"  Classifier val accuracy: {val_accuracy:.4f}")

    # --- Regressor (re‑use best depth & estimators from classifier) ---
    print("  Training XGBRegressor…")
    xgb_regressor = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=xgb_configuration["random_state"],
    )
    xgb_regressor.fit(X_train_scaled, y_train_regression)

    # Save locally
    _save_artifact(best_classifier, "xgboost_classifier.pkl", save_dir)
    _save_artifact(xgb_regressor, "xgboost_regressor.pkl", save_dir)
    _save_artifact(scaler, "xgboost_scaler.pkl", save_dir)

    # Log XGBoost Models to MLFlow
    with mlflow.start_run(run_name="XGBoost", nested=True):
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_accuracy": cv_accuracy,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        # Log scaler as artifact locally
        scaler_path = os.path.join(save_dir, "xgboost_scaler.pkl")
        mlflow.log_artifact(scaler_path)

        # Log the XGBoost classifier with signature
        signature_classifier = infer_signature(X_train_scaled, best_classifier.predict(X_train_scaled))
        mlflow.xgboost.log_model(
            best_classifier,
            "xgboost_classifier",
            signature=signature_classifier,
            input_example=X_train_scaled[:5]
        )

        # Log the XGBoost regressor with signature
        signature_regressor = infer_signature(X_train_scaled, xgb_regressor.predict(X_train_scaled))
        mlflow.xgboost.log_model(
            xgb_regressor,
            "xgboost_regressor",
            signature=signature_regressor,
            input_example=X_train_scaled[:5]
        )

        # Register the classifier as the primary model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/xgboost_classifier",
            "stock_direction_predictor"
        )

    return {
        "classifier": best_classifier,
        "regressor": xgb_regressor,
        "scaler": scaler,
        "best_params": best_params,
        "val_acc": val_accuracy,
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

    # Save feature column list and test set for evaluate.py
    _save_artifact(feature_columns, "feature_columns.pkl", save_dir)
    test_df.to_csv(os.path.join(save_dir, "test_set.csv"))
    val_df.to_csv(os.path.join(save_dir, "val_set.csv"))
    print(f"  Saved feature_columns.pkl and test_set.csv")

    results = {}

    # Start a parent MLflow run for the whole training process
    with mlflow.start_run(run_name="Training_Pipeline"):

        # --- Model 1: Logistic Regression ---
        results["logistic"] = train_logistic_regression(
            X_train, y_train_classification, X_val, y_val_classification,
            feature_columns, config, save_dir
        )

        # --- Model 2: Polynomial Regression ---
        results["polynomial"] = train_polynomial_regression(
            X_train, y_train_regression, X_val, y_val_regression,
            feature_columns, config, save_dir
        )

        # --- Model 3: Linear Regression ---
        results["linear_regression"] = train_linear_regression(
            X_train, y_train_regression,
            X_val, y_val_regression,
            feature_columns, config, save_dir
        )

        # --- Model 4: XGBoost ---
        results["xgboost"] = train_xgboost(
            X_train, y_train_classification, y_train_regression,
            X_val, y_val_classification, y_val_regression,
            feature_columns, config, save_dir
        )

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Metric':<12} {'Value':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for name, result in results.items():
        if result:
            if "val_acc" in result:
                metric_name = "Val Acc"
                metric_value = result["val_acc"]
            elif "val_rmse" in result:
                metric_name = "Val RMSE"
                metric_value = result["val_rmse"]
            else:
                metric_name = "N/A"
                metric_value = 0.0
            print(f"  {name:<25} {metric_name:<12} {metric_value:>10.4f}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_all()
