"""
Train all models and save them to models/saved/.

Models trained:
  1. Logistic Regression (classification baseline)
  2. XGBoost Classifier + XGBoost Regressor (primary model)

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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes targets & Ticker)."""
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def chronological_split(df: pd.DataFrame, cfg: dict):
    """
    Split into train / val / test **chronologically** (no shuffle).

    Returns
    -------
    train_df, val_df, test_df
    """
    train_r = cfg["split"]["train_ratio"]
    val_r = cfg["split"]["val_ratio"]

    n = len(df)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))

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
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Model 1 — Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, X_val, y_val,
                              feature_cols, cfg, save_dir):
    """
    Train Logistic Regression with hyperparameter tuning.
    Logs everything to MLflow and saves the best model locally.
    """
    print("\n" + "=" * 60)
    print("[train] Model 1 — Logistic Regression")
    print("=" * 60)

    # Scale features for using standardization
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Hyperparameter grid from config (or define here)
    parameters = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'penalty': ['l2']
    }

    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    # Use TimeSeriesSplit for chronological cross‑validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(logistic_regression, parameters, cv=tscv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_sc, y_train)

    best_logistic_regression = grid.best_estimator_
    best_params = grid.best_params_
    cv_accuracy = grid.best_score_

    # Evaluate on validation set
    y_pred_val = best_logistic_regression.predict(X_val_sc)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    val_precision = precision_score(y_val, y_pred_val, zero_division=0)
    val_recall = recall_score(y_val, y_pred_val, zero_division=0)
    val_f1 = f1_score(y_val, y_pred_val, zero_division=0)

    print(f"  Best params: {best_params}")
    print(f"  CV accuracy: {cv_accuracy:.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")

    # Log to MLflow
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
        signature = infer_signature(X_train_sc, best_lr.predict(X_train_sc))
        mlflow.sklearn.log_model(
            best_lr,
            "logistic_model",
            signature=signature,
            input_example=X_train_sc[:5]
        )

    # Save locally for compatibility
    joblib.dump(best_lr, os.path.join(save_dir, "logistic_regression.pkl"))
    _save_artifact(scaler, "logistic_scaler.pkl", save_dir)

    return {
        "model": best_lr,
        "scaler": scaler,
        "best_params": best_params,
        "val_acc": val_acc
    }


# ---------------------------------------------------------------------------
# Model 2 — XGBoost Classifier + Regressor
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train_cls, y_train_reg,
                  X_val, y_val_cls, y_val_reg,
                  feature_cols, cfg, save_dir) -> dict:
    print("\n" + "=" * 60)
    print("[train] Model 2 — XGBoost (Classifier + Regressor)")
    print("=" * 60)

    xgb_cfg = cfg["xgboost"]
    gs = xgb_cfg["grid_search"]

    # Scale features (optional for tree models, but we keep it)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    # Compute scale_pos_weight for class imbalance
    n_down = int((y_train_cls == 0).sum())
    n_up = int((y_train_cls == 1).sum())
    spw = n_down / max(n_up, 1)
    print(f"  Class balance — DOWN: {n_down}, UP: {n_up}, "
          f"scale_pos_weight: {spw:.4f}")

    # --- Classifier with GridSearchCV + TimeSeriesSplit ---
    base_clf = XGBClassifier(
        random_state=xgb_cfg["random_state"],
        eval_metric="logloss",
        scale_pos_weight=spw,
        verbosity=0,
        use_label_encoder=False
    )
    param_grid = {
        "n_estimators": gs["n_estimators"],
        "max_depth": gs["max_depth"],
        "learning_rate": gs["learning_rate"],
    }
    tscv = TimeSeriesSplit(n_splits=xgb_cfg["cv_folds"])
    grid = GridSearchCV(
        base_clf, param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train_sc, y_train_cls)

    best_clf = grid.best_estimator_
    best_params = grid.best_params_
    cv_acc = grid.best_score_

    # Evaluate classifier on validation set
    y_pred_val_cls = best_clf.predict(X_val_sc)
    val_acc = accuracy_score(y_val_cls, y_pred_val_cls)
    val_prec = precision_score(y_val_cls, y_pred_val_cls, zero_division=0)
    val_rec = recall_score(y_val_cls, y_pred_val_cls, zero_division=0)
    val_f1 = f1_score(y_val_cls, y_pred_val_cls, zero_division=0)

    print(f"  Best params: {best_params}")
    print(f"  Classifier CV accuracy: {cv_acc:.4f}")
    print(f"  Classifier val accuracy: {val_acc:.4f}")

    # --- Regressor (re‑use best depth & estimators from classifier) ---
    print("  Training XGBRegressor…")
    regressor = XGBRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        random_state=xgb_cfg["random_state"],
        verbosity=0,
    )
    regressor.fit(X_train_sc, y_train_reg)

    # Save locally
    _save_artifact(best_clf, "xgboost_classifier.pkl", save_dir)
    _save_artifact(regressor, "xgboost_regressor.pkl", save_dir)
    _save_artifact(scaler, "xgboost_scaler.pkl", save_dir)

    # --- MLflow logging ---
    with mlflow.start_run(run_name="XGBoost", nested=True):
        # Log parameters
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "cv_accuracy": cv_acc,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1
        })

        # Log scaler as artifact
        scaler_path = os.path.join(save_dir, "xgboost_scaler.pkl")
        mlflow.log_artifact(scaler_path)

        # Log classifier with signature
        signature_clf = infer_signature(X_train_sc, best_clf.predict(X_train_sc))
        mlflow.xgboost.log_model(
            best_clf,
            "xgboost_classifier",
            signature=signature_clf,
            input_example=X_train_sc[:5]
        )

        # Log regressor (optional) – we might not need it in the registry
        signature_reg = infer_signature(X_train_sc, regressor.predict(X_train_sc))
        mlflow.xgboost.log_model(
            regressor,
            "xgboost_regressor",
            signature=signature_reg,
            input_example=X_train_sc[:5]
        )

        # Register the classifier as the primary model
        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/xgboost_classifier",
            "stock_direction_predictor"
        )

    return {
        "classifier": best_clf,
        "regressor": regressor,
        "scaler": scaler,
        "best_params": best_params,
        "val_acc": val_acc,
    }


# ---------------------------------------------------------------------------
# Main training orchestrator
# ---------------------------------------------------------------------------

def train_all() -> dict:
    """Train all models and return a results dict."""
    cfg = _load_config()
    save_dir = os.path.join(_PROJECT_ROOT, cfg["output"]["model_save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # --- Load data ---
    print("[train] Loading training dataset…")
    df = load_dataset()
    feature_cols = get_feature_columns(df)

    print(f"[train] Features ({len(feature_cols)}): {feature_cols[:5]}… "
          f"(+{len(feature_cols)-5} more)")

    # --- Chronological split ---
    train_df, val_df, test_df = chronological_split(df, cfg)

    X_train = train_df[feature_cols].values
    y_train_cls = train_df["Direction"].values
    y_train_reg = train_df["Pct_Change"].values   # scale‑independent target

    X_val = val_df[feature_cols].values
    y_val_cls = val_df["Direction"].values
    y_val_reg = val_df["Pct_Change"].values

    # Save feature column list and test set for evaluate.py
    _save_artifact(feature_cols, "feature_columns.pkl", save_dir)
    test_df.to_csv(os.path.join(save_dir, "test_set.csv"))
    val_df.to_csv(os.path.join(save_dir, "val_set.csv"))
    print(f"  Saved feature_columns.pkl and test_set.csv")

    results = {}

    # Start a parent MLflow run for the whole training process
    with mlflow.start_run(run_name="Training_Pipeline"):

        # --- Model 1: Logistic Regression ---
        results["logistic"] = train_logistic_regression(
            X_train, y_train_cls, X_val, y_val_cls,
            feature_cols, cfg, save_dir
        )

        # --- Model 2: XGBoost ---
        results["xgboost"] = train_xgboost(
            X_train, y_train_cls, y_train_reg,
            X_val, y_val_cls, y_val_reg,
            feature_cols, cfg, save_dir
        )

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Val Acc':>10}")
    print(f"  {'-'*25} {'-'*10}")
    for name, res in results.items():
        if res:
            va = res.get("val_acc", 0)
            print(f"  {name:<25} {va:>10.4f}")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_all()
