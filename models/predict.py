"""
Load saved models and return predictions for a given ticker.

Primary interface:
    predict_polynomial_regression(feature_vector) → predicted change and next close
    predict_xgboost(feature_vector)  → direction, confidence, price

Models are loaded lazily and cached.
"""

import os
import sys
import pickle
import warnings

import numpy as np
import yaml
import mlflow

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths & MLflow setup
# ---------------------------------------------------------------------------
# Set tracking URI (will be overridden by environment variable in production)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_artifact(filename: str, save_dir: str):
    """Load a pickle file from the given directory."""
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Have you run  python models/train.py  yet?"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Lazy-loaded model cache
# ---------------------------------------------------------------------------
_cache: dict = {}


def _get_save_dir() -> str:
    cfg = _load_config()
    return os.path.join(_PROJECT_ROOT, cfg["output"]["model_save_dir"])


def _ensure_xgboost_loaded():
    """
    Load XGBoost classifier, regressor, scaler, and feature columns.
    All are loaded from local pickle files (not MLflow registry) to avoid path issues.
    """
    if "xgb_classifier" in _cache:
        return

    save_dir = _get_save_dir()

    # Load classifier, regressor, scaler, and feature columns from local files
    _cache["xgb_classifier"] = _load_artifact("xgboost_classifier.pkl", save_dir)
    _cache["xgb_regressor"] = _load_artifact("xgboost_regressor.pkl", save_dir)
    _cache["xgb_scaler"] = _load_artifact("xgboost_scaler.pkl", save_dir)
    _cache["feature_columns"] = _load_artifact("feature_columns.pkl", save_dir)


def _ensure_polynomial_loaded():
    """Load Polynomial Regression model, scaler, and feature columns."""
    if "poly_regression" in _cache:
        return

    save_dir = _get_save_dir()
    _cache["poly_regression"] = _load_artifact("polynomial_regression.pkl", save_dir)
    _cache["poly_scaler"] = _load_artifact("polynomial_scaler.pkl", save_dir)
    if "feature_columns" not in _cache:
        _cache["feature_columns"] = _load_artifact("feature_columns.pkl", save_dir)


def _ensure_linear_regression_loaded():
    """Load Linear Regression model and scaler into cache."""
    if "lin_regression" in _cache:
        return

    save_dir = _get_save_dir()
    _cache["lin_regression"] = _load_artifact("linear_regression.pkl",        save_dir)
    _cache["lin_scaler"]     = _load_artifact("linear_regression_scaler.pkl", save_dir)
    if "feature_columns" not in _cache:
        _cache["feature_columns"] = _load_artifact("feature_columns.pkl", save_dir)


# ---------------------------------------------------------------------------
# Public API — XGBoost
# ---------------------------------------------------------------------------

def get_feature_columns() -> list[str]:
    """Return the list of feature column names the models expect."""
    if "feature_columns" not in _cache:
        save_dir = _get_save_dir()
        _cache["feature_columns"] = _load_artifact("feature_columns.pkl", save_dir)
    return _cache["feature_columns"]


def predict_polynomial_regression(feature_vector: np.ndarray,
                                  current_close: float | None = None) -> dict:
    """
    Run Polynomial Regression prediction on a single feature vector.

    Returns
    -------
    dict with keys:
        pct_change: float — predicted percentage change
        price     : float — predicted next-day closing price
    """
    _ensure_polynomial_loaded()

    polynomial_model = _cache["poly_regression"]
    scaler = _cache["poly_scaler"]

    X = np.asarray(feature_vector)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_sc = scaler.transform(X)
    pct_change = float(polynomial_model.predict(X_sc)[0])
    if current_close is not None:
        price = current_close * (1 + pct_change / 100)
    else:
        price = pct_change

    return {
        "pct_change": round(pct_change, 4),
        "price": round(price, 2)
    }


def predict_linear_regression(feature_vector: np.ndarray,
                               current_close: float | None = None) -> dict:
    """
    Run Linear Regression prediction on a single feature vector.

    Returns
    -------
    dict with keys:
        pct_change: float — predicted percentage change
        price     : float — predicted next-day closing price
    """
    _ensure_linear_regression_loaded()

    model  = _cache["lin_regression"]
    scaler = _cache["lin_scaler"]

    X = np.asarray(feature_vector)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_sc = scaler.transform(X)
    pct_change = float(model.predict(X_sc)[0])
    if current_close is not None:
        price = current_close * (1 + pct_change / 100)
    else:
        price = pct_change

    return {
        "pct_change": round(pct_change, 4),
        "price": round(price, 2)
    }


def predict_xgboost(feature_vector: np.ndarray,
                    current_close: float | None = None) -> dict:
    """
    Run XGBoost prediction on a single feature vector.

    Parameters
    ----------
    feature_vector : np.ndarray, shape (1, n_features) or (n_features,)
        The engineered features for the most recent trading day.
    current_close : float or None
        Today's closing price.  Required to convert the regressor's
        percentage-change prediction back into a dollar price.

    Returns
    -------
    dict with keys:
        direction   : str   — "UP" or "DOWN"
        confidence  : float — probability (0–100 %)
        price       : float — predicted next-day closing price
    """
    _ensure_xgboost_loaded()

    classifier = _cache["xgb_classifier"]
    regressor = _cache["xgb_regressor"]
    scaler = _cache["xgb_scaler"]
    feature_columns = _cache["feature_columns"]
    cfg = _load_config()
    top_n = cfg["output"]["top_features_to_display"]

    # Reshape to 2-D if needed
    X = np.asarray(feature_vector)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    X_sc = scaler.transform(X)

    # Classification
    probability = classifier.predict_proba(X_sc)[0]  # [prob_0, prob_1]
    direction_idx = int(np.argmax(probability))
    direction = "UP" if direction_idx == 1 else "DOWN"
    confidence = probability[direction_idx] * 100

    # Regression — model predicts percentage change, convert to price
    pct_change = float(regressor.predict(X_sc)[0])
    if current_close is not None:
        price = current_close * (1 + pct_change / 100)
    else:
        # Fallback: return raw pct change if current_close not provided
        price = pct_change

    return {
        "direction": direction,
        "confidence": round(confidence, 2),
        "price": round(price, 2)
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[predict] Loading models to verify…")
    try:
        cols = get_feature_columns()
        print(f"  Feature columns ({len(cols)}): {cols[:5]}…")
        print("  XGBoost models loaded OK.")
    except Exception as e:
        print(f"  ERROR: {e}")
