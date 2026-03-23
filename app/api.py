"""
Flask API for Stock Price Prediction.

Usage:
    export FLASK_APP=app/api.py
    flask run --host=0.0.0.0 --port=5000

Then send GET request:
    curl "http://localhost:5000/predict?ticker=AAPL"
"""

import os
import sys
from datetime import datetime
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# Path setup – make project root importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.fetch_price import fetch_price_data
from features.technical_indicators import compute_indicators
from models.predict import (
    predict_xgboost,
    get_feature_columns,
)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Load models once at startup (optional – they are lazy‑loaded anyway)
print("Loading models...", file=sys.stderr)
try:
    _ = get_feature_columns()          # triggers XGBoost loading
    print("XGBoost models loaded.", file=sys.stderr)
except Exception as e:
    print(f"Error loading XGBoost: {e}", file=sys.stderr)

print("API ready.", file=sys.stderr)


@app.route('/predict', methods=['GET'])
def predict():
    """GET /predict?ticker=AAPL  → JSON prediction."""
    ticker = request.args.get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({"error": "Missing ticker parameter"}), 400

    # ----------------------------------------------------------------------
    # 1. Fetch price data
    # ----------------------------------------------------------------------
    try:
        price_df = fetch_price_data(ticker, mode="prediction")
    except Exception as e:
        return jsonify({"error": f"Failed to fetch price data: {e}"}), 500

    if price_df.empty:
        return jsonify({"error": f"No price data found for {ticker}"}), 404

    current_close = round(float(price_df["Close"].iloc[-1]), 2)

    # ----------------------------------------------------------------------
    # 2. Compute technical indicators
    # ----------------------------------------------------------------------
    try:
        feature_df = compute_indicators(price_df, drop_na=True)
    except Exception as e:
        return jsonify({"error": f"Failed to compute indicators: {e}"}), 500

    if feature_df.empty:
        return jsonify({"error": "Not enough data to compute indicators"}), 422

    # ----------------------------------------------------------------------
    # 3. Build feature vector
    # ----------------------------------------------------------------------
    try:
        feature_columns = get_feature_columns()
    except Exception as e:
        return jsonify({"error": f"Failed to load feature columns: {e}"}), 500

    missing = [column for column in feature_columns if columm not in feature_df.columns]
    if missing:
        return jsonify({"error": f"Missing feature columns: {missing}"}), 500

    latest_row = feature_df[feature_columns].iloc[[-1]].values
    full_matrix = feature_df[feature_columns].values

    # ----------------------------------------------------------------------
    # 4. Run predictions
    # ----------------------------------------------------------------------
    try:
        xgb_result = predict_xgboost(latest_row, current_close=current_close)
    except Exception as e:
        return jsonify({"error": f"XGBoost prediction failed: {e}"}), 500


    # ----------------------------------------------------------------------
    # 5. Build response with explicit type conversion
    # ----------------------------------------------------------------------
    response = {
        "ticker": ticker,
        "date": datetime.today().strftime("%Y-%m-%d"),
        "current_close": float(current_close),
        "xgboost": {
            "direction": str(xgb_result["direction"]),
            "confidence": float(xgb_result["confidence"]),
            "predicted_next_close": float(xgb_result["price"]),
            "top_features": [
                {
                    "feature": str(name),
                    "importance": float(imp)
                }
                for name, imp in xgb_result["top_features"]
            ]
        }
    }

    return jsonify(response)


@app.route('/health', methods=['GET'])
def health():
    """Simple health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
