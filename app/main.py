"""
Entry point: accept a ticker symbol and run the full prediction pipeline.
Sentiment analysis is completely removed.
"""

import os
import sys
from datetime import datetime

import yaml

# ---------------------------------------------------------------------------
# Paths
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

_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_prediction(ticker: str) -> None:
    """Execute the full prediction pipeline for *ticker* and print results."""
    cfg = _load_config()
    ticker = ticker.upper().strip()
    today_str = datetime.today().strftime("%Y-%m-%d")

    print()
    print("=" * 62)
    print(f"  STOCK PRICE PREDICTION ENGINE")
    print(f"  Ticker : {ticker}")
    print(f"  Date   : {today_str}")
    print("=" * 62)

    # Step 1 — Fetch price data
    print("\n[1/4] Fetching price data…")
    price_df = fetch_price_data(ticker, mode="prediction")
    if price_df.empty:
        print(f"\n  ERROR: Could not fetch price data for '{ticker}'.")
        return
    current_close = round(float(price_df["Close"].iloc[-1]), 2)

    # Step 2 — Compute technical indicators
    print("\n[2/4] Computing technical indicators…")
    feat_df = compute_indicators(price_df, drop_na=True)
    if feat_df.empty:
        print("  ERROR: Not enough data to compute indicators.")
        return

    # Step 3 — Build feature vector (no sentiment column)
    print("\n[3/4] Building feature vector…")
    feature_cols = get_feature_columns()
    # Ensure all expected columns exist
    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns: {missing}")
        for c in missing:
            feat_df[c] = 0.0

    latest_row = feat_df[feature_cols].iloc[[-1]].values
    full_matrix = feat_df[feature_cols].values

    # Step 4 — Run predictions
    print("\n[4/4] Running predictions…")
    try:
        xgb_result = predict_xgboost(latest_row, current_close=current_close)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    # Print results (sentiment section removed)
    _print_results(ticker, today_str, current_close, xgb_result)


def _print_results(ticker, today, current_close, xgb):
    """Print a clean, readable CLI summary (no sentiment)."""
    arrow_xgb = "▲" if xgb["direction"] == "UP" else "▼"

    print()
    print("=" * 62)
    print(f"  PREDICTION RESULTS — {ticker}")
    print(f"  Generated: {today}")
    print("=" * 62)

    print(f"""
  Current Close Price    : ${current_close:,.2f}

  ── XGBoost Prediction ────────────────────────────────
     Direction            : {arrow_xgb} {xgb['direction']}
     Confidence           : {xgb['confidence']:.1f}%
     Predicted Next Close : ${xgb['price']:,.2f}""")

    print(f"""
  ── Top Features Driving Prediction ───────────────────""")
    for rank, (name, imp) in enumerate(xgb["top_features"], 1):
        print(f"     {rank}. {name:<25} (importance: {imp:.4f})")

    print(f"""
  ── Disclaimer ────────────────────────────────────────
     This is NOT financial advice. Predictions are based
     on historical patterns and may not reflect future
     market behaviour. Do not use for real trading.
""")
    print("=" * 62)


def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = input("Enter stock ticker symbol (e.g. AAPL): ").strip()
        if not ticker:
            print("No ticker provided. Exiting.")
            return
    run_prediction(ticker)


if __name__ == "__main__":
    main()
