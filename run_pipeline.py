"""
Merge price features into a single training DataFrame.

No sentiment column is added – only technical indicators are used.
"""

import os
import sys

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")
_CACHE_DIR = os.path.join(_PROJECT_ROOT, "data", "cache")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.fetch_price import fetch_price_data
from features.technical_indicators import compute_indicators


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target columns:
      - Direction: 1 if next day Close > today Close, else 0
      - Next_Close: actual next-day closing price
    Then drop the last row (no target available).
    """
    df = df.copy()
    df["Next_Close"] = df["Close"].shift(-1)
    df["Direction"] = (df["Next_Close"] > df["Close"]).astype(int)
    df["Pct_Change"] = (df["Next_Close"] - df["Close"]) / df["Close"] * 100
    df = df.iloc[:-1]
    return df


def build_training_dataset(save: bool = True) -> pd.DataFrame:
    """
    Build the combined training dataset for all tickers in config.
    No sentiment column is included.
    """
    cfg = _load_config()
    tickers = cfg["data"]["training_tickers"]

    all_frames: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers, 1):
        print(f"\n{'='*60}")
        print(f"[build_dataset] Processing {ticker}  ({i}/{len(tickers)})")
        print(f"{'='*60}")

        # 1. Fetch full-history price data + VIX
        price_df = fetch_price_data(ticker, mode="training")
        if price_df.empty:
            print(f"[build_dataset] Skipping {ticker} — no price data.")
            continue

        # 2. Compute all technical indicators
        feat_df = compute_indicators(price_df, drop_na=True)
        if feat_df.empty:
            print(f"[build_dataset] Skipping {ticker} — no rows after "
                  "indicator warm-up.")
            continue

        # 3. Target columns + drop last row
        feat_df = _add_targets(feat_df)

        # 4. Tag with ticker symbol
        feat_df["Ticker"] = ticker

        print(f"[build_dataset] {ticker}: {len(feat_df)} rows ready.")
        all_frames.append(feat_df)

    if not all_frames:
        print("[build_dataset] ERROR: No data collected for any ticker.")
        return pd.DataFrame()

    combined = pd.concat(all_frames, axis=0)
    combined.sort_index(inplace=True)

    print(f"\n{'='*60}")
    print(f"[build_dataset] Combined dataset: {combined.shape[0]} rows, "
          f"{combined.shape[1]} columns")
    print(f"  Tickers: {combined['Ticker'].unique().tolist()}")
    print(f"  Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"  Direction balance: "
          f"{combined['Direction'].value_counts().to_dict()}")
    print(f"{'='*60}")

    # Save
    if save:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        out_path = os.path.join(_CACHE_DIR, "training_dataset.csv")
        combined.to_csv(out_path)
        print(f"[build_dataset] Saved → {out_path}")

    return combined


if __name__ == "__main__":
    dataset = build_training_dataset(save=True)
    if not dataset.empty:
        print(f"\nFinal shape : {dataset.shape}")
        print(f"Columns     : {list(dataset.columns)}")
        print(f"\nSample (last 3 rows):\n{dataset.tail(3)}")
    else:
        print("Dataset is empty.")
