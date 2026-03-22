"""
Compute technical indicators from OHLCV data.

All window sizes are read from config.yaml — nothing is hardcoded.
Every feature for day T is computed using only data from day T and earlier
(no look-ahead bias). After computation, rows with NaN from the warm-up
period are dropped.
"""

import os
import sys

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def _add_sma(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Simple Moving Average of Close for each window."""
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def _add_ema(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Exponential Moving Average of Close for each window."""
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def _add_macd(df: pd.DataFrame, ema_windows: list[int],
              signal_window: int) -> pd.DataFrame:
    """
    MACD = EMA_short - EMA_long
    MACD_Signal = EMA of MACD over *signal_window* days.
    Assumes EMA columns already exist.
    """
    short, long = sorted(ema_windows)  # e.g. 12, 26
    df["MACD"] = df[f"EMA_{short}"] - df[f"EMA_{long}"]
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_window, adjust=False).mean()
    return df


def _add_rsi(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Standard RSI formula over *window* days."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"RSI_{window}"] = 100 - (100 / (1 + rs))
    return df


def _add_bollinger(df: pd.DataFrame, window: int,
                   num_std: int) -> pd.DataFrame:
    """Bollinger Bands: SMA_window ± num_std * std."""
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()

    df["BB_upper"] = sma + num_std * std
    df["BB_lower"] = sma - num_std * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / sma
    return df


def _add_atr(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average True Range over *window* days."""
    high_low = df["High"] - df["Low"]
    high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
    low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_prev_close, low_prev_close],
                           axis=1).max(axis=1)
    df[f"ATR_{window}"] = true_range.rolling(window=window).mean()
    return df


def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """On-Balance Volume (cumulative)."""
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0  # first row has no prior close
    df["OBV"] = (df["Volume"] * direction).cumsum()
    return df


def _add_lag_features(df: pd.DataFrame, lag_close_days: int,
                      lag_volume_days: int) -> pd.DataFrame:
    """Lag features for Close and Volume."""
    for i in range(1, lag_close_days + 1):
        df[f"Lag_Close_{i}"] = df["Close"].shift(i)
    for i in range(1, lag_volume_days + 1):
        df[f"Lag_Volume_{i}"] = df["Volume"].shift(i)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day_of_Week (0=Mon … 4=Fri) and Month (1–12)."""
    df["Day_of_Week"] = df.index.dayofweek   # 0–6; stocks only trade 0–4
    df["Month"] = df.index.month
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame,
                       drop_na: bool = True) -> pd.DataFrame:
    """
    Add all technical indicators to an OHLCV + VIX DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Open, High, Low, Close, Volume, VIX_Close.
        Index must be a DatetimeIndex named "Date".
    drop_na : bool
        If True, drop rows with NaN caused by indicator warm-up.

    Returns
    -------
    pd.DataFrame
        Original columns + all indicator columns. Sorted by Date ascending.
    """
    cfg = _load_config()
    ind = cfg["indicators"]

    # Work on a copy so the caller's DataFrame is not mutated
    df = df.copy()
    df.sort_index(inplace=True)

    # --- Simple Moving Averages ---
    df = _add_sma(df, ind["sma_windows"])

    # --- Exponential Moving Averages ---
    df = _add_ema(df, ind["ema_windows"])

    # --- MACD & Signal ---
    df = _add_macd(df, ind["ema_windows"], ind["macd_signal_window"])

    # --- RSI ---
    df = _add_rsi(df, ind["rsi_window"])

    # --- Bollinger Bands ---
    df = _add_bollinger(df, ind["bollinger_window"], ind["bollinger_std"])

    # --- ATR ---
    df = _add_atr(df, ind["atr_window"])

    # --- OBV ---
    df = _add_obv(df)

    # --- Lag features ---
    df = _add_lag_features(df, ind["lag_close_days"], ind["lag_volume_days"])

    # --- Calendar features ---
    df = _add_calendar_features(df)

    # VIX_Close should already be present from fetch_price.py.
    # If for some reason it isn't, add a zero column.
    if "VIX_Close" not in df.columns:
        df["VIX_Close"] = 0.0

    # --- Drop NaN warm-up rows ---
    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        dropped = before - after
        if dropped:
            print(f"[indicators] Dropped {dropped} warm-up rows "
                  f"({before} → {after}).")

    return df


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage: python technical_indicators.py
    # Fetches AAPL prediction data and prints indicator columns.
    sys.path.insert(0, _PROJECT_ROOT)
    from data.fetch_price import fetch_price_data

    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    raw = fetch_price_data(ticker, mode="prediction")
    if raw.empty:
        print("No price data — cannot compute indicators.")
    else:
        result = compute_indicators(raw)
        print(f"\nShape: {result.shape}")
        print(f"Columns ({len(result.columns)}):\n  {list(result.columns)}")
        print(f"\nLast row:\n{result.iloc[-1]}")
