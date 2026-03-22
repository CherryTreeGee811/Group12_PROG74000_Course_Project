"""
Fetch OHLCV price data and VIX index from yfinance.

Two modes:
  - training:   period="max" for full history
  - prediction: last N trading days (default 300) for live inference

Results are cached as CSV in data/cache/ to avoid redundant API calls.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CACHE_DIR = os.path.join(_THIS_DIR, "cache")
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

# Ensure cache directory exists
os.makedirs(_CACHE_DIR, exist_ok=True)


def _load_config() -> dict:
    """Load config.yaml and return the full dict."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, mode: str) -> str:
    """Return the path to the cached CSV for a given ticker and mode."""
    safe_ticker = ticker.replace("^", "_").upper()
    return os.path.join(_CACHE_DIR, f"{safe_ticker}_{mode}.csv")


def _cache_is_fresh(path: str) -> bool:
    """Return True if the cached file exists and was written today."""
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return mtime.date() == datetime.today().date()


def _download_ticker(ticker: str, period: str | None = None,
                     start: str | None = None) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker via yfinance.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    try:
        tk = yf.Ticker(ticker)
        if period:
            df = tk.history(period=period, auto_adjust=True)
        else:
            df = tk.history(start=start, auto_adjust=True)
    except Exception as exc:
        print(f"[fetch_price] ERROR downloading {ticker}: {exc}")
        return pd.DataFrame()

    if df is None or df.empty:
        print(f"[fetch_price] WARNING: No data returned for {ticker}.")
        return pd.DataFrame()

    # Keep only the columns we need
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # yfinance returns a DatetimeIndex — normalise to date-only
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"

    return df


def _fetch_vix(period: str | None = None,
               start: str | None = None) -> pd.Series:
    """
    Download VIX closing prices and return as a Series indexed by Date.
    """
    vix_df = _download_ticker("^VIX", period=period, start=start)
    if vix_df.empty:
        return pd.Series(dtype="float64", name="VIX_Close")
    series = vix_df["Close"].rename("VIX_Close")
    return series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_price_data(ticker: str, mode: str = "training",
                     use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch OHLCV + VIX data for *ticker*.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL", "MSFT").
    mode : str
        "training"   → download full history (period="max").
        "prediction" → download the last N calendar days so that
                       at least 300 *trading* days are available.
    use_cache : bool
        If True, return cached CSV when it was written today.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume, VIX_Close
        Index: DatetimeIndex named "Date"
    """
    cfg = _load_config()
    cache_file = _cache_path(ticker, mode)

    # --- Try cache first ---
    if use_cache and _cache_is_fresh(cache_file):
        print(f"[fetch_price] Using today's cache for {ticker} ({mode}).")
        df = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        return df

    # --- Determine download parameters ---
    if mode == "training":
        period = cfg["data"]["training_history_period"]  # "max"
        start = None
    else:
        # For prediction: fetch enough calendar days to cover ≥300 trading days.
        # Trading days ≈ 252/year, so 300 trading days ≈ 430 calendar days.
        # Add a generous buffer.
        lookback = cfg["data"]["prediction_lookback_days"]
        calendar_days = int(lookback * 1.7)  # ~510 calendar days
        start_date = (datetime.today() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")
        period = None
        start = start_date

    # --- Download stock data ---
    print(f"[fetch_price] Downloading {ticker} ({mode})...")
    stock_df = _download_ticker(ticker, period=period, start=start)
    if stock_df.empty:
        print(f"[fetch_price] Failed to fetch data for {ticker}.")
        return pd.DataFrame()

    # --- Download VIX data with matching range ---
    if mode == "training":
        vix = _fetch_vix(period=period)
    else:
        vix = _fetch_vix(start=start)

    # --- Merge on Date ---
    if not vix.empty:
        stock_df = stock_df.join(vix, how="left")
        # Forward-fill VIX for any days where VIX traded but the stock didn't,
        # or vice-versa (holidays in different markets).
        stock_df["VIX_Close"] = stock_df["VIX_Close"].ffill()
    else:
        stock_df["VIX_Close"] = 0.0
        print("[fetch_price] WARNING: VIX data unavailable; filled with 0.")

    # Drop rows where essential columns are NaN (e.g. very first rows)
    stock_df.dropna(subset=["Close"], inplace=True)

    # --- Save to cache ---
    stock_df.to_csv(cache_file)
    print(f"[fetch_price] Cached → {cache_file}  ({len(stock_df)} rows)")

    return stock_df


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage: python fetch_price.py AAPL training
    #        python fetch_price.py NVDA prediction
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    mode_arg = sys.argv[2] if len(sys.argv) > 2 else "prediction"

    data = fetch_price_data(ticker_arg, mode=mode_arg)
    if not data.empty:
        print(f"\nShape: {data.shape}")
        print(data.head())
        print("...\n")
        print(data.tail())
    else:
        print("No data returned.")
