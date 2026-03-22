# Stock Price Prediction Engine

A machine learning system that predicts whether a stock's price will move **UP or DOWN** the next trading day, along with an approximate closing price. It combines technical indicators from historical OHLCV data, VIX volatility index — all running locally with zero paid API costs.

---

## Architecture

```
                        ┌──────────────────────────────────────────┐
                        │           DATA SOURCES                   │
                        └──────────────────────────────────────────┘

    yfinance (OHLCV)                                    yfinance (^VIX)
         │                                                     │
         ▼                                                     │
  ┌─────────────┐                                              │
  │ Technical   │                                              │
  │ Indicators  │                                              │
  │ SMA, EMA,   │                                              │
  │ RSI, MACD,  │                                              │
  │ BB, ATR,    │                                              │
  │ OBV, Lags   │                                              │
  └──────┬──────┘                                              │
         │                                                     │
         ▼                                                     ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                    COMBINED FEATURE VECTOR                     │
  │   30+ features per trading day                                 │
  └────────────────────────────┬───────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
     │   Logistic   │ │   XGBoost    │ │    LSTM      │
     │  Regression  │ │  Classifier  │ │   Neural     │
     │  (baseline)  │ │  + Regressor │ │   Network    │
     └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
            │                │                │
            ▼                ▼                ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                       PREDICTIONS                              │
  │   Direction: UP / DOWN    |    Next-Day Price    |   Confidence │
  └────────────────────────────────────────────────────────────────┘
```

---

## Models

| Model | Type | Purpose |
|---|---|---|
| **Logistic Regression** | Classification | Baseline — every other model must beat this |
| **XGBoost** | Classification + Regression | Primary model — direction prediction + price estimate |
| **LSTM** | Classification | Deep learning — learns from 60-day sequences of features |

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the training dataset

```bash
python features/build_dataset.py
```

This fetches historical data for 3 tickers, computes all indicators, and saves `data/cache/training_dataset.csv`.

### 3. Train all models

```bash
python models/train.py
```

Trains Logistic Regression, XGBoost (with GridSearchCV + TimeSeriesSplit), and LSTM. All models and scalers are saved to `models/saved/`.

### 4. Evaluate models

```bash
python models/evaluate.py
```

Prints accuracy, precision, recall, F1, directional accuracy, MAE, RMSE, MAPE, and generates three charts in `models/saved/`:
- `predicted_vs_actual.png` — XGBoost predicted vs actual closing prices
- `feature_importance.png` — top 20 features driving XGBoost decisions
- `model_comparison.png` — side-by-side metric comparison table

### 5. Run a prediction

```bash
python app/main.py AAPL
```

Or simply run `python app/main.py` and enter the ticker when prompted.

### One-command training pipeline

```bash
python run_pipeline.py
```

Runs steps 3 → 4 → 5 in sequence.

---

## Project Structure

```
stock-predictor/
├── .env                         ← API key (user fills this)
├── .env.example                 ← template
├── .gitignore                   ← excludes generated data/models
├── config.yaml                  ← all tunable parameters
├── requirements.txt             ← pinned Python dependencies
├── run_pipeline.py              ← one-command: build → train → evaluate
├── README.md                    ← this file
│
├── data/
│   ├── fetch_price.py           ← OHLCV + VIX from yfinance
│   ├── fetch_news.py            ← headlines from NewsAPI
│   └── cache/                   ← cached CSVs (git-ignored)
│
├── features/
│   ├── technical_indicators.py  ← SMA, EMA, RSI, MACD, BB, ATR, OBV, lags
│   ├── sentiment.py             ← FinBERT sentiment scoring
│   └── build_dataset.py         ← merges everything into training CSV
│
├── models/
│   ├── train.py                 ← trains all 3 models
│   ├── evaluate.py              ← metrics + charts
│   ├── predict.py               ← loads models, returns predictions
│   └── saved/                   ← .pkl, .h5 model files (git-ignored)
│
├── notebooks/
│   └── exploration.ipynb        ← optional data exploration
│
└── app/
    └── main.py                  ← CLI entry point
```

---

## Known Limitations

1. **Stock prices are partially random.** This model targets directional accuracy above 50%, not perfect price prediction.

2. **The model cannot react to brand-new event types it has never seen in training.** VIX and sentiment scores serve as indirect proxies for geopolitical and macro shocks.

3. **Look-ahead bias is deliberately avoided throughout.** All features for day T use only data from day T and earlier (Close lags use T−1 through T−10).

4. **This is not financial advice and should not be used for real trading decisions.**

---
# Group12_PROG74000_Course_Project
