# ML-Stock-price-prediction-project

Comparative Analysis of Machine Learning Models for Stock Price Prediction
This repository contains code and experiments comparing ARIMA, XGBoost, and SVR models for predicting Apple (AAPL) stock prices and returns using 10 years of daily OHLCV data, plus a simple ensemble/strategy derived from these models.​

#Project Goal
Evaluate which model (or combination of models) provides the most accurate and practically useful forecasts for investment decisions, and translate these forecasts into simple, strategy-relevant insights.​

# Data
Asset: AAPL

Horizon: ~10 years of daily OHLCV data from 2015‑12‑09 to present

File: AAPL_10y_OHLCV.csv

Fields: Date, Open, High, Low, Close, Volume

Engineered features (for XGBoost/SVR/ensemble): returns, moving averages, EMAs, MACD, RSI, rolling volatility, and lagged prices/returns.​

# Models and Strategy

**ARIMA (Close price)**:

Daily close, 85/15 train–test, rolling one-step-ahead forecasting.​

Metrics: RMSE 3.99, MAE 2.72, MAPE 1.224%, R² 0.964, accuracy 98.78%.​

Role: Linear baseline and trend-following component.​

**XGBoost (Close price + features)**:

Train on data before 2023, test on 2023–2024 with technical and lag features.​

Metrics: RMSE 4.58, MAE 2.96, MAPE 1.563%, R² 0.973.​

Role: Nonlinear, feature-driven component capturing momentum and complex patterns.​

**SVR (Returns)**:

Linear kernel on engineered return features.​

Metrics: RMSE 0.02047, R² 0.7493, residuals near zero and roughly normal.​

Role: Volatility‑smoothing, risk-aware component emphasizing stable predictions.​

Ensemble / Strategy Layer

Combines ARIMA and XGBoost forecasts (ensemble) and uses SVR return predictions to adjust risk stance.​

# Intended use:

ARIMA for baseline trend.

XGBoost for nonlinear adjustments and momentum.

SVR to moderate exposure during high‑volatility regimes.​

#Model Comparison

All quantitative model comparison is performed between ARIMA, XGBoost, and SVR on the AAPL dataset.​

Diebold–Mariano test on squared errors shows ARIMA has significantly lower expected squared error than XGBoost at the 1% level (mean difference −9.57, t = −2.88, p = 0.0043).​

7‑day forecasts diverge: ARIMA predicts a mild move toward ~280 USD, XGBoost toward ~270 USD, illustrating different risk/return profiles that the ensemble and strategy seek to reconcile.​

# How to Run
bash
git clone https://github.com/<user>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Ensure data/AAPL_10y_OHLCV.csv is present
python scripts/run_arima.py
python scripts/run_xgboost.py
python scripts/run_svr.py

# Roles
Ashutosh Talekar: Data pipeline, ARIMA, hypothesis testing​, Build final ensemble combining all three models

Julia Knox: Data prep, XGBoost, report and documentation

Bhavya Grover: SVR model, create full evaluation (RMSE, MAE, MAPE, R²), final comparison and results
