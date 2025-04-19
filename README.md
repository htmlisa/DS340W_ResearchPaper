# 🏠 National House Price Prediction Using Time-Series XGBoost

Note: This is a research paper made for DS340W course in Penn State

## 📌 Abstract

This research will explore the use of time series machine learning models to forecast national house price trends using macroeconomic indicators. While traditional house price models often rely on property-level features, this study shifts focus to broader economic variables such as Gross Dometic Product (GDP), mortgage rates, employment levels, and inflation that significantly influence housing market dynamics. Building on (3) foundational research papers (Parent Paper, Supporting Paper 1 and Supporting Paper 2. This study integrates advanced methods in explainable boosting, loss function optimization, and interpretable local modeling to enhance both accuracy and transparency.  

We develop a time-aware XGBoost model and evaluate its performance using RMSE, MAE, and R2 metrics. An expanding-window validation approach is applied to maintain temporal integrity, and SHapley Additive exPlanations (SHAP) are used to interpret how each economic factor contributes to price fluctuations over time. Through this paper, readers will gain insight into how machine learning particularly XGBoost can be adapted for time series forecasting, how model interpretability can be preserved using SHAP, and how tailoring loss function can improve real-world relevance. The study ultimately provides a replicable and transparent modeling approach that can aid financial institutions, housing analysts, and policymakers in understanding and anticipating housing market trends. 

# 🏠 National House Price Prediction Using Time-Series XGBoost

This repository contains the code, results, and documentation for a research project exploring the use of time-aware machine learning to predict national house price trends using macroeconomic indicators. The primary model used in this study is XGBoost, enhanced with time-series transformations and interpretability tools such as SHAP.

## 📊 Dataset

- **Source**: [Kaggle - Factors Affecting USA Home Prices](https://www.kaggle.com/code/faryarmemon/factors-affecting-usa-home-prices)
- **Period**: January 1999 – February 2021
- **Target Variable**: House Price Index (HPI)
- **Features**:
  - Gross Domestic Product (GDP)
  - Mortgage Rate
  - Employment Rate
  - Producer Price Index - Residential (PPI_res)
  - Money Supply (M3)
  - Consumer Confidence Index (CCI)
  - Delinquency Rate
  - Housing Credit Availability Index (HCAI)

## 🔧 Methods

- 🧠 **Model**: Time-series-enhanced [XGBoost](https://xgboost.readthedocs.io/)
- 📐 **Feature Engineering**:
  - Lag features
  - Rolling averages (3-month, 6-month)
  - Percentage change calculations
- 📊 **Evaluation Metrics**:
  - RMSE, MAE, R²
  - Mean Percentage Error (MPE)
  - ±10% prediction accuracy
- 🔍 **Interpretability**:
  - SHAP (SHapley Additive Explanations) for global and temporal feature impact
  - Gain-based feature importance analysis

## 📈 Results

| Metric | Train Set | Test Set |
|--------|-----------|----------|
| RMSE   | 0.214     | 39.120   |
| MAE    | 0.156     | 23.699   |
| R²     | 1.000     | -0.123   |
| MPE    | 0.12%     | 9.82%    |
| ±10% Accuracy | 100.00% | 59.29% |

- **Training Accuracy** was excellent, but the **Test performance showed overfitting**, suggesting that macroeconomic features alone struggle to predict recent housing surges (e.g. post-2016).

## 📉 Visuals

### SHAP Impact Over Time
Shows dominant macroeconomic drivers like `m3` and `ppi_res`.

### Feature Importance (Gain)
Highlights top engineered features contributing to XGBoost's decision paths.

### Actual vs Predicted HPI
Demonstrates underperformance on post-2016 data where prices rose steeply.

## ⚠️ Limitations

- The model **overfits** historical data and fails to generalize during periods of economic shocks.
- Static macroeconomic indicators are insufficient to capture **short-term volatility** or external interventions (e.g. stimulus packages).
- XGBoost is not natively time-aware and requires engineered lags, which may miss deep temporal patterns.

## 📌 Future Work

- Integrate hybrid models (e.g., ARIMA + XGBoost or LSTM).
- Add real-time features like Google Trends, sentiment scores, or market activity indicators.
- Explore regional models or multi-level forecasting structures.


