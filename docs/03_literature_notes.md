# Literature Notes for Dengue Forecasting

This document summarizes recent dengue forecasting themes that should guide the Rawalpindi Dengue Early Alert App.

## Main Lessons

Recent dengue early-warning systems commonly combine:

- lagged climate variables
- recent dengue incidence
- seasonality
- spatial dependence
- machine learning or hybrid statistical-ML models
- uncertainty estimates

The most useful models are not necessarily the most complex. The best model should be selected by local time-based validation.

## Relevant Recent Directions

### Lagged Climate Models

Dengue risk is often associated with delayed rainfall, humidity, and temperature effects. Recent Bangladesh and India/Pune work supports climate lags from weeks to months, depending on outcome and temporal aggregation.

For Rawalpindi weekly forecasting, lags from 1-8 weeks should be tested first. Longer lags can be explored for seasonal risk outlooks.

### Tree-Based Machine Learning

XGBoost, Random Forest, CatBoost, and LightGBM are strong candidates for tabular weekly dengue forecasting. They handle nonlinear effects and interactions between rainfall, humidity, temperature, recent cases, and seasonality.

In the existing local study, XGBoost was best on the short 2025 Rawalpindi validation set.

### Deep Learning

LSTM and spatiotemporal deep learning models can perform well in large multi-region datasets, especially when spatial signals are available. However, they can overfit or underperform when data are limited.

For this app, deep learning should be benchmarked, not assumed to be best.

### Hybrid Models

Hybrid approaches combine statistical time-series models with machine learning, for example SARIMA/SARIMAX plus XGBoost residual correction. These can be useful because dengue has strong seasonality plus nonlinear weather effects.

### Spatial and Graph Models

Recent research increasingly uses spatial influence from neighboring regions or mobility. For Rawalpindi, this can be approximated using UC adjacency, tehsil grouping, and historical hotspot proximity.

Graph neural networks may be explored later, but the first defensible version can use spatial smoothing and UC risk allocation.

### Uncertainty and Decision Support

Operational early-warning systems should show uncertainty. A single point forecast can mislead decision makers.

The app should show:

- lower forecast
- expected forecast
- upper forecast
- alert category
- confidence level

## Sources Reviewed

- Forecasting dengue across Brazil with LSTM neural networks: https://pubmed.ncbi.nlm.nih.gov/40075398/
- Climate-driven dengue forecasting in Bangladesh: https://arxiv.org/abs/2604.18642
- When climate variables improve dengue forecasting: https://arxiv.org/abs/2404.05266
- Nature summary on AI-powered dengue outbreak forecasting: https://www.nature.com/articles/d44151-025-00023-3
- Gavi summary on dengue early-warning lead time: https://www.gavi.org/vaccineswork/ai-model-predicts-dengue-outbreaks-two-months-they-start
- Review of dengue forecasting and early warning systems: https://pmc.ncbi.nlm.nih.gov/articles/PMC12810927/

## Takeaway for This App

The recommended first serious model is:

XGBoost, LightGBM, or CatBoost ensemble for city-level weekly forecasts, combined with UC-level spatial risk allocation and calibrated alert categories.

LSTM or graph models should be treated as experimental extensions unless they outperform the tree-based and statistical baselines under rolling-origin validation.

