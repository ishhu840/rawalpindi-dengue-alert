# Methodology Draft

## Study Objective

The objective is to develop a dengue early-warning app for Rawalpindi that forecasts expected reported dengue cases and identifies high-risk Union Councils one or more weeks ahead.

## Outcome Definition

The primary outcome should be weekly reported dengue cases in Rawalpindi.

For UC-level alerts, the secondary outcome should be weekly reported dengue cases per Rawalpindi Union Council, where patient records can be reliably resolved to a UC.

## Why Weather Lags Are Required

Dengue transmission responds to environmental conditions with delay. Rainfall may create breeding containers and larval habitats, but reported human cases appear only after mosquito development, adult mosquito infection, human infection, incubation, care-seeking, testing, confirmation, and reporting.

For this reason, meteorological variables should be represented as lagged and rolling-window predictors.

Candidate lag ranges:

- weekly dengue cases: 1-8 weeks
- rainfall: 1-8 weeks
- temperature: 1-8 weeks
- humidity: 1-8 weeks
- pressure and wind speed: 1-4 weeks

Candidate rolling windows:

- cumulative rainfall over 2, 4, 6, and 8 weeks
- mean temperature over 2, 4, and 6 weeks
- mean humidity over 2, 4, and 6 weeks
- recent dengue incidence over 2, 4, and 8 weeks

## Candidate Predictors

### Epidemiological

- weekly dengue cases
- lagged dengue cases
- rolling average dengue cases
- outbreak momentum
- previous year same-week cases

### Meteorological

- mean temperature
- maximum temperature
- minimum temperature
- humidity
- rainfall
- rainy days
- atmospheric pressure
- wind speed
- thermal range if available

### Seasonal

- ISO week
- month
- monsoon indicator
- cyclical week encoding using sine and cosine

### Spatial

- UC historical burden
- tehsil
- neighboring UC burden
- distance from prior hotspots
- spatially smoothed recent risk

### Demographic and Environmental if Available

- UC population
- population density
- urban/rural category
- vegetation or built-up indicators
- water bodies or drainage features

## Modeling Strategy

### City-Level Forecast

Train models to predict total weekly Rawalpindi reported dengue cases.

Recommended candidate models:

- seasonal naive baseline
- SARIMAX
- Negative Binomial regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- ensemble of top-performing models

LSTM or deep learning can be tested, but should not be selected unless it wins under strict time-based validation.

### UC-Level Forecast

Use a two-stage model:

1. Forecast total Rawalpindi cases.
2. Allocate expected cases to UCs using UC-level risk shares.

This avoids overfitting sparse UC-week counts while still producing actionable area-wise alerts.

Candidate UC risk-share formula:

`risk_score_uc = weighted_historical_burden + weighted_recent_uc_activity + weighted_neighbor_activity + seasonal_uc_factor`

Then:

`expected_cases_uc = total_forecast_rawalpindi * risk_score_uc / sum(all_uc_risk_scores)`

## Alert Categories

UC alert levels should use both absolute expected cases and comparison with each UC's historical baseline.

Example:

- Green: low expected risk
- Yellow: above usual baseline or early rise
- Orange: high expected cases or rapid growth
- Red: outbreak-level risk or severe increase from baseline

Final thresholds should be calibrated from historical UC distributions.

## Reporting Fraction

The available data represent reported or registered dengue cases, not all true infections.

The main model should forecast expected reported cases.

If the app includes estimated true burden, it should use scenario assumptions:

- reported forecast
- conservative underreporting adjustment
- moderate underreporting adjustment
- high underreporting adjustment

These should be labeled clearly as assumptions.

## Validation

Use rolling-origin time-series validation:

- train through 2018, test 2019
- train through 2019, test 2020
- train through 2020, test 2021
- train through 2021, test 2022
- train through 2022, test 2023
- train through 2023, test 2024

Use 2025 only as final external validation if it remains unseen.

Evaluation metrics:

- MAE
- RMSE
- MAPE or sMAPE
- R2
- alert-level precision and recall
- peak timing error
- calibration of uncertainty intervals

## Forecast Horizon

Recommended app horizons:

- next 1 week: main operational alert
- next 2 weeks: useful secondary alert
- next 1-3 months: scenario/risk outlook, not exact case prediction

Because weather effects are lagged, next-week forecasts depend heavily on observed weather from the previous 1-8 weeks. Forecast weather becomes more important for 2-3 week forecasts.

