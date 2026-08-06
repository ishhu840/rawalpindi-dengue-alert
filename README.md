# Rawalpindi Dengue Early Alert App

This folder is for the new web-based dengue early alert app. It uses the two existing folders only as source material:

- `Study # 1 Weekly_AI_Study_Rawapidni_Dengue_Forcast`
- `Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise`

The original folders should remain unchanged. New model training, validation, app code, documentation, and generated outputs should live here.

## Objective

Build a research-grade web app that estimates upcoming dengue risk for Rawalpindi and shows expected weekly dengue cases by Union Council (UC).

The app should answer:

- How many reported dengue cases are expected in Rawalpindi next week?
- Which Rawalpindi UCs are likely to have the highest risk?
- What alert category should each UC receive?
- How uncertain is the forecast?
- Which recent weather and case-history signals are driving the alert?

## Core Scientific Framing

Dengue cases are not modeled as an immediate response to same-day weather. Dengue risk is modeled as a delayed response to prior meteorological suitability and recent transmission intensity.

Weather can influence dengue after a delay because rainfall creates breeding sites, temperature affects mosquito development and viral replication, and human case registration occurs after infection, incubation, illness onset, diagnosis, and reporting.

Therefore, the model should use lagged and rolling weather features from previous weeks, not only current weather.

## Recommended System Design

Use a two-stage forecasting system:

1. City-level weekly forecast
   - Predict total expected reported dengue cases for Rawalpindi.
   - Use weekly dengue history, lagged weather, rolling weather, seasonality, and outbreak momentum.

2. UC-level spatial allocation and alert model
   - Estimate each Rawalpindi UC's share of expected cases.
   - Use historical UC burden, recent UC activity, tehsil/UC spatial patterns, neighboring UC influence, and available demographic/context features.
   - Distribute the city-level forecast into UC-level expected cases.

This is more defensible than trying to forecast every UC independently when many UCs have sparse weekly records.

## Initial Folder Structure

- `docs/`
  - methodology and planning documents

Recommended future folders:

- `data_raw/`
- `data_processed/`
- `notebooks/`
- `src/`
- `models/`
- `outputs/`
- `app/`
- `reports/`

## Current Best Model Direction

The existing study shows XGBoost performed best on the available 2025 Rawalpindi validation window. However, the final app should not blindly reuse that model.

The next step is to retrain and compare:

- XGBoost
- LightGBM
- CatBoost
- Random Forest baseline
- SARIMAX or Negative Binomial baseline
- optional LSTM or temporal deep learning only if it beats simpler models under proper time-based validation

The final app should use the best validated model or a small ensemble.

## App Output

The first useful version should show:

- Rawalpindi total expected cases for next week
- UC-wise expected cases
- UC alert levels: green, yellow, orange, red
- interactive Rawalpindi UC map
- top-risk UC table
- lower/expected/upper forecast range
- recent rainfall, temperature, humidity, and lag-risk summary
- model validation summary

## Current Working Prototype

The first local prototype has been created in `app/`.

It includes:

- a Rawalpindi forecast summary
- UC polygon alert map
- highest-risk UC table
- four-week forecast cards
- model comparison panel
- generated forecast JSON
- generated UC forecast GeoJSON

The model/output generator is:

`src/build_alert_outputs.py`

It currently benchmarks available scikit-learn models using rolling-origin year validation and writes outputs to:

- `data/latest_forecast.json`
- `data/rawalpindi_uc_forecast.geojson`
- `reports/model_validation_summary.csv`
- `reports/rolling_origin_validation.csv`
- `reports/external_2025_validation.json`

Current first-pass selected model:

Gradient Boosting.

## Weekly Surveillance Input

`data/recent_cases.csv` is how observed dengue counts reach the model:

```csv
year,week,cases
2026,30,14
2026,31,22
```

`year` and `week` are ISO. This matters more than any other input — `Cases_Lag_1w`
alone carries about 95% of the model's feature importance. Resolution order for
each lag is:

1. an observed row in `recent_cases.csv`
2. the model's own nowcast, for weeks after the last observation
3. historical seasonal median, as a last resort

With the file empty the app still runs, but the forecast is a seasonal average
rather than an outbreak signal, and the page says so. If the newest row is more
than 8 weeks old the nowcast chain is skipped and the app falls back to
seasonal history.

## How To Run Locally

From this folder:

```bash
python src/update_live_forecast.py     # fetch weather, refresh forecast
python -m http.server 8765
```

Then open:

`http://127.0.0.1:8765/`

`src/build_alert_outputs.py` re-runs the full model benchmark, but needs the two
sibling study folders and so cannot run in CI.

## UC Alert Levels

UC colours come from expected reported cases for the forecast week, using
thresholds calibrated from the historical transmission-season distribution of
allocated UC-week values:

| Level | Expected cases |
|---|---|
| Red | ≥ 8 |
| Orange | ≥ 3 |
| Yellow | ≥ 0.5 |
| Green | below 0.5 |

Historical burden sets each UC's share of the city forecast, but no longer sets
the colour by itself. It previously did, which pinned the high-burden UCs to Red
year-round and left the map unable to respond to the forecast at all.

## Important Note About Reporting Fraction

The model should first forecast reported dengue cases, because the available data are reported/registered cases.

If true infections are needed, the app can include scenario multipliers such as conservative, moderate, and high underreporting assumptions. These should be clearly labeled as assumptions, not measured truth.
