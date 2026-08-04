# Next Steps

## Immediate Next Step

Before app coding, build the data and validation pipeline.

## Step 1: Data Audit

Produce:

- case count summary by year
- Rawalpindi weekly case series
- UC mapping completeness report
- date-field comparison report
- missing weather report

Decision needed:

Choose the date field for weekly aggregation:

- Entry Date
- Date of onset
- Confirmation Date
- Reporting Date

Recommended:

Compare all available date fields, then use the most epidemiologically meaningful reliable field. If onset date has too much missingness, use confirmation/reporting date.

## Step 2: Weather Rebuild

Use one consistent weather source where possible.

Recommended:

- Open-Meteo historical weather for model backfill
- Open-Meteo forecast for live forecast refresh

Rawalpindi approximate coordinates:

- latitude: 33.5651
- longitude: 73.0169

## Step 3: Feature Engineering

Create weekly model features:

- lagged cases
- lagged weather
- rolling weather
- seasonal features
- outbreak momentum features

## Step 4: Model Benchmark

Run rolling-origin validation for 2019-2024.

Compare:

- seasonal naive
- SARIMAX
- Negative Binomial
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- ensemble

## Step 5: External 2025 Validation

Use 2025 as an external test if data are not used during training.

Check:

- total predicted vs actual
- weekly trend
- peak timing
- alert usefulness

## Step 6: UC Allocation

Generate weekly UC-level historical counts.

Build UC risk scores using:

- historical burden
- recent UC activity
- tehsil patterns
- neighboring UC smoothing
- seasonal hotspot behavior

## Step 7: App Prototype

Build local app:

- dashboard
- UC map
- top-risk table
- methodology panel
- model validation view

## Step 8: Weekly Operational Script

Create one command:

`python src/run_weekly_alert.py`

Expected behavior:

- fetch weather
- build latest features
- predict total cases
- allocate to UCs
- write frontend JSON

