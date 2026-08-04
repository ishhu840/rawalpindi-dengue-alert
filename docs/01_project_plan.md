# Project Plan: Rawalpindi Dengue Early Alert App

## Goal

Create a weekly dengue early alert system for Rawalpindi that combines historical dengue case data, UC-level location patterns, recent weather, forecast weather, and validated machine learning models.

The system should support research use and public-health decision support.

## Source Folders

### Weekly AI Forecast Study

Folder:

`/Users/ishtiaq/Desktop/Phd Study Coding worok /Study # 1 Weekly_AI_Study_Rawapidni_Dengue_Forcast`

Useful contents:

- prepared weekly train/test/validation data
- existing Random Forest, XGBoost, and LSTM scripts
- saved models and scalers
- 2025 validation outputs
- feature column list

Important observation:

Several scripts use older absolute paths. Before rerunning them, paths should be refactored to use project-relative paths.

### Area-Wise Mapping Study

Folder:

`/Users/ishtiaq/Desktop/Phd Study Coding worok /Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise`

Useful contents:

- patient-level confirmed dengue workbook
- cleaned UC-mapped workbook
- Rawalpindi UC boundary GeoJSON
- existing map HTML files
- `build_uc_map.py`
- `map_data.json`, `map_data.js`
- `uc_counts.js`, `uc_hotspots.js`

Important observation:

`rawalpindi_uc_heatmap.html` is the strongest existing starting point for area-wise UC polygons. `dengue_map.html` is useful for bubble/heatmap visualization, but not the final UC area alert map.

## Development Phases

### Phase 1: Data Audit

Tasks:

- verify date columns and week definitions
- verify Rawalpindi case totals by year and week
- identify whether UC-level data are available for 2013-2024, 2025, or only partial years
- check missing Latitude, Longitude, UC, Location, Tehsil
- create a clean data dictionary
- decide whether forecast target is onset date, confirmation date, entry date, or reporting date

Recommended target:

Use weekly reported/confirmed cases. If onset date is reliable, compare onset-date aggregation against confirmation/reporting-date aggregation.

### Phase 2: City-Level Weekly Forecast Dataset

Create one weekly Rawalpindi dataset with:

- Year
- ISO Week
- Week start date
- reported dengue cases
- weather variables
- lagged dengue cases
- lagged weather variables
- rolling weather variables
- seasonality features

Candidate features:

- cases lag 1-8 weeks
- rolling cases 2, 4, 8 weeks
- rainfall lag 1-8 weeks
- rainfall rolling sum 2, 4, 6, 8 weeks
- temperature mean lag 1-8 weeks
- temperature suitability days
- humidity lag 1-8 weeks
- pressure lag 1-4 weeks
- wind speed lag 1-4 weeks
- week of year
- month
- monsoon indicator
- outbreak momentum
- previous year same-week cases if available

### Phase 3: Model Benchmarking

Do not use random train/test splitting as the main validation. Dengue forecasting is temporal, so validation must respect time order.

Recommended validation:

- train 2013-2018, test 2019
- train 2013-2019, test 2020
- train 2013-2020, test 2021
- train 2013-2021, test 2022
- train 2013-2022, test 2023
- train 2013-2023, test 2024
- final external check on 2025 if available

Models to compare:

- seasonal naive baseline
- SARIMAX
- Negative Binomial regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- ensemble of best tree models
- LSTM or temporal deep model only if data volume and validation justify it

Metrics:

- MAE
- RMSE
- MAPE or sMAPE
- R2
- peak-week timing error
- outbreak alert precision/recall
- calibration of prediction intervals

### Phase 4: UC-Level Risk Allocation

Create weekly UC counts for Rawalpindi from the mapped patient-level file.

UC allocation model options:

- historical proportional allocation by UC
- season-specific UC proportions
- exponentially weighted recent UC activity
- multinomial/Dirichlet allocation model
- tree model predicting UC share
- spatial smoothing using neighboring UCs

Recommended first version:

Combine city-level forecast with UC risk scores:

`expected_uc_cases = expected_city_cases * uc_risk_share`

Where `uc_risk_share` is based on historical UC burden, recent UC burden, tehsil patterns, and neighboring risk.

### Phase 5: Weather Pipeline

Use API weather data for both training consistency and weekly refresh.

Recommended first API:

Open-Meteo, because it provides historical weather and forecast data without requiring an API key.

Use:

- historical weather for backfilling and consistency checks
- recent past weather for last 8-12 weeks
- forecast weather for next 7-16 days

Important:

For next-week dengue forecasting, recent past weather is likely more important than future weather because dengue response is delayed.

### Phase 6: App Build

The app should include:

- dashboard summary
- UC alert map
- forecast table
- alert explanation panel
- model validation panel
- weekly refresh script

Initial app type:

Local web app first. Once methodology and outputs are validated, it can be deployed.

## Deliverables

Minimum viable research app:

- validated model comparison report
- trained final model
- weekly forecast generator
- UC-level forecast table
- interactive Rawalpindi UC alert map
- documentation of methodology and limitations

