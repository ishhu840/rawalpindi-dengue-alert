# App Requirements

## Product Purpose

The app should support weekly dengue preparedness in Rawalpindi by showing expected reported cases and UC-level alert areas.

## Primary Users

- public health officers
- dengue surveillance teams
- field response teams
- researchers
- thesis supervisors/reviewers

## Main Views

### 1. Dashboard

Shows:

- current forecast week
- expected Rawalpindi cases
- lower and upper range
- change compared with previous week
- number of red/orange/yellow/green UCs
- recent rainfall and weather-risk summary

### 2. UC Alert Map

Interactive Rawalpindi map with UC polygons.

Each UC should show:

- alert level color
- expected cases
- historical baseline
- recent cases if available
- tehsil
- model confidence

Recommended colors:

- green: low
- yellow: elevated
- orange: high
- red: outbreak alert

### 3. Top-Risk UCs Table

Columns:

- rank
- UC
- tehsil
- expected cases
- alert level
- recent trend
- historical percentile

### 4. Forecast Details

Shows:

- weather lags used
- recent rainfall totals
- recent temperature suitability
- recent dengue momentum
- selected model
- model validation metrics

### 5. Methodology Panel

Plain-language explanation:

- the model forecasts reported cases
- weather impact is delayed
- UC estimates are risk allocation estimates
- forecasts have uncertainty
- underreporting scenarios are assumptions

## Weekly Refresh Workflow

The app should support a script or button that:

1. fetches recent weather
2. fetches weather forecast
3. updates lagged features
4. generates city-level forecast
5. creates UC-level allocation
6. writes app-ready JSON
7. refreshes dashboard and map

## Forecast Horizons

Minimum:

- next week

Useful:

- next 2 weeks

Optional:

- 1-3 month seasonal risk outlook

The 1-3 month view should be labeled as a risk outlook, not exact prediction.

## Technical Recommendation

Start as a local web app with static generated JSON outputs.

Recommended stack:

- Python for data processing, model training, and forecast generation
- scikit-learn, xgboost, lightgbm, catboost if available
- pandas/geopandas for data preparation
- Leaflet for UC map
- simple HTML/CSS/JavaScript or React if a richer app is needed

Static JSON-first architecture is good for reproducibility:

- `outputs/latest_forecast.json`
- `outputs/uc_forecast.geojson`
- `outputs/model_metrics.json`

Then the frontend reads these files.

