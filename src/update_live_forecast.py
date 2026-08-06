#!/usr/bin/env python3
"""Refresh the public forecast using the Study 1 model and live weather.

This runs the XGBoost model trained and validated in
`Study # 1 Weekly_AI_Study_Rawapidni_Dengue_Forcast`, which won that study's
held-out 2025 Rawalpindi test (RMSE 30.10, MAE 26.81, R2 0.620, MAPE 11.5%).
The saved estimator and scaler are vendored under models/ so CI does not need
the study folder. Nothing is retrained here -- the published app and the study
are the same model.

That validation supplied real observed counts for Cases_Lag_1w/2w/3w. Those
lags dominate the model, so they are resolved in priority order: observed
surveillance counts from data/recent_cases.csv, then the model's own nowcast
for weeks after the last observation, then seasonal history as a last resort.
Substituting seasonal history for all three takes the same model from R2 0.620
to R2 -10.6 on the 2025 window, so every published week records which basis it
actually used.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data_processed" / "model_training_dataset_2013_2024.csv"
PUBLIC_DATA = ROOT / "data"
RECENT_CASES = PUBLIC_DATA / "recent_cases.csv"
MODEL_DIR = ROOT / "models"
STUDY_VALIDATION = ROOT / "reports" / "study1_2025_validation_metrics.csv"
LAT, LON = 33.5651, 73.0169

# Study 1's selected model and its held-out 2025 Rawalpindi scores.
SELECTED_MODEL = "XGBoost"
STUDY_SCORES = {"rmse": 30.10, "mae": 26.81, "r2": 0.620, "mape": 11.5, "weeks": 7}

# UC alert thresholds on expected reported cases, calibrated from the historical
# transmission-season distribution of allocated UC-week values (weeks 30-48,
# 2013-2024): Yellow ~p75, Orange ~p93, Red ~p97. Historical burden is shown as
# context but no longer sets the colour on its own.
UC_YELLOW, UC_ORANGE, UC_RED = 0.5, 3.0, 8.0

# Beyond this many weeks without surveillance data, rolling the model forward
# from its own output stops being meaningful and we fall back to seasonal history.
MAX_NOWCAST_WEEKS = 8

def load_study_model() -> tuple[xgb.Booster, dict, list[str]]:
    """Study 1's XGBoost, loaded from version-stable formats.

    The study saved .joblib pickles, but those carry the scikit-learn and
    XGBoost versions they were written with (1.6.1 / an older XGBoost) and warn
    about invalid results when unpickled by newer ones. CI installs whatever is
    current, so the booster is loaded from XGBoost's native JSON and the scaler
    is reduced to its mean/scale arrays -- the transform is only (x - mean) / scale.
    Both were exported from the original artifacts and reproduce the study's
    published 2025 metrics exactly.
    """
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "xgb_model.json"))
    scaler = json.loads((MODEL_DIR / "xgb_scaler.json").read_text(encoding="utf-8"))
    return booster, scaler, list(scaler["features"])


def best_iteration_range(booster: xgb.Booster) -> tuple[int, int]:
    """Trees to score with.

    Study 1 trained with early stopping: the booster holds 140 rounds but the
    best iteration was 89. XGBRegressor.predict() truncates there automatically,
    a raw Booster does not, and using all 140 shifts the 2025 validation from
    R2 0.620 to 0.503. So the cut has to be applied explicitly.
    """
    best = booster.attributes().get("best_iteration")
    return (0, int(best) + 1) if best is not None else (0, booster.num_boosted_rounds())


def predict_cases(booster: xgb.Booster, scaler: dict, features: list[str], row: dict) -> float:
    """One week-ahead prediction, back-transformed from log space and floored."""
    values = np.array([[float(row[f]) for f in features]], dtype=float)
    scaled = (values - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    log_pred = booster.inplace_predict(scaled, iteration_range=best_iteration_range(booster))
    return float(np.maximum(np.expm1(log_pred)[0], 0.0))


def json_safe(value):
    """Replace NaN/Infinity with null, recursively.

    Python writes bare `NaN` into JSON, which is not valid JSON and makes the
    browser's fetch().json() throw -- taking the whole page down, not just the
    affected field. Study 1's metrics table has an empty Correlation cell for
    LSTM, which arrives here as NaN.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    return value


def week_add(year: int, week: int, offset: int) -> tuple[int, int]:
    d = datetime.fromisocalendar(year, week, 1).date() + timedelta(days=7 * offset)
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def fetch_weather(start: date, end: date) -> tuple[pd.DataFrame, str]:
    frames = []

    def parse(payload: dict) -> pd.DataFrame:
        daily = payload.get("daily", {})
        if not daily.get("time"):
            return pd.DataFrame()
        frame = pd.DataFrame(daily)
        frame["date"] = pd.to_datetime(frame.pop("time")).dt.date
        return frame

    archive_end = min(end, date.today() - timedelta(days=1))
    if start <= archive_end:
        params = {
            "latitude": LAT, "longitude": LON,
            "start_date": start.isoformat(), "end_date": archive_end.isoformat(),
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,pressure_msl_mean,wind_speed_10m_mean",
            "timezone": "auto",
        }
        response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=25)
        if response.ok:
            frames.append(parse(response.json()))

    forecast_start = max(start, date.today())
    if forecast_start <= end:
        params = {
            "latitude": LAT, "longitude": LON, "forecast_days": min(16, (end - forecast_start).days + 1),
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,pressure_msl_mean,wind_speed_10m_mean",
            "timezone": "auto",
        }
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=25)
        if response.ok:
            frames.append(parse(response.json()))

    if not frames:
        return pd.DataFrame(), "Weather refresh failed; historical seasonal weather used"
    weather = pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")
    return weather.rename(columns={
        "temperature_2m_mean": "Temp_Avg",
        "relative_humidity_2m_mean": "Humidity_Avg",
        "precipitation_sum": "Rainfall_Total",
        "pressure_msl_mean": "Pressure_Avg",
        "wind_speed_10m_mean": "WindSpeed_Avg",
    }), "Fresh Open-Meteo history and forecast weather used"


def weekly_weather(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return weather
    iso = pd.to_datetime(weather["date"]).dt.isocalendar()
    weather = weather.copy()
    weather["Year"], weather["Week"] = iso.year.astype(int), iso.week.astype(int)
    out = weather.groupby(["Year", "Week"], as_index=False).agg({
        "Temp_Avg": "mean", "Humidity_Avg": "mean", "Rainfall_Total": "sum",
        "Pressure_Avg": "mean", "WindSpeed_Avg": "mean", "date": "count",
    }).rename(columns={"date": "Days"})
    out["Month"] = pd.to_datetime(weather.groupby(["Year", "Week"])["date"].min().values).month
    return out


def load_observed_cases() -> dict[tuple[int, int], float]:
    """Observed weekly Rawalpindi cases keyed by (ISO year, ISO week).

    The file is optional. An absent or empty file simply means every recent lag
    falls back to seasonal history, which is the behaviour this app had before a
    surveillance channel existed.
    """
    if not RECENT_CASES.exists():
        return {}
    try:
        frame = pd.read_csv(RECENT_CASES)
    except pd.errors.EmptyDataError:
        return {}
    required = {"year", "week", "cases"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{RECENT_CASES.name} must have columns {sorted(required)}")
    frame = frame.dropna(subset=["year", "week", "cases"])
    return {
        (int(row.year), int(row.week)): float(row.cases)
        for row in frame.itertuples(index=False)
    }


def run_forecast(
    model,
    scaler,
    feature_cols: list[str],
    hist: pd.DataFrame,
    weather: pd.DataFrame,
    observed: dict[tuple[int, int], float],
    horizon: int = 4,
) -> tuple[list[dict], dict]:
    """Roll Study 1's XGBoost forward one week at a time.

    Weeks between the last observation and the first published week are
    nowcast so that Cases_Lag_1w -- the feature the study's 2025 validation
    supplied from real surveillance -- is grounded in observed data for as
    long as the file allows, instead of jumping straight to a seasonal median.
    """
    raw = hist[hist["City"].eq("Rawalpindi")].copy()
    seasonal = raw.groupby("Week").median(numeric_only=True)
    population = float(raw["Population"].dropna().iloc[-1])
    weather_weekly = weekly_weather(weather)
    weather_lookup = {(int(r.Year), int(r.Week)): r for r in weather_weekly.itertuples(index=False)}
    observed_weeks = {(y, w) for (y, w) in observed}

    def value(year: int, week: int, column: str) -> float:
        row = weather_lookup.get((year, week))
        if row is not None and hasattr(row, column) and pd.notna(getattr(row, column)):
            return float(getattr(row, column))
        if week in seasonal.index and column in seasonal.columns:
            return float(seasonal.loc[week, column])
        return float(raw[column].median())

    def seasonal_cases(week: int) -> float:
        if week in seasonal.index:
            return float(seasonal.loc[week, "Cases_Raw"])
        return float(raw["Cases_Raw"].median())

    predicted: dict[tuple[int, int], float] = {}

    def case_lag(year: int, week: int) -> tuple[float, str]:
        if (year, week) in observed:
            return observed[(year, week)], "observed"
        if (year, week) in predicted:
            return predicted[(year, week)], "nowcast"
        return seasonal_cases(week), "climatology"

    first_monday = date.today() + timedelta(days=(7 - date.today().weekday()))
    first_year, first_week = first_monday.isocalendar()[0], first_monday.isocalendar()[1]

    # Start the roll-forward at the week after the last observation so the
    # unreported gap is nowcast rather than replaced by climatology.
    start_year, start_week = first_year, first_week
    if observed_weeks:
        last_year, last_week = max(observed_weeks)
        gap_year, gap_week = week_add(last_year, last_week, 1)
        gap = 0
        probe = (gap_year, gap_week)
        while probe != (first_year, first_week) and gap < MAX_NOWCAST_WEEKS:
            probe = week_add(*probe, 1)
            gap += 1
        if gap < MAX_NOWCAST_WEEKS:
            start_year, start_week = gap_year, gap_week

    published: list[dict] = []
    year, week = start_year, start_week
    reached_target = False
    while True:
        monday = datetime.fromisocalendar(year, week, 1).date()
        row = {"Year": year, "Week": week, "Month": monday.month, "Population": population}
        for column in ["Temp_Avg", "Humidity_Avg", "Rainfall_Total", "Pressure_Avg", "WindSpeed_Avg"]:
            row[column] = value(year, week, column)
        bases = []
        for lag in (1, 2, 3):
            lag_year, lag_week = week_add(year, week, -lag)
            cases, basis = case_lag(lag_year, lag_week)
            row[f"Cases_Lag_{lag}w"] = cases
            bases.append(basis)
        for lag in (2, 3):
            lag_year, lag_week = week_add(year, week, -lag)
            row[f"Temp_Avg_Lag_{lag}w"] = value(lag_year, lag_week, "Temp_Avg")
            row[f"Rainfall_Total_Lag_{lag}w"] = value(lag_year, lag_week, "Rainfall_Total")

        expected = predict_cases(model, scaler, feature_cols, row)
        predicted[(year, week)] = expected

        reached_target = reached_target or (year, week) == (first_year, first_week)
        if reached_target:
            published.append({
                "year": year,
                "week": week,
                "expected_cases": round(expected, 1),
                "lower_cases": round(max(expected * 0.65, 0.0), 1),
                "upper_cases": round(expected * 1.45 + 3, 1),
                "seasonal_norm": round(seasonal_cases(week), 1),
                "cases_basis": bases[0],
                "weather_basis": weather_basis(year, week, weather_lookup),
            })
            if len(published) == horizon:
                break
        year, week = week_add(year, week, 1)

    provenance = {
        "observed_weeks": len(observed_weeks),
        "nowcast_weeks": sum(1 for key in predicted if key < (first_year, first_week)),
        "cases_through": (
            {"year": max(observed_weeks)[0], "week": max(observed_weeks)[1]}
            if observed_weeks else None
        ),
    }
    return published, provenance


def weather_basis(year: int, week: int, weather_lookup: dict) -> str:
    """Whether this week's own weather came from the API, and how completely.

    Open-Meteo's free forecast reaches ~16 days, so the later weeks of the
    horizon have no real weather behind them at all.
    """
    row = weather_lookup.get((year, week))
    if row is None:
        return "climatology"
    days = int(getattr(row, "Days", 0))
    if days >= 7:
        return "forecast"
    return "partial" if days > 0 else "climatology"


def uc_alert(expected: float) -> str:
    """Alert level for one UC from its expected reported cases this week.

    Historical burden sets the allocation share, not the colour. Letting it set
    the colour directly -- as this did previously -- pinned the high-burden UCs
    to Red all year round and made the map unable to respond to the forecast.
    """
    if expected >= UC_RED:
        return "Red"
    if expected >= UC_ORANGE:
        return "Orange"
    if expected >= UC_YELLOW:
        return "Yellow"
    return "Green"


def update_geojson(first_cases: float) -> tuple[dict, list[dict]]:
    source = json.loads((PUBLIC_DATA / "rawalpindi_uc_forecast.geojson").read_text(encoding="utf-8"))
    features = [f for f in source["features"] if f.get("properties", {}).get("tehsil") == "Rawalpindi Tehsil"]
    weights = [max(float(f.get("properties", {}).get("historical_cases", 0)), 0) + 1 for f in features]
    total = sum(weights) or 1
    rows = []
    for feature, weight in zip(features, weights):
        props = feature["properties"]
        expected = first_cases * weight / total
        historical = float(props.get("historical_cases", 0) or 0)
        props["expected_cases"] = round(expected, 2)
        props["share_pct"] = round(100.0 * weight / total, 2)
        props["alert"] = uc_alert(expected)
        rows.append({
            "uc": props.get("uc", ""), "tehsil": props.get("tehsil", ""),
            "expected_cases": round(expected, 2), "historical_cases": int(round(historical)),
            "share_pct": props["share_pct"], "alert": props["alert"],
        })
    result = {**source, "features": features}
    rows.sort(key=lambda r: r["expected_cases"], reverse=True)
    return result, rows[:25]


def main() -> None:
    hist = pd.read_csv(TRAINING)
    hist = hist[hist["Cases_Raw"].notna()].copy()
    model, scaler, features = load_study_model()

    horizon_end = date.today() + timedelta(days=45)
    weather, weather_status = fetch_weather(date.today() - timedelta(days=84), horizon_end)
    observed = load_observed_cases()
    weekly, provenance = run_forecast(model, scaler, features, hist, weather, observed)
    geojson, top_ucs = update_geojson(weekly[0]["expected_cases"])

    if provenance["cases_through"]:
        through = provenance["cases_through"]
        surveillance_status = (
            f"Recent case lags use observed Rawalpindi surveillance counts through "
            f"{through['year']} week {through['week']}"
            + (f", nowcast across {provenance['nowcast_weeks']} unreported week(s)."
               if provenance["nowcast_weeks"] else ".")
        )
    else:
        surveillance_status = (
            "No observed case counts are loaded, so recent case lags fall back to "
            "historical seasonal baselines. Add rows to data/recent_cases.csv to "
            "ground the forecast in reported cases."
        )

    external = {"model": SELECTED_MODEL, **STUDY_SCORES}
    if STUDY_VALIDATION.exists():
        table = pd.read_csv(STUDY_VALIDATION)
        external["all_models"] = table.to_dict(orient="records")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_note": "Forecasts are expected reported dengue cases from the Study 1 XGBoost model. UC values allocate the city forecast by historical burden share.",
        "weather_status": weather_status,
        "surveillance_status": surveillance_status,
        "cases_through": provenance["cases_through"],
        "uc_thresholds": {"yellow": UC_YELLOW, "orange": UC_ORANGE, "red": UC_RED},
        "selected_model": {
            "name": SELECTED_MODEL,
            "source": "Study 1 Weekly AI Dengue Forecast — saved model, not retrained",
            "validation": "held-out 2025 Rawalpindi, weeks 36-42",
            "rmse": STUDY_SCORES["rmse"], "mae": STUDY_SCORES["mae"],
            "r2": STUDY_SCORES["r2"], "mape": STUDY_SCORES["mape"],
        },
        "model_comparison": external.get("all_models", []),
        "external_2025_validation": external,
        "weekly_forecasts": weekly, "top_ucs": top_ucs,
        "alert_counts": dict(pd.Series([f["properties"]["alert"] for f in geojson["features"]]).value_counts()),
    }
    PUBLIC_DATA.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DATA / "latest_forecast.json").write_text(
        json.dumps(
            json_safe(payload), indent=2, allow_nan=False,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        ),
        encoding="utf-8",
    )
    (PUBLIC_DATA / "rawalpindi_uc_forecast.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    print(json.dumps({
        "generated_at": payload["generated_at"],
        "weather_status": weather_status,
        "cases_through": provenance["cases_through"],
        "nowcast_weeks": provenance["nowcast_weeks"],
        "next_week": weekly[0],
        "alert_counts": payload["alert_counts"],
    }, default=str))


if __name__ == "__main__":
    main()
