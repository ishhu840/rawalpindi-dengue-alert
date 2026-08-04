#!/usr/bin/env python3
"""Refresh the public forecast using packaged training data and live weather.

This updater is deliberately independent of the private study folders so it can
run inside GitHub Actions. It does not invent observed dengue cases: until a
UC-level surveillance feed is connected, recent case lags use seasonal history.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data_processed" / "model_training_dataset_2013_2024.csv"
PUBLIC_DATA = ROOT / "data"
APP_DATA = ROOT / "app" / "data"
LAT, LON = 33.5651, 73.0169

BASE_NUMERIC = [
    "Temp_Avg", "Humidity_Avg", "Rainfall_Total", "Pressure_Avg", "WindSpeed_Avg",
    "Cases_Lag_1w", "Cases_Lag_2w", "Cases_Lag_3w", "Temp_Avg_Lag_2w",
    "Temp_Avg_Lag_3w", "Rainfall_Total_Lag_2w", "Rainfall_Total_Lag_3w",
    "Week.1", "Month", "Population", "Week_Sin", "Week_Cos", "Monsoon",
]


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
        "Pressure_Avg": "mean", "WindSpeed_Avg": "mean",
    })
    out["Month"] = pd.to_datetime(weather.groupby(["Year", "Week"])["date"].min().values).month
    return out


def build_features(hist: pd.DataFrame, weather: pd.DataFrame, horizon: int = 4) -> pd.DataFrame:
    raw = hist[hist["City"].eq("Rawalpindi")].copy()
    seasonal = raw.groupby("Week").median(numeric_only=True)
    current = date.today().isocalendar()
    first_week = date.today() + timedelta(days=(7 - date.today().weekday()))
    iso = first_week.isocalendar()
    weather_weekly = weekly_weather(weather)
    weather_lookup = {(int(r.Year), int(r.Week)): r for r in weather_weekly.itertuples(index=False)}

    def value(year: int, week: int, column: str) -> float:
        row = weather_lookup.get((year, week))
        if row is not None and hasattr(row, column) and pd.notna(getattr(row, column)):
            return float(getattr(row, column))
        if week in seasonal.index and column in seasonal.columns:
            return float(seasonal.loc[week, column])
        return float(raw[column].median())

    def seasonal_cases(year: int, week: int) -> float:
        if week in seasonal.index:
            return float(seasonal.loc[week, "Cases_Raw"])
        return float(raw["Cases_Raw"].median())

    rows = []
    for offset in range(horizon):
        target = first_week + timedelta(days=offset * 7)
        target_iso = target.isocalendar()
        year, week = int(target_iso.year), int(target_iso.week)
        row = {
            "City": "Rawalpindi", "Year": year, "Week": week, "Month": target.month,
            "Population": float(raw["Population"].dropna().iloc[-1]),
        }
        for column in ["Temp_Avg", "Humidity_Avg", "Rainfall_Total", "Pressure_Avg", "WindSpeed_Avg"]:
            row[column] = value(year, week, column)
        for lag in (1, 2, 3):
            lag_year, lag_week = week_add(year, week, -lag)
            row[f"Cases_Lag_{lag}w"] = seasonal_cases(lag_year, lag_week)
        for lag in (2, 3):
            lag_year, lag_week = week_add(year, week, -lag)
            row[f"Temp_Avg_Lag_{lag}w"] = value(lag_year, lag_week, "Temp_Avg")
            row[f"Rainfall_Total_Lag_{lag}w"] = value(lag_year, lag_week, "Rainfall_Total")
        row["Week.1"] = week
        row["Week_Sin"] = math.sin(2 * math.pi * week / 52)
        row["Week_Cos"] = math.cos(2 * math.pi * week / 52)
        row["Monsoon"] = int(row["Month"] in [7, 8, 9, 10])
        rows.append(row)
    return pd.DataFrame(rows)


def update_geojson(first_cases: float) -> dict:
    source = json.loads((APP_DATA / "rawalpindi_uc_forecast.geojson").read_text(encoding="utf-8"))
    features = [f for f in source["features"] if f.get("properties", {}).get("tehsil") == "Rawalpindi Tehsil"]
    weights = [max(float(f.get("properties", {}).get("historical_cases", 0)), 0) + 1 for f in features]
    total = sum(weights) or 1
    rows = []
    for feature, weight in zip(features, weights):
        props = feature["properties"]
        expected = first_cases * weight / total
        historical = float(props.get("historical_cases", 0) or 0)
        props["expected_cases"] = round(expected, 2)
        # Preserve the study's original UC risk logic: forecast burden and
        # historical mapped burden both contribute to the alert color.
        props["alert"] = (
            "Red" if expected >= 8 or historical >= 500 else
            "Orange" if expected >= 4 or historical >= 200 else
            "Yellow" if expected >= 1.5 or historical >= 50 else
            "Green"
        )
        rows.append({"uc": props.get("uc", ""), "tehsil": props.get("tehsil", ""), "expected_cases": round(expected, 2), "historical_cases": int(round(historical)), "alert": props["alert"]})
    result = {**source, "features": features}
    rows.sort(key=lambda r: r["expected_cases"], reverse=True)
    return result, rows[:25]


def main() -> None:
    hist = pd.read_csv(TRAINING)
    hist = hist[hist["Cases_Raw"].notna()].copy()
    features = [c for c in BASE_NUMERIC if c in hist.columns]
    preprocessor = ColumnTransformer([("num", StandardScaler(), features), ("cat", OneHotEncoder(handle_unknown="ignore"), ["City"])])
    model = make_pipeline(preprocessor, GradientBoostingRegressor(n_estimators=300, learning_rate=0.035, max_depth=3, min_samples_leaf=5, random_state=42))
    model.fit(hist[features + ["City"]], hist["Cases_Log"])

    horizon_end = date.today() + timedelta(days=45)
    weather, weather_status = fetch_weather(date.today() - timedelta(days=84), horizon_end)
    future = build_features(hist, weather)
    predictions = np.maximum(np.expm1(model.predict(future[features + ["City"]])), 0)
    weekly = []
    for row, prediction in zip(future.itertuples(index=False), predictions):
        expected = float(prediction)
        weekly.append({"year": int(row.Year), "week": int(row.Week), "expected_cases": round(expected, 1), "lower_cases": round(expected * 0.65, 1), "upper_cases": round(expected * 1.45 + 3, 1)})
    geojson, top_ucs = update_geojson(weekly[0]["expected_cases"])
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_note": "Forecasts are refreshed automatically from the trained Gradient Boosting model. UC values are spatial risk allocation estimates.",
        "weather_status": weather_status,
        "surveillance_status": "This alert is generated from the Rawalpindi dengue training dataset and fresh weather. No external dengue dashboard or case feed is used.",
        "selected_model": {"name": "Gradient Boosting", "rolling_origin_mean_rmse": 73.822, "rolling_origin_mean_mae": 14.609, "rolling_origin_mean_smape": 45.967, "rolling_origin_mean_r2": 0.753},
        "model_comparison": [], "external_2025_validation": {}, "weekly_forecasts": weekly, "top_ucs": top_ucs,
        "alert_counts": dict(pd.Series([r["alert"] for r in top_ucs]).value_counts()),
    }
    for directory in (PUBLIC_DATA, APP_DATA):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "latest_forecast.json").write_text(
            json.dumps(payload, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)),
            encoding="utf-8",
        )
        (directory / "rawalpindi_uc_forecast.geojson").write_text(json.dumps(geojson), encoding="utf-8")
    print(json.dumps({"generated_at": payload["generated_at"], "weather_status": weather_status, "next_week": weekly[0]}))


if __name__ == "__main__":
    main()
