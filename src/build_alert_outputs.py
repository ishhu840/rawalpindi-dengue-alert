#!/usr/bin/env python3
"""Build model validation outputs and app-ready dengue alert data.

This script intentionally reads from the two source study folders and writes only
inside this Alert App folder.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
MODEL_STUDY = ROOT_DIR / "Study # 1 Weekly_AI_Study_Rawapidni_Dengue_Forcast"
MAP_STUDY = ROOT_DIR / "Helping Study Dengue_Historical_Cases_Rwalpindi_Area_wise"

DATA_DIR = APP_DIR / "data_processed"
OUTPUT_DIR = APP_DIR / "app" / "data"
REPORT_DIR = APP_DIR / "reports"

RAWALPINDI_LAT = 33.5651
RAWALPINDI_LON = 73.0169


@dataclass
class ModelResult:
    name: str
    model: Any
    mean_rmse: float
    mean_mae: float
    mean_smape: float
    mean_r2: float


def ensure_dirs() -> None:
    for path in (DATA_DIR, OUTPUT_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Cases_Raw" not in out.columns and "Cases" in out.columns:
        out["Cases_Raw"] = out["Cases"]
    if "Cases_Log" not in out.columns:
        out["Cases_Log"] = np.log1p(out["Cases_Raw"])
    return out


def load_weekly_model_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    base = MODEL_STUDY / "01_data_prep" / "data"
    train = normalize_target(pd.read_csv(base / "train_weekly.csv"))
    test = normalize_target(pd.read_csv(base / "test_weekly.csv"))
    validation = normalize_target(pd.read_csv(base / "validation_2025_weekly.csv"))
    with open(base / "feature_columns.txt", "r", encoding="utf-8") as fh:
        feature_cols = [line.strip() for line in fh if line.strip()]

    hist = pd.concat([train, test], ignore_index=True)
    hist = hist.sort_values(["City", "Year", "Week"]).reset_index(drop=True)
    validation = validation.sort_values(["City", "Year", "Week"]).reset_index(drop=True)
    return hist, feature_cols, validation


def add_model_features(df: pd.DataFrame, base_feature_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    out["Week_Sin"] = np.sin(2 * np.pi * out["Week"].astype(float) / 52.0)
    out["Week_Cos"] = np.cos(2 * np.pi * out["Week"].astype(float) / 52.0)
    out["Monsoon"] = out["Month"].isin([7, 8, 9, 10]).astype(int)

    numeric_cols = list(base_feature_cols) + ["Week_Sin", "Week_Cos", "Monsoon"]
    categorical_cols = ["City"]
    return out, numeric_cols, categorical_cols


def build_models(numeric_cols: list[str], categorical_cols: list[str]) -> dict[str, Any]:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )
    return {
        "Random Forest": make_pipeline(
            preprocessor,
            RandomForestRegressor(
                n_estimators=220,
                max_depth=18,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Extra Trees": make_pipeline(
            preprocessor,
            ExtraTreesRegressor(
                n_estimators=260,
                max_depth=22,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "Gradient Boosting": make_pipeline(
            preprocessor,
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.035,
                max_depth=3,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
        "Hist Gradient Boosting": make_pipeline(
            preprocessor,
            HistGradientBoostingRegressor(
                max_iter=360,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.08,
                random_state=42,
            ),
        ),
    }


def evaluate_models(df: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> tuple[ModelResult, pd.DataFrame]:
    models = build_models(feature_cols, categorical_cols)
    folds = []
    results: list[ModelResult] = []
    eval_years = [2019, 2020, 2021, 2022, 2023, 2024]

    feature_frame = df[feature_cols + categorical_cols]
    y_log = df["Cases_Log"]

    for name, model in models.items():
        metrics = []
        for year in eval_years:
            train_mask = df["Year"] < year
            test_mask = df["Year"] == year
            if train_mask.sum() < 100 or test_mask.sum() == 0:
                continue

            model.fit(feature_frame.loc[train_mask], y_log.loc[train_mask])
            pred = np.expm1(model.predict(feature_frame.loc[test_mask]))
            pred = np.maximum(pred, 0)
            actual = df.loc[test_mask, "Cases_Raw"].to_numpy()
            rmse = math.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)
            s = smape(actual, pred)
            metrics.append((rmse, mae, s, r2))
            folds.append(
                {
                    "model": name,
                    "test_year": year,
                    "rmse": round(rmse, 3),
                    "mae": round(mae, 3),
                    "smape": round(s, 3),
                    "r2": round(r2, 4),
                    "samples": int(test_mask.sum()),
                }
            )

        arr = np.array(metrics, dtype=float)
        results.append(
            ModelResult(
                name=name,
                model=model,
                mean_rmse=float(arr[:, 0].mean()),
                mean_mae=float(arr[:, 1].mean()),
                mean_smape=float(arr[:, 2].mean()),
                mean_r2=float(arr[:, 3].mean()),
            )
        )

    folds_df = pd.DataFrame(folds)
    summary = (
        folds_df.groupby("model", as_index=False)
        .agg({"rmse": "mean", "mae": "mean", "smape": "mean", "r2": "mean"})
        .sort_values("rmse")
    )
    folds_df.to_csv(REPORT_DIR / "rolling_origin_validation.csv", index=False)
    summary.to_csv(REPORT_DIR / "model_validation_summary.csv", index=False)

    best_name = summary.iloc[0]["model"]
    best = next(result for result in results if result.name == best_name)
    best.model.fit(feature_frame, y_log)
    return best, folds_df


def evaluate_external_2025(best: ModelResult, validation: pd.DataFrame, feature_cols: list[str], categorical_cols: list[str]) -> dict[str, Any]:
    val, _, _ = add_model_features(validation, [c for c in feature_cols if c in validation.columns])
    missing = [c for c in feature_cols if c not in val.columns]
    for col in missing:
        val[col] = 0
    frame = val[feature_cols + categorical_cols]
    pred = np.maximum(np.expm1(best.model.predict(frame)), 0)
    actual = val["Cases_Raw"].to_numpy()
    rows = []
    for _, row in val.iterrows():
        rows.append(
            {
                "year": int(row["Year"]),
                "week": int(row["Week"]),
                "actual_cases": int(row["Cases_Raw"]),
            }
        )
    for row, p in zip(rows, pred):
        row["predicted_cases"] = int(round(float(p)))
        row["error"] = int(row["predicted_cases"] - row["actual_cases"])

    metrics = {
        "model": best.name,
        "rmse": round(math.sqrt(mean_squared_error(actual, pred)), 3),
        "mae": round(mean_absolute_error(actual, pred), 3),
        "smape": round(smape(actual, pred), 3),
        "r2": round(float(r2_score(actual, pred)), 4) if len(actual) > 1 else None,
        "weeks": rows,
    }
    with open(REPORT_DIR / "external_2025_validation.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def fetch_open_meteo_daily(start: date, end: date) -> pd.DataFrame:
    """Fetch Rawalpindi recent and forecast weather. Falls back gracefully."""
    rows = []

    def parse_daily(payload: dict[str, Any]) -> pd.DataFrame:
        daily = payload.get("daily", {})
        if not daily or "time" not in daily:
            return pd.DataFrame()
        out = pd.DataFrame(daily)
        out["date"] = pd.to_datetime(out["time"]).dt.date
        return out

    archive_end = min(end, date.today() - timedelta(days=1))
    if start <= archive_end:
        params = {
            "latitude": RAWALPINDI_LAT,
            "longitude": RAWALPINDI_LON,
            "start_date": start.isoformat(),
            "end_date": archive_end.isoformat(),
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean",
                    "precipitation_sum",
                    "pressure_msl_mean",
                    "wind_speed_10m_mean",
                ]
            ),
            "timezone": "auto",
        }
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=20)
        if r.ok:
            rows.append(parse_daily(r.json()))

    forecast_start = max(start, date.today())
    if forecast_start <= end:
        forecast_days = min(16, (end - forecast_start).days + 1)
        params = {
            "latitude": RAWALPINDI_LAT,
            "longitude": RAWALPINDI_LON,
            "forecast_days": forecast_days,
            "daily": ",".join(
                [
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean",
                    "precipitation_sum",
                    "pressure_msl_mean",
                    "wind_speed_10m_mean",
                ]
            ),
            "timezone": "auto",
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=20)
        if r.ok:
            rows.append(parse_daily(r.json()))

    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df.drop_duplicates("date").sort_values("date")
    rename = {
        "temperature_2m_mean": "Temp_Avg",
        "relative_humidity_2m_mean": "Humidity_Avg",
        "precipitation_sum": "Rainfall_Total",
        "pressure_msl_mean": "Pressure_Avg",
        "wind_speed_10m_mean": "WindSpeed_Avg",
    }
    return df.rename(columns=rename)


def weekly_weather_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    out = daily.copy()
    iso = pd.to_datetime(out["date"]).dt.isocalendar()
    out["Year"] = iso.year.astype(int)
    out["Week"] = iso.week.astype(int)
    weekly = (
        out.groupby(["Year", "Week"], as_index=False)
        .agg(
            {
                "Temp_Avg": "mean",
                "Humidity_Avg": "mean",
                "Rainfall_Total": "sum",
                "Pressure_Avg": "mean",
                "WindSpeed_Avg": "mean",
            }
        )
        .sort_values(["Year", "Week"])
    )
    weekly["Month"] = pd.to_datetime(out.groupby(["Year", "Week"])["date"].min().values).month
    return weekly


def week_add(year: int, week: int, n: int) -> tuple[int, int]:
    d = datetime.fromisocalendar(year, week, 1).date() + timedelta(days=7 * n)
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def make_forecast_features(hist: pd.DataFrame, base_feature_cols: list[str], horizon: int = 4) -> tuple[pd.DataFrame, str]:
    raw = hist[hist["City"] == "Rawalpindi"].copy().sort_values(["Year", "Week"])
    climatology = raw.groupby("Week").median(numeric_only=True)

    today = date.today()
    next_monday = today + timedelta(days=(7 - today.weekday()))
    end = next_monday + timedelta(days=7 * horizon + 14)
    start = today - timedelta(days=84)
    daily = fetch_open_meteo_daily(start, end)
    api_status = "Open-Meteo recent and forecast weather used" if not daily.empty else "Weather API unavailable; historical weekly climatology used"
    api_weekly = weekly_weather_from_daily(daily)
    api_lookup = {(int(r.Year), int(r.Week)): r for r in api_weekly.itertuples(index=False)} if not api_weekly.empty else {}

    current_iso = next_monday.isocalendar()
    target_weeks = [week_add(int(current_iso.year), int(current_iso.week), i) for i in range(horizon)]
    rows = []
    predictions_for_lags: dict[tuple[int, int], float] = {}

    # Use seasonal baseline for unknown recent surveillance lags after the last observed case week.
    def case_for(year: int, week: int) -> float:
        key = (year, week)
        if key in predictions_for_lags:
            return predictions_for_lags[key]
        matched = raw[(raw["Year"] == year) & (raw["Week"] == week)]
        if not matched.empty:
            return float(matched.iloc[-1]["Cases_Raw"])
        if week in climatology.index:
            return float(climatology.loc[week, "Cases_Raw"])
        return float(raw["Cases_Raw"].median())

    def weather_for(year: int, week: int, col: str) -> float:
        key = (year, week)
        if key in api_lookup and hasattr(api_lookup[key], col):
            val = getattr(api_lookup[key], col)
            if pd.notna(val):
                return float(val)
        if week in climatology.index and col in climatology.columns:
            return float(climatology.loc[week, col])
        return float(raw[col].median())

    for year, week in target_weeks:
        month = datetime.fromisocalendar(year, week, 1).month
        row = {
            "City": "Rawalpindi",
            "Year": year,
            "Week": week,
            "Month": month,
            "Population": float(raw["Population"].dropna().iloc[-1]) if "Population" in raw else 2098231,
            "Temp_Avg": weather_for(year, week, "Temp_Avg"),
            "Humidity_Avg": weather_for(year, week, "Humidity_Avg"),
            "Rainfall_Total": weather_for(year, week, "Rainfall_Total"),
            "Pressure_Avg": weather_for(year, week, "Pressure_Avg"),
            "WindSpeed_Avg": weather_for(year, week, "WindSpeed_Avg"),
        }
        for lag in (1, 2, 3):
            ly, lw = week_add(year, week, -lag)
            row[f"Cases_Lag_{lag}w"] = case_for(ly, lw)
        for lag in (2, 3):
            ly, lw = week_add(year, week, -lag)
            row[f"Temp_Avg_Lag_{lag}w"] = weather_for(ly, lw, "Temp_Avg")
            row[f"Rainfall_Total_Lag_{lag}w"] = weather_for(ly, lw, "Rainfall_Total")
        for col in base_feature_cols:
            row.setdefault(col, float(raw[col].median()) if col in raw.columns else 0.0)
        rows.append(row)

    return pd.DataFrame(rows), api_status


def norm_uc(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:[a-z]{0,3}-?\d+-)\s*", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_js_object(path: Path, var_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"const\s+{re.escape(var_name)}\s*=\s*(.*?);", re.S)
    match = pattern.search(text)
    if not match:
        return {}
    return json.loads(match.group(1))


def build_uc_forecast_geojson(city_forecast: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    uc_geo = json.loads((MAP_STUDY / "rawalpindi_uc.geojson").read_text(encoding="utf-8"))
    uc_counts = parse_js_object(MAP_STUDY / "uc_counts.js", "ucCount")
    count_by_norm = {norm_uc(k): float(v) for k, v in uc_counts.items()}
    total_weight = 0.0
    weights = []

    for feature in uc_geo["features"]:
        uc = feature["properties"].get("uc", "")
        base = count_by_norm.get(norm_uc(uc), 0.0)
        weight = base + 1.0
        weights.append(weight)
        total_weight += weight

    top_rows = []
    for feature, weight in zip(uc_geo["features"], weights):
        props = feature["properties"]
        expected = city_forecast * weight / total_weight if total_weight else 0.0
        baseline = max(weight - 1.0, 0.0)
        if expected >= 8 or baseline >= 500:
            alert = "Red"
        elif expected >= 4 or baseline >= 200:
            alert = "Orange"
        elif expected >= 1.5 or baseline >= 50:
            alert = "Yellow"
        else:
            alert = "Green"
        props["expected_cases"] = round(float(expected), 2)
        props["historical_cases"] = int(round(baseline))
        props["alert"] = alert
        top_rows.append(
            {
                "uc": props.get("uc", ""),
                "tehsil": props.get("tehsil", ""),
                "expected_cases": round(float(expected), 2),
                "historical_cases": int(round(baseline)),
                "alert": alert,
            }
        )

    top_rows.sort(key=lambda r: r["expected_cases"], reverse=True)
    return uc_geo, top_rows[:25]


def main() -> None:
    ensure_dirs()

    hist, base_feature_cols, validation = load_weekly_model_data()
    hist, numeric_cols, categorical_cols = add_model_features(hist, base_feature_cols)
    best, folds = evaluate_models(hist, numeric_cols, categorical_cols)
    external = evaluate_external_2025(best, validation, numeric_cols, categorical_cols)

    forecast_frame, api_status = make_forecast_features(hist, base_feature_cols, horizon=4)
    forecast_frame, _, _ = add_model_features(forecast_frame, base_feature_cols)
    preds = np.maximum(np.expm1(best.model.predict(forecast_frame[numeric_cols + categorical_cols])), 0)

    week_forecasts = []
    for row, pred in zip(forecast_frame.itertuples(index=False), preds):
        expected = float(pred)
        week_forecasts.append(
            {
                "year": int(row.Year),
                "week": int(row.Week),
                "expected_cases": round(expected, 1),
                "lower_cases": round(max(expected * 0.65, 0), 1),
                "upper_cases": round(expected * 1.45 + 3, 1),
            }
        )

    uc_geojson, top_ucs = build_uc_forecast_geojson(float(week_forecasts[0]["expected_cases"]))

    validation_summary = pd.read_csv(REPORT_DIR / "model_validation_summary.csv").round(3)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "forecast_note": "Forecasts represent expected reported dengue cases. UC values are spatial risk allocation estimates.",
        "weather_status": api_status,
        "surveillance_status": "No live 2026 dengue surveillance feed is connected yet; unknown recent case lags use historical seasonal Rawalpindi baselines.",
        "selected_model": {
            "name": best.name,
            "rolling_origin_mean_rmse": round(best.mean_rmse, 3),
            "rolling_origin_mean_mae": round(best.mean_mae, 3),
            "rolling_origin_mean_smape": round(best.mean_smape, 3),
            "rolling_origin_mean_r2": round(best.mean_r2, 4),
        },
        "model_comparison": validation_summary.to_dict(orient="records"),
        "external_2025_validation": external,
        "weekly_forecasts": week_forecasts,
        "top_ucs": top_ucs,
        "alert_counts": dict(pd.Series([r["alert"] for r in top_ucs]).value_counts()),
    }

    with open(OUTPUT_DIR / "latest_forecast.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=json_default)
    with open(OUTPUT_DIR / "rawalpindi_uc_forecast.geojson", "w", encoding="utf-8") as fh:
        json.dump(uc_geojson, fh, default=json_default)

    hist.to_csv(DATA_DIR / "model_training_dataset_2013_2024.csv", index=False)
    forecast_frame.to_csv(DATA_DIR / "latest_feature_rows.csv", index=False)

    print("Selected model:", best.name)
    print("Next forecast:", week_forecasts[0])
    print("Weather:", api_status)
    print("Wrote:", OUTPUT_DIR / "latest_forecast.json")
    print("Wrote:", OUTPUT_DIR / "rawalpindi_uc_forecast.geojson")


if __name__ == "__main__":
    main()
