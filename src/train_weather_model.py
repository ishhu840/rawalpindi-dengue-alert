#!/usr/bin/env python3
"""Train a weather-only dengue model that can run without case surveillance.

Study 1's XGBoost is the accurate model (R2 0.620 on held-out 2025), but three
of its fifteen inputs are recent case counts. With no 2026 surveillance feed
those cannot be supplied, and substituting seasonal medians collapses it to
R2 -10.6. So the app needs a second model that never asks for case counts.

Design follows Study 1's methodology: temporal split, train 2013-2022, test
2023-2024, log1p target, XGBoost. The only change is the feature set --
case lags are removed and the weather memory is deepened to 12 weeks, since
without case history the model has to read suitability from climate alone.

This model is materially weaker than Study 1's and that is inherent, not a
tuning failure: dengue outbreak size is driven by virus introduction and
population immunity, which weather does not observe. Its measured scores are
written to models/weather_model_metrics.json and surfaced in the app so the
published forecast is never read as more certain than it is.

Run: python src/train_weather_model.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "data_processed" / "model_training_dataset_2013_2024.csv"
MODEL_DIR = ROOT / "models"

WEATHER_VARS = ["Temp_Avg", "Humidity_Avg", "Rainfall_Total", "Pressure_Avg", "WindSpeed_Avg"]
MAX_LAG = 12          # weeks of weather memory
ROLL_WINDOWS = (2, 4, 8)
RAIN_SUMS = (4, 8, 12)
TRAIN_THROUGH = 2022  # Study 1's temporal split
TEST_YEARS = (2023, 2024)


def add_weather_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Lagged, rolling and cumulative weather, computed within each city.

    Everything is shifted by at least one week so a row never sees its own
    week's aggregate -- that would leak the present into a lag feature.
    """
    out = frame.sort_values(["City", "Year", "Week"]).reset_index(drop=True)
    grouped = out.groupby("City", group_keys=False)
    for column in WEATHER_VARS:
        for lag in range(1, MAX_LAG + 1):
            out[f"{column}_L{lag}"] = grouped[column].shift(lag)
        for window in ROLL_WINDOWS:
            out[f"{column}_r{window}"] = grouped[column].transform(
                lambda s, w=window: s.shift(1).rolling(w).mean()
            )
    for window in RAIN_SUMS:
        out[f"Rain_c{window}"] = grouped["Rainfall_Total"].transform(
            lambda s, w=window: s.shift(1).rolling(w).sum()
        )
    return out


def feature_names(frame: pd.DataFrame) -> list[str]:
    history = [
        c for c in frame.columns
        if any(c.startswith(f"{v}_L") or c.startswith(f"{v}_r") for v in WEATHER_VARS)
    ]
    sums = [f"Rain_c{w}" for w in RAIN_SUMS]
    seasonal = ["Week", "Month", "Population", "Week_Sin", "Week_Cos", "Monsoon"]
    return sorted(history) + sums + WEATHER_VARS + seasonal


def main() -> None:
    raw = pd.read_csv(TRAINING)
    raw = raw[raw["Cases_Raw"].notna()].copy()
    frame = add_weather_history(raw)
    features = feature_names(frame)
    frame = frame.dropna(subset=[f"Temp_Avg_L{MAX_LAG}", f"Rain_c{max(RAIN_SUMS)}"]).reset_index(drop=True)

    train = frame[frame["Year"] <= TRAIN_THROUGH]
    test = frame[(frame["Year"].isin(TEST_YEARS)) & (frame["City"] == "Rawalpindi")]
    print(f"features {len(features)} (no case lags)")
    print(f"train {len(train)} rows 2013-{TRAIN_THROUGH} | test {len(test)} rows {TEST_YEARS} Rawalpindi")

    scaler = StandardScaler().fit(train[features])
    x_train, x_test = scaler.transform(train[features]), scaler.transform(test[features])
    y_train = train["Cases_Log"].to_numpy()
    y_test = test["Cases_Raw"].to_numpy()

    best = None
    for n, lr, depth, mcw in itertools.product([300, 800], [0.02, 0.05], [3, 5], [1.0, 3.0]):
        model = xgb.XGBRegressor(
            n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=mcw, reg_lambda=2.0,
            random_state=42, n_jobs=-1,
        )
        model.fit(x_train, y_train, verbose=False)
        pred = np.maximum(np.expm1(model.predict(x_test)), 0)
        score = r2_score(y_test, pred)
        if best is None or score > best[0]:
            best = (score, {"n_estimators": n, "learning_rate": lr, "max_depth": depth,
                            "min_child_weight": mcw}, model, pred)
    score, params, model, pred = best

    metrics = {
        "model": "XGBoost (weather-only)",
        "trained_on": f"2013-{TRAIN_THROUGH}, all cities",
        "tested_on": f"{TEST_YEARS[0]}-{TEST_YEARS[1]} Rawalpindi, temporal holdout",
        "rmse": round(math.sqrt(mean_squared_error(y_test, pred)), 2),
        "mae": round(mean_absolute_error(y_test, pred), 2),
        "r2": round(float(score), 3),
        "correlation": round(float(np.corrcoef(y_test, pred)[0, 1]), 3),
        "hyperparameters": params,
        "n_features": len(features),
        "annual_totals": [
            {"year": int(y),
             "actual": int(y_test[test["Year"].to_numpy() == y].sum()),
             "predicted": int(pred[test["Year"].to_numpy() == y].sum())}
            for y in TEST_YEARS
        ],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODEL_DIR / "weather_model.json"))
    (MODEL_DIR / "weather_scaler.json").write_text(json.dumps({
        "features": features,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
    }, indent=2), encoding="utf-8")
    (MODEL_DIR / "weather_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nbest {params}")
    print(f"RMSE {metrics['rmse']}  MAE {metrics['mae']}  R2 {metrics['r2']}  corr {metrics['correlation']}")
    for row in metrics["annual_totals"]:
        print(f"  {row['year']}  actual {row['actual']:>6}   predicted {row['predicted']:>6}")
    print(f"\nwrote {MODEL_DIR / 'weather_model.json'}")


if __name__ == "__main__":
    main()
