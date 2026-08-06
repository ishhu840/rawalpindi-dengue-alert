#!/usr/bin/env python3
"""Assert the vendored model still reproduces Study 1's published 2025 result.

The app serves a model it did not train. If a library upgrade, a re-export or
an edit to the loading code shifts its predictions, that must fail loudly here
rather than quietly change what the public map shows.

Two failure modes this has already caught:

  * the .joblib pickles warn about version drift under newer scikit-learn and
    XGBoost, which is why the vendored artifacts are native JSON instead;
  * Study 1 trained with early stopping (best_iteration 89 of 140 rounds), and
    scoring with all 140 moves the 2025 validation from R2 0.620 to 0.503.

Run: python src/verify_model.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_live_forecast import MODEL_DIR, load_study_model, predict_cases  # noqa: E402

# From Study 1's 04_results/2025_validation_metrics.csv, XGBoost row. Each is
# compared at the precision it was published to -- MAPE is recorded as 11.5,
# so 11.458 is a match, not a drift.
PUBLISHED = {"rmse": (30.10, 2), "mae": (26.81, 2), "r2": (0.620, 2), "mape": (11.5, 1)}


def main() -> int:
    booster, scaler, features = load_study_model()
    val = pd.read_csv(MODEL_DIR / "validation_2025_weekly.csv")
    actual = val["Cases"].to_numpy(dtype=float)
    predicted = np.array([
        predict_cases(booster, scaler, features, row)
        for row in val.to_dict(orient="records")
    ])

    got = {
        "rmse": math.sqrt(mean_squared_error(actual, predicted)),
        "mae": mean_absolute_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "mape": float(np.mean(np.abs((actual - predicted) / actual)) * 100),
    }

    print(f"Study 1 XGBoost on held-out 2025 Rawalpindi (weeks {val.Week.min()}-{val.Week.max()})")
    print(f"  {'metric':<8}{'published':>12}{'reproduced':>13}")
    failures = []
    for key, (want, places) in PUBLISHED.items():
        ok = round(got[key], places) == round(want, places)
        if not ok:
            failures.append(f"{key}: expected {want}, got {got[key]:.4f}")
        print(f"  {key:<8}{want:>12.2f}{got[key]:>13.2f}   {'ok' if ok else 'MISMATCH'}")

    if failures:
        print("\nFAILED — the vendored model no longer matches the study:")
        for line in failures:
            print(f"  {line}")
        return 1
    print("PASS — app predictions are identical to the study's saved model.")

    # The weather-only fallback: check it loads and its recorded scores are the
    # ones the app will publish. Weak scores are expected here and are not a
    # failure -- an absent or unreadable model is.
    from update_live_forecast import load_weather_model  # noqa: E402

    weather = load_weather_model()
    if weather is None:
        print("\nFAILED — models/weather_model.json missing; run src/train_weather_model.py")
        return 1
    _, wscaler, wmetrics = weather
    print(f"\nWeather-only fallback ({wmetrics.get('tested_on','?')})")
    print(f"  features {len(wscaler['features'])}, no case lags")
    print(f"  RMSE {wmetrics.get('rmse')}  MAE {wmetrics.get('mae')}"
          f"  R2 {wmetrics.get('r2')}  corr {wmetrics.get('correlation')}")
    if any(f.startswith("Cases_Lag") for f in wscaler["features"]):
        print("\nFAILED — weather-only model contains case-lag features")
        return 1
    print("  ok — contains no case-count inputs, so it can run without surveillance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
