"""Fit the submission model and write both output files.

Run:  python -m src.predict

Produces:
  validation_predictions.csv        12,000 rows of load_id,predicted_rate
  data/december_chart_inputs.csv    the provided file with predicted_rate filled

Both are then checked by the provided scorer:

  python score.py --predictions validation_predictions.csv \
      --december-predictions data/december_chart_inputs.csv

The model is fitted once, on the whole of train_test.csv, and used for both
outputs -- the December chart has to reflect the same model that produced the
submission, or it is not evidence about anything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA, ROOT, clean, load_raw
from .features import (
    attach_coordinates,
    attach_market_signals,
    build_city_coordinates,
    daily_market_signals,
)
from .model import make_hybrid

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
DECEMBER_PATH = DATA / "december_chart_inputs.csv"

# Column order the scorer requires, checked exactly.
SUBMISSION_COLUMNS = ["load_id", "predicted_rate"]
DECEMBER_COLUMNS = [
    "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate",
]


def prepare_december(
    december: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame
) -> pd.DataFrame:
    """Rebuild the columns december_chart_inputs.csv ships without.

    It provides seven columns and omits the coordinates and both market signals.
    Coordinates come from the city lookup, which is exact. The market signals
    come from the December rows of validation.csv, which cover all 31 days --
    so this is a lookup of conditions we were given, not a forecast.
    """
    prepared = attach_coordinates(december, build_city_coordinates(train, validation))
    prepared = attach_market_signals(
        prepared, daily_market_signals(validation[validation["date"].dt.month == 12])
    )
    prepared["weight"] = prepared["weight"].abs()
    return prepared


def write_validation_predictions(model, validation: pd.DataFrame) -> pd.DataFrame:
    """Fill the provided template rather than building a frame from scratch.

    The template fixes both the id set and their order, so filling it by
    load_id makes a mismatch impossible instead of merely unlikely.
    """
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    predicted = pd.Series(model.predict(validation), index=validation["load_id"])

    template["predicted_rate"] = template["load_id"].map(predicted)
    if template["predicted_rate"].isna().any():
        missing = int(template["predicted_rate"].isna().sum())
        raise ValueError(f"{missing} template load_id values got no prediction")

    template = template[SUBMISSION_COLUMNS]
    template.to_csv(PREDICTIONS_PATH, index=False)
    return template


def write_december_predictions(model, december: pd.DataFrame) -> pd.DataFrame:
    """Write predicted_rate back into the provided file, preserving its shape.

    The scorer checks the seven original columns in their original order, so the
    engineered columns used to make the prediction are dropped again here.
    """
    december = december.copy()
    december["predicted_rate"] = model.predict(december)

    output = december[DECEMBER_COLUMNS].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    output.to_csv(DECEMBER_PATH, index=False)
    return output


def main() -> None:
    train, train_report = clean(load_raw("train_test.csv"))
    validation, _ = clean(load_raw("validation.csv"))
    december = load_raw("december_chart_inputs.csv")

    print(f"training rows {len(train):,}  "
          f"({train_report.corrupted_flagged:,} flagged, excluded from fitting)")

    model = make_hybrid().fit(train)
    print(f"fitted {model.name}")

    submission = write_validation_predictions(model, validation)
    print(f"\nwrote {PREDICTIONS_PATH.name}: {len(submission):,} rows")
    print(f"  predicted_rate  min ${submission.predicted_rate.min():,.2f}  "
          f"mean ${submission.predicted_rate.mean():,.2f}  "
          f"max ${submission.predicted_rate.max():,.2f}")

    prepared = prepare_december(december, train, validation)
    chart = write_december_predictions(model, prepared)
    print(f"\nwrote {DECEMBER_PATH.name}: {len(chart)} rows")
    print(f"  predicted_rate  min ${chart.predicted_rate.min():,.2f}  "
          f"max ${chart.predicted_rate.max():,.2f}  "
          f"first-to-last {100 * (chart.predicted_rate.iloc[-1] / chart.predicted_rate.iloc[0] - 1):+.1f}%")


if __name__ == "__main__":
    main()
