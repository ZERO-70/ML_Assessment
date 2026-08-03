"""Loading and cleaning for the freight rate data.

The four issues found in notebooks/01_exploratory_analysis.ipynb:

  1. `weight` carries sign-flip entry errors (negative values whose magnitudes
     sit inside the valid range) -> absolute value.
  2. `weight` and `market_index` have missing values, roughly twice as often in
     validation as in training -> imputed, each on its own terms.
  3. ~1.4% of training `posted_rate` values are multiplied by ~3 or ~1/3
     -> flagged against an equipment x distance peer group.
  4. Eight cities and 736 lanes in validation never appear in training. That is
     a feature-design problem, not a cleaning one, and is handled in features.py.

Cleaning never uses the target, so it applies unchanged to validation data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DISTANCE_BINS = [0, 250, 500, 1000, 1500, 2500, np.inf]

BASELINE_RATE_PER_MILE = (1.0, 4.0)

CORRUPTION_BOUNDS = (0.5, 2.0)


@dataclass
class CleaningReport:
    """What the cleaner changed, so a notebook can show it rather than assert it."""

    rows: int
    weight_sign_fixed: int
    weight_imputed: int
    market_index_imputed: int
    market_index_from_date: int
    corrupted_flagged: int = 0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"count": list(vars(self).values())}, index=list(vars(self).keys())
        )


def load_raw(name: str) -> pd.DataFrame:
    """Read one of the provided CSVs with `date` parsed."""
    return pd.read_csv(DATA / name, parse_dates=["date"])


def fix_weight(frame: pd.DataFrame) -> tuple[pd.Series, int, int]:
    """Undo sign-flip errors, then fill the gaps.

    Weight correlates only weakly with the rate per mile (r = 0.09) and the
    median is nearly identical across equipment types (31.4k-31.6k lb), so a
    single median fill is enough — a more elaborate scheme would add complexity
    without changing predictions.
    """
    weight = frame["weight"]
    sign_fixed = int((weight < 0).sum())
    weight = weight.abs()

    to_impute = int(weight.isna().sum())
    weight = weight.fillna(weight.median())
    return weight, sign_fixed, to_impute


def fix_market_index(frame: pd.DataFrame) -> tuple[pd.Series, int, int]:
    """Fill missing `market_index` from the same day's other loads.

    `market_index` behaves as a daily market level plus small per-row noise: its
    within-date standard deviation never exceeds 0.03 against a range of
    0.68-1.47. So other loads on the same date pin down a missing value almost
    exactly, which is far better than any global statistic.
    """
    market_index = frame["market_index"]
    missing_before = int(market_index.isna().sum())

    daily_mean = frame.groupby("date")["market_index"].transform("mean")
    market_index = market_index.fillna(daily_mean)
    from_date = missing_before - int(market_index.isna().sum())

    # Only bites if an entire date were missing, which does not occur here.
    market_index = market_index.fillna(market_index.median())
    return market_index, missing_before, from_date


def flag_corrupted(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Flag training rows whose `posted_rate` was multiplied by ~3 or ~1/3.

    Anomaly is judged against comparable loads rather than the global
    distribution: a 300-mile Reefer and a 2,400-mile Dry Van are priced
    differently per mile. The baseline is built only from rows inside
    BASELINE_RATE_PER_MILE so the corrupted rows cannot contaminate the very
    yardstick used to detect them.

    Returns (is_corrupted, ratio) so callers can inspect the margin, not just
    the verdict.
    """
    rate_per_mile = frame["posted_rate"] / frame["distance"]
    distance_bin = pd.cut(frame["distance"], DISTANCE_BINS)

    trusted = rate_per_mile.between(*BASELINE_RATE_PER_MILE)
    baseline = (
        rate_per_mile[trusted]
        .groupby([frame.loc[trusted, "equipment"], distance_bin[trusted]], observed=True)
        .median()
    )
    expected = pd.MultiIndex.from_arrays([frame["equipment"], distance_bin]).map(baseline)
    ratio = pd.Series(rate_per_mile.to_numpy() / expected.to_numpy(), index=frame.index)

    low, high = CORRUPTION_BOUNDS
    return ~ratio.between(low, high), ratio


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply every fix. Adds `is_corrupted`/`rate_ratio` when a target is present."""
    cleaned = frame.copy()

    cleaned["weight"], sign_fixed, weight_imputed = fix_weight(cleaned)
    cleaned["market_index"], market_imputed, from_date = fix_market_index(cleaned)

    report = CleaningReport(
        rows=len(cleaned),
        weight_sign_fixed=sign_fixed,
        weight_imputed=weight_imputed,
        market_index_imputed=market_imputed,
        market_index_from_date=from_date,
    )

    if "posted_rate" in cleaned.columns:
        cleaned["is_corrupted"], cleaned["rate_ratio"] = flag_corrupted(cleaned)
        report.corrupted_flagged = int(cleaned["is_corrupted"].sum())

    return cleaned, report
