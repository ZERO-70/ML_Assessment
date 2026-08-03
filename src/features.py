"""Feature engineering for the freight rate model.

Three constraints from notebooks/01_exploratory_analysis.ipynb drive the design:

  1. Eight cities and 736 lanes in validation never appear in training, so
     nothing may key on the raw city or lane string. Every geographic feature is
     derived from coordinates, which are a clean per-city lookup and place the
     unseen cities inside the envelope of the known ones.

  2. A ~7% secular trend survives `market_index` and `quote_signal`, and the
     December chart varies nothing but the date. Time therefore enters as
     `days_since_origin`, a single continuous term a linear component can
     project past the last training date. Cyclical effects are kept separate as
     day-of-week, so the trend term stays clean.

  3. `december_chart_inputs.csv` has no coordinates and no market signals.
     Both are reconstructed from data we were given -- coordinates from the
     city lookup, market signals from the December rows of validation.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_MILES = 3958.7613

# All time features are measured from the first training date so that train,
# validation and the December chart share one origin.
TIME_ORIGIN = pd.Timestamp("2025-01-01")

EQUIPMENT_TYPES = ("Dry Van", "Flatbed", "Reefer")

FEATURE_COLUMNS = [
    # load
    "distance",
    "log_distance",
    "weight",
    "weight_per_mile",
    # geography
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_distance",
    "circuity",
    "bearing_sin",
    "bearing_cos",
    # market
    "market_index",
    "quote_signal",
    # time
    "days_since_origin",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    # equipment, one-hot
    *[f"equipment_{name.replace(' ', '_').lower()}" for name in EQUIPMENT_TYPES],
]


def haversine_miles(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Great-circle distance in miles."""
    lat1_r, lon1_r, lat2_r, lon2_r = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2_r - lat1_r) / 2) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin((lon2_r - lon1_r) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def bearing_degrees(
    lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series
) -> pd.Series:
    """Initial compass bearing from origin to destination.

    Direction of travel matters in freight: a lane and its reverse price
    differently because trucks reposition toward high-demand regions. Encoded
    as sin/cos downstream so that 359 degrees and 1 degree are adjacent.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    delta_lon = lon2_r - lon1_r
    y = np.sin(delta_lon) * np.cos(lat2_r)
    x = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(delta_lon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def build_city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Map every city name to its coordinates.

    Each city resolves to exactly one coordinate pair across the whole dataset
    (verified in the EDA), so pickup and delivery rows can be pooled. Uses only
    feature columns, never the target, so it is safe to build from training and
    validation together.
    """
    parts = []
    for frame in frames:
        for role in ("pickup", "delivery"):
            if f"{role}_lat" not in frame.columns:
                continue
            parts.append(
                frame[[role, f"{role}_lat", f"{role}_lon"]].rename(
                    columns={role: "city", f"{role}_lat": "lat", f"{role}_lon": "lon"}
                )
            )
    return pd.concat(parts).drop_duplicates(subset="city").set_index("city")


def attach_coordinates(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing coordinate columns from the city lookup.

    Needed for december_chart_inputs.csv, which ships without them.
    """
    result = frame.copy()
    for role in ("pickup", "delivery"):
        if f"{role}_lat" in result.columns:
            continue
        result[f"{role}_lat"] = result[role].map(lookup["lat"])
        result[f"{role}_lon"] = result[role].map(lookup["lon"])
    return result


def attach_market_signals(frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing `market_index`/`quote_signal` from per-date values.

    `daily` is indexed by date with those two columns -- for the December chart
    it comes from the December rows of validation.csv, where every one of the
    31 days is present.
    """
    result = frame.copy()
    for column in ("market_index", "quote_signal"):
        if column in result.columns:
            continue
        result[column] = result["date"].map(daily[column])
    return result


def daily_market_signals(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-date mean of the two market signals."""
    return frame.groupby("date")[["market_index", "quote_signal"]].mean()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered column. Expects cleaned input with coordinates present."""
    result = frame.copy()

    result["log_distance"] = np.log(result["distance"])
    result["weight_per_mile"] = result["weight"] / result["distance"]

    result["haversine_distance"] = haversine_miles(
        result["pickup_lat"], result["pickup_lon"],
        result["delivery_lat"], result["delivery_lon"],
    )
    # Road distance runs a consistent ~1.18x the straight line. Departures from
    # that carry route information the raw mileage does not.
    result["circuity"] = result["distance"] / result["haversine_distance"]

    bearing = np.radians(
        bearing_degrees(
            result["pickup_lat"], result["pickup_lon"],
            result["delivery_lat"], result["delivery_lon"],
        )
    )
    result["bearing_sin"] = np.sin(bearing)
    result["bearing_cos"] = np.cos(bearing)

    # A single continuous trend term, so a linear component can extrapolate it
    # into November and December rather than flat-lining at the last leaf.
    result["days_since_origin"] = (result["date"] - TIME_ORIGIN).dt.days

    # The weekly cycle is genuinely periodic and must NOT be extrapolated, so it
    # is encoded separately from the trend.
    day_of_week = result["date"].dt.dayofweek
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    result["is_weekend"] = (day_of_week >= 5).astype(int)

    for name in EQUIPMENT_TYPES:
        column = f"equipment_{name.replace(' ', '_').lower()}"
        result[column] = (result["equipment"] == name).astype(int)

    return result


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the model input columns in a fixed order."""
    return frame[FEATURE_COLUMNS]
