"""Temporal validation harness.

The real task is fixed: train on 2025-01-01..2025-10-31, then predict the next
two months. Any honest local estimate of that has to have the same shape, so the
folds here are rolling-origin: train on everything up to a cutoff, test on the
two months that follow, never the reverse. A random split would let the model
see September while predicting July and would report a score we could not trust.

On which rows to score: ~1.4% of training labels are corrupted (see data.py).
The graders' held-out labels are presumably contaminated the same way, so the
headline metrics are computed on every row. Metrics on the clean subset are
reported alongside, because that is the only number that reflects real skill --
no model can predict a label that was multiplied by three at random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Matches the real forecast horizon: validation spans November and December.
TEST_HORIZON_DAYS = 61

# Three folds fit inside the training window while leaving each one a six-month
# minimum of history to learn the trend from.
DEFAULT_FOLDS = 3
MIN_TRAIN_DAYS = 181


@dataclass(frozen=True)
class Fold:
    """One rolling-origin split, described by dates rather than row positions."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def masks(self, dates: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Boolean masks selecting this fold's training and test rows."""
        train = dates.between(self.train_start, self.train_end)
        test = dates.between(self.test_start, self.test_end)
        return train, test


def rolling_origin_folds(
    dates: pd.Series,
    n_folds: int = DEFAULT_FOLDS,
    horizon_days: int = TEST_HORIZON_DAYS,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> list[Fold]:
    """Build expanding-window folds, each testing on `horizon_days` of future.

    The training window always starts at the first date and grows; only the
    cutoff moves. That mirrors the real setup, where the model is fitted on all
    available history before forecasting forward.
    """
    start, end = dates.min(), dates.max()
    last_cutoff = end - pd.Timedelta(days=horizon_days)
    earliest_cutoff = start + pd.Timedelta(days=min_train_days)

    if last_cutoff < earliest_cutoff:
        raise ValueError("date range too short for the requested horizon and history")

    cutoffs = pd.date_range(earliest_cutoff, last_cutoff, periods=n_folds)

    folds = []
    for index, cutoff in enumerate(cutoffs, start=1):
        cutoff = cutoff.normalize()
        folds.append(
            Fold(
                index=index,
                train_start=start,
                train_end=cutoff,
                test_start=cutoff + pd.Timedelta(days=1),
                test_end=min(cutoff + pd.Timedelta(days=horizon_days), end),
            )
        )
    return folds


def describe_folds(folds: list[Fold], dates: pd.Series) -> pd.DataFrame:
    """Row counts and spans per fold, for eyeballing before trusting any score."""
    rows = []
    for fold in folds:
        train_mask, test_mask = fold.masks(dates)
        rows.append(
            {
                "fold": fold.index,
                "train": f"{fold.train_start.date()} -> {fold.train_end.date()}",
                "train_rows": int(train_mask.sum()),
                "test": f"{fold.test_start.date()} -> {fold.test_end.date()}",
                "test_rows": int(test_mask.sum()),
                "gap_days": (fold.test_start - fold.train_end).days,
            }
        )
    return pd.DataFrame(rows)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error metrics for a set of rate predictions.

    RMSE is reported because it is the usual default, but MAE and median APE
    carry more signal here: with ~1.4% of labels multiplied by three, RMSE
    largely measures how much corruption happened to land in the test window.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    absolute_percentage = np.abs(error) / actual

    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(absolute_percentage)),
        "median_ape": float(np.median(absolute_percentage)),
        "bias": float(np.mean(error)),
    }


def score_split(
    actual: np.ndarray, predicted: np.ndarray, is_corrupted: np.ndarray
) -> dict[str, float]:
    """Score on every row, and again on the uncorrupted subset.

    Both matter. The `all_` numbers are comparable to what the graders will see;
    the `clean_` numbers say whether the model is actually any good.
    """
    is_corrupted = np.asarray(is_corrupted, dtype=bool)
    metrics = {f"all_{name}": value for name, value in score(actual, predicted).items()}
    clean = score(actual[~is_corrupted], predicted[~is_corrupted])
    metrics.update({f"clean_{name}": value for name, value in clean.items()})
    metrics["corrupted_in_test"] = float(is_corrupted.mean())
    return metrics


def cross_validate(
    frame: pd.DataFrame,
    fit_predict,
    folds: list[Fold] | None = None,
    target: str = "posted_rate",
) -> pd.DataFrame:
    """Run `fit_predict` across every fold and collect per-fold metrics.

    `fit_predict(train_frame, test_frame)` must fit only on `train_frame` and
    return predictions for `test_frame`. Keeping the signature that narrow is
    deliberate: a model cannot accidentally reach data it should not see.
    """
    folds = folds or rolling_origin_folds(frame["date"])
    is_corrupted = (
        frame["is_corrupted"] if "is_corrupted" in frame.columns
        else pd.Series(False, index=frame.index)
    )

    results = []
    for fold in folds:
        train_mask, test_mask = fold.masks(frame["date"])
        train_frame, test_frame = frame[train_mask], frame[test_mask]

        predicted = np.asarray(fit_predict(train_frame, test_frame), dtype=float)
        if len(predicted) != len(test_frame):
            raise ValueError(
                f"fold {fold.index}: got {len(predicted)} predictions "
                f"for {len(test_frame)} test rows"
            )

        metrics = score_split(
            test_frame[target].to_numpy(), predicted, is_corrupted[test_mask].to_numpy()
        )
        results.append({"fold": fold.index, "test_rows": len(test_frame), **metrics})

    frame_out = pd.DataFrame(results)
    mean_row = frame_out.drop(columns=["fold"]).mean()
    mean_row["fold"] = "mean"
    return pd.concat([frame_out, mean_row.to_frame().T], ignore_index=True)


def out_of_fold_predictions(
    frame: pd.DataFrame,
    fit_predict,
    folds: list[Fold] | None = None,
    target: str = "posted_rate",
) -> pd.DataFrame:
    """Collect every fold's test-set predictions into one frame.

    An average metric says how large the error is; it does not say *where*. This
    returns the individual predictions alongside their original rows, so error
    can be broken down by distance, equipment, date or any other column.

    Every prediction here comes from a model that never saw that row while
    fitting -- the same guarantee `cross_validate` relies on. Note the folds do
    not cover the first six months, which are training-only in every fold, so
    this is a subset of `frame` rather than all of it.
    """
    folds = folds or rolling_origin_folds(frame["date"])

    parts = []
    for fold in folds:
        train_mask, test_mask = fold.masks(frame["date"])
        test_frame = frame[test_mask]
        predicted = np.asarray(fit_predict(frame[train_mask], test_frame), dtype=float)

        part = test_frame.copy()
        part["fold"] = fold.index
        part["predicted"] = predicted
        part["error"] = predicted - test_frame[target].to_numpy()
        part["absolute_percentage_error"] = (
            part["error"].abs() / test_frame[target].to_numpy()
        )
        parts.append(part)

    return pd.concat(parts)
