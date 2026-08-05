"""Candidate rate models.

Every model here shares three decisions that came out of the EDA:

  * **Fit in log space.** `posted_rate` is right-skewed (1.90) and near-symmetric
    in logs (-0.49), and freight prices multiplicatively. Fitting `log(rate)`
    makes the error proportional rather than absolute, so a $200 miss on a
    short haul counts as heavily as a $2,000 miss on a long one.

  * **Drop flagged rows when fitting.** ~1.4% of training labels were multiplied
    by ~3 or ~1/3. Those rows are excluded from fitting but never from scoring.

  * **One feature contract.** All models consume `features.FEATURE_COLUMNS`, so
    a comparison between them is a comparison of the model, not of the inputs.

The two candidates differ in exactly one way that matters here: whether they can
extrapolate the time trend past the last training date. That difference is the
point of the comparison, not an incidental detail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, add_features, feature_matrix

RANDOM_STATE = 0

# The only feature whose November and December values fall outside the range
# observed during training, and therefore the only one a tree cannot handle.
# HybridModel can optionally withhold it from its boosted stage.
TREND_COLUMN = "days_since_origin"


class RateModel:
    """Wraps a scikit-learn estimator with the shared fitting protocol.

    Takes cleaned frames in and returns dollar predictions out, so callers never
    handle the log transform or the feature build themselves -- the training
    path and the prediction path cannot drift apart.
    """

    def __init__(self, estimator, name: str, exclude_corrupted: bool = True):
        self.estimator = estimator
        self.name = name
        self.exclude_corrupted = exclude_corrupted

    def fit(self, frame: pd.DataFrame) -> "RateModel":
        fitting = frame
        if self.exclude_corrupted and "is_corrupted" in frame.columns:
            fitting = frame[~frame["is_corrupted"]]

        features = feature_matrix(add_features(fitting))
        target = np.log(fitting["posted_rate"].to_numpy())
        self.estimator.fit(features, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = feature_matrix(add_features(frame))
        return np.exp(self.estimator.predict(features))

    def fit_predict(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
        """Adapter matching the signature `evaluation.cross_validate` expects."""
        return self.fit(train_frame).predict(test_frame)


def make_linear(exclude_corrupted: bool = True) -> RateModel:
    """Ridge regression on standardised features.

    Deliberately the simple option, and the only one of the two that can
    extrapolate: `days_since_origin` enters as a coefficient, so a date beyond
    the training window still moves the prediction in the right direction.
    Ridge rather than plain least squares because `distance`, `log_distance`
    and `haversine_distance` are near-collinear by construction.
    """
    estimator = Pipeline(
        [("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
    )
    return RateModel(estimator, "ridge (log target)", exclude_corrupted)


def make_boosted(exclude_corrupted: bool = True) -> RateModel:
    """Gradient-boosted trees.

    Expected to win on the rolling folds, where every test date sits inside the
    range of dates it was trained on. Its weakness is structural rather than a
    tuning problem: a tree splits on thresholds it saw during training, so any
    date past the last training day lands in the same terminal leaf as the last
    day it did see, and the learned trend simply stops.
    """
    estimator = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return RateModel(estimator, "gradient boosting (log target)", exclude_corrupted)


class HybridModel:
    """Ridge for the trend, gradient boosting for everything it leaves behind.

    Neither candidate alone is adequate: ridge extrapolates the time trend but
    underfits load-specific structure, boosting does the reverse. Fitting them
    in sequence gets both, because the two failures are in different places.

    Stage 1 fits `log(rate)` with ridge, including the trend term.
    Stage 2 fits stage 1's *residual* with gradient boosting -- the same
    residual-correction idea boosting already uses internally, applied once
    more with a linear model as the first learner.

    The prediction is `exp(stage 1 + stage 2)`.

    Stage 2 *keeps* `days_since_origin`, which is not the obvious choice. The
    intuition against it is that a tree splitting on raw time re-learns the
    trend as thresholds, and thresholds stop at the last training date. That
    intuition is wrong here, and measurably so: stage 1 has already removed the
    trend, so what stage 2 sees is a residual with no trend left to extrapolate.
    Flat-lining a flat signal costs nothing.

    Withholding it, meanwhile, costs a lot -- the tree loses the residual time
    structure that *is* in range. Measured on the rolling folds:

        stage 2 without the trend column   MAE $68.63   median APE 2.36%
        stage 2 with the trend column      MAE $61.80   median APE 2.06%

    and both extrapolate identically (+2.1% vs +2.0% from October to December).
    `trend_in_residual_stage=False` reproduces the withheld variant.
    """

    def __init__(self, exclude_corrupted: bool = True, trend_in_residual_stage: bool = True):
        self.name = "hybrid (ridge trend + boosted residual)"
        self.exclude_corrupted = exclude_corrupted
        self.trend_in_residual_stage = trend_in_residual_stage
        self.trend_stage = Pipeline(
            [("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))]
        )
        self.residual_stage = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )

    def _residual_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.trend_in_residual_stage:
            return features
        return features.drop(columns=[TREND_COLUMN])

    def fit(self, frame: pd.DataFrame) -> "HybridModel":
        fitting = frame
        if self.exclude_corrupted and "is_corrupted" in frame.columns:
            fitting = frame[~frame["is_corrupted"]]

        features = feature_matrix(add_features(fitting))
        target = np.log(fitting["posted_rate"].to_numpy())

        self.trend_stage.fit(features, target)
        residual = target - self.trend_stage.predict(features)
        self.residual_stage.fit(self._residual_features(features), residual)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        features = feature_matrix(add_features(frame))
        log_rate = self.trend_stage.predict(features) + self.residual_stage.predict(
            self._residual_features(features)
        )
        return np.exp(log_rate)

    def fit_predict(self, train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
        return self.fit(train_frame).predict(test_frame)


def make_hybrid(exclude_corrupted: bool = True) -> HybridModel:
    """The submission model."""
    return HybridModel(exclude_corrupted)


def median_rate_per_mile(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> np.ndarray:
    """The naive reference: one global $/mile, ignoring every other feature."""
    rate = (train_frame["posted_rate"] / train_frame["distance"]).median()
    return (test_frame["distance"] * rate).to_numpy()


def coefficient_table(model: RateModel) -> pd.DataFrame:
    """Standardised Ridge coefficients, largest absolute effect first.

    Coefficients are on standardised inputs, so they are directly comparable:
    each is the change in log(rate) per one standard deviation of that feature.
    """
    ridge = model.estimator.named_steps["ridge"]
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient": ridge.coef_})
        .assign(magnitude=lambda frame: frame["coefficient"].abs())
        .sort_values("magnitude", ascending=False)
        .drop(columns="magnitude")
        .reset_index(drop=True)
    )
