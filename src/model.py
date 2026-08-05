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
