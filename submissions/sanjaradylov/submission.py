"""Quick baseline for the Tour de France skore challenge by @sanjaradylov."""

from __future__ import annotations

import pandas as pd
import skore  # noqa: F401  — required: CI aborts submissions that do not use skore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import MissingIndicator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder
from skrub import ApplyToCols, selectors as s, TableVectorizer


class RaceFeatureBuilder(TransformerMixin, BaseEstimator):
    """Adds domain features."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        avg_rank = df["avg_prior_stage_rank"]
        best_rank = df["best_prior_stage_rank"]
        last_rank = df["last_stage_rank"]
        gc_rank = df["gc_rank_before"]
        prior_stages = df["prior_stages_ridden"]
        stage_number = df["stage_number"]
        distance_km = df["distance_km"]

        df["recent_form_delta_"] = last_rank - avg_rank
        df["consistency_gap_"] = avg_rank - best_rank
        df["gc_vs_recent_form_"] = gc_rank - avg_rank
        df["distance_per_stage_"] = distance_km / (prior_stages + 1.0)
        df["progress_ratio_"] = stage_number / (prior_stages + 1.0)

        return df


def build_estimator():
    """Returns an unfitted skrub-sklearn baseline."""
    return make_pipeline(
        ApplyToCols(MissingIndicator(features="all"), cols=s.has_nulls(proportion=0.3)),
        RaceFeatureBuilder(),
        TableVectorizer(cardinality_threshold=30, high_cardinality=TargetEncoder()),
        HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,
            max_iter=300,
            max_depth=3,
            random_state=42,
        ),
    )
