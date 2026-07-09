"""Challenger submission: predict next-stage finishing rank with skore.

Experiment cut: train through 7 July 2026 (stage 4), score stage 5 (8 July).

Uses a linear model on form + GC + stage context features — stronger than a
naive mean / last-rank baseline on this holdout.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skore import EstimatorReport  # noqa: F401 — required by CI skore usage gate


NUMERIC = [
    "year",
    "stage_number",
    "bib",
    "age",
    "distance_km",
    "prior_stages_ridden",
    "avg_prior_stage_rank",
    "best_prior_stage_rank",
    "last_stage_rank",
    "gc_rank_before",
    "gc_time_gap_before_s",
    "days_since_start",
]
CATEGORICAL = ["stage_type", "team"]


def build_estimator():
    """Return an unfitted sklearn-compatible estimator / Pipeline."""
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                RidgeCV(alphas=np.logspace(-2, 3, 25)),
            ),
        ]
    )
