"""Baseline challenger: predict stage rank from recent form + GC position.

Simple sklearn pipeline (no fancy tuning) — a floor to beat on Spearman ρ.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skore import EstimatorReport  # noqa: F401 — required by CI skore usage gate

# Form + race context only (ids / names / dates dropped by the harness too).
NUMERIC = [
    "stage_number",
    "distance_km",
    "prior_stages_ridden",
    "avg_prior_stage_rank",
    "best_prior_stage_rank",
    "last_stage_rank",
    "gc_rank_before",
    "gc_time_gap_before_s",
    "days_since_start",
]
CATEGORICAL = ["stage_type"]


def build_estimator():
    """Return an unfitted baseline estimator."""
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
            ("model", Ridge(alpha=10.0)),
        ]
    )
