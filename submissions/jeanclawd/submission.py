"""Tour de France stage-rank submission — jeanclawd.

Target: ``stage_rank`` (finishing position on the next stage; 1 = winner).
Metric: Spearman ρ within the scored stage (higher is better).

Methodology (Probabl skore / skrub skills):
- Explored ``data.csv`` with a leave-one-stage-out (group = stage) protocol,
  scoring each held-out stage with Spearman ρ — the challenge metric — because
  rows are grouped by stage and a plain random split would leak same-stage rows.
- Per-stage feature analysis showed rider-quality signals dominate the ranking:
  ``gc_rank_before``, ``gc_time_gap_before_s``, ``avg_prior_stage_rank`` and
  ``last_stage_rank`` are the strongest predictors, especially on the
  GC-selective stages (mountain / ITT) that make up most of the remaining Tour.
- We add two engineered "form" features that summarise a rider's standing
  (mean / best of their GC rank and recent finishing ranks). These lift
  mountain-stage ρ materially at a negligible cost elsewhere.
- Model: skrub ``TableVectorizer`` (native categorical handling for ``team`` /
  ``stage_type``) + ``HistGradientBoostingRegressor`` (native NaN support — GC
  gaps are missing on early stages). Shallow trees + many low-LR iterations
  keep the ranking well-calibrated without overfitting the ~30 stages of data.

The challenge CI fits this estimator on the full labelled history and scores it
with a skore ``EstimatorReport``; running this file directly reproduces the
leave-one-stage-out evaluation with skore.
"""

from __future__ import annotations

import skore  # noqa: F401 — required: CI aborts submissions that do not use skore
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

# Rider-standing / recent-form columns (lower = stronger rider).
_FORM_COLS = ["gc_rank_before", "avg_prior_stage_rank", "last_stage_rank"]
_BEST_COLS = ["gc_rank_before", "best_prior_stage_rank", "last_stage_rank"]


class FormFeatures(BaseEstimator, TransformerMixin):
    """Add compact rider-standing summaries to the feature frame.

    ``form_mean`` averages a rider's GC rank and recent stage ranks; ``best_form``
    takes their best (min). Both give the model a clean monotonic "how good is
    this rider right now" signal, which is what orders GC-selective stages.
    """

    def fit(self, X, y=None):  # noqa: D102, ARG002
        return self

    def transform(self, X):  # noqa: D102
        X = X.copy()
        X["form_mean"] = X[_FORM_COLS].mean(axis=1)
        X["best_form"] = X[_BEST_COLS].min(axis=1)
        return X


def build_estimator():
    """Return an unfitted sklearn-compatible estimator / Pipeline."""
    return make_pipeline(
        FormFeatures(),
        TableVectorizer(),
        HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=500,
            learning_rate=0.03,
            l2_regularization=1.0,
            random_state=0,
        ),
    )


if __name__ == "__main__":
    # Local methodology check: leave-one-stage-out Spearman ρ via skore.
    import warnings

    import pandas as pd
    from scipy.stats import spearmanr

    warnings.filterwarnings("ignore")

    DROP = {"stage_rank", "rider_id", "rider_name", "stage_name", "stage_date"}
    data = pd.read_csv("data/data.csv").dropna(subset=["stage_rank"])
    scores = []
    for (yr, st), grp in data.groupby(["year", "stage_number"]):
        if len(grp) < 10:
            continue
        train = data.drop(index=grp.index)
        Xtr = train.drop(columns=[c for c in DROP if c in train])
        ytr = train["stage_rank"].astype(float)
        Xte = grp.drop(columns=[c for c in DROP if c in grp]).reindex(columns=Xtr.columns)
        yte = grp["stage_rank"].astype(float)
        report = skore.EstimatorReport(
            build_estimator(), X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte
        )
        pred = report.estimator_.predict(Xte)
        scores.append((grp["stage_type"].iloc[0], spearmanr(yte, pred).statistic))
    df = pd.DataFrame(scores, columns=["stage_type", "rho"])
    print(f"Leave-one-stage-out Spearman rho: mean={df['rho'].mean():.3f}")
    print(df.groupby("stage_type")["rho"].mean().round(3).to_string())
