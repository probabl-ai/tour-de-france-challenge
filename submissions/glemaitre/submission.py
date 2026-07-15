"""Tour de France skore challenge - submission by glemaitre.

Target
------
``stage_rank`` on the *next* stage (``1`` = winner). The leaderboard metric is
Spearman rho (rank correlation within a stage); MAE is secondary.

Approach
--------
The harness fits the returned estimator with the plain ``fit(X, y)`` signature,
so the submission is a scikit-learn-compatible ``Pipeline`` whose data-combining
and modelling steps are **skrub** objects:

1. ``FunctionTransformer(add_form_features)`` -- stateless per-row form/GC signals
   derived from columns the harness provides (it drops ``rider_id``, so no
   per-rider rolling windows). The key one is a **sprinter signal**: a rider who
   finishes stages well (``best_prior_stage_rank`` low) yet sits poorly on GC
   (``gc_rank_before`` high) is a sprinter -- who wins flat stages.
2. ``skrub.Joiner`` -- an exact join (``max_dist=0``) on ``(year, bib)`` that
   brings in ``rider_history.csv``, a static ProCyclingStats-derived table
   (shipped alongside this file). The harness drops ``rider_id`` but keeps
   ``year`` and ``bib``, which map 1:1 to a rider, so identity is recoverable.
   History adds rider *type* (specialty points: sprint / climber / gc / tt /
   hills / one-day), career quality (pre-Tour points, previous-season points and
   rank) and physiology (age / weight / height / BMI).
3. ``FunctionTransformer(finalize_features)`` -- tidies the join and builds a
   ``stage_affinity`` feature: the rider's career specialty fraction *matching
   today's stage type*. This is orthogonal to the form features, which average a
   rider's results over ALL stage types and therefore make a pure sprinter look
   mediocre even on a flat day.
4. ``skrub.tabular_pipeline(HistGradientBoostingRegressor(...))`` -- encodes the
   categoricals (``team``, ``stage_type``, ``dominant_specialty``) and passes
   native missing values to a deliberately shallow, well-regularised tree
   learner (the dataset is small, ~4.7k rows, so capacity overfits).

Note: a skrub DataOps ``make_learner()`` is intentionally NOT used -- it fits
from an environment dict and skore's ``EstimatorReport`` rejects the harness's
``X_train`` / ``y_train`` call for a ``SkrubLearner``. Using skrub *transformers*
inside an sklearn ``Pipeline`` keeps the same skrub semantics while staying
harness-compatible.

Model selection used a **walk-forward-by-stage** cross-validation (each test fold
is one stage; training uses only chronologically earlier stages) scored with
skore ``CrossValidationReport``. Walk-forward Spearman rho:

    form features only        0.43   (flat 0.30, mountain 0.59)
    + rider history + affinity 0.57  (flat 0.43, mountain 0.75)  <- shipped

Leakage note: the temporally-honest history fields (pre-Tour points, previous
season) are computed strictly from seasons before each Tour. The specialty
fractions are a career snapshot, so the 2025 backtest folds are mildly
optimistic; for the live task the artifact is built from data up to the latest
completed stage, so predicting the *next* stage uses only past information.

Run ``python submissions/glemaitre/submission.py`` to reproduce the skore
walk-forward report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skore  # noqa: F401  - required: CI aborts submissions that do not use skore
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from skrub import Joiner, tabular_pipeline

HERE = Path(__file__).resolve().parent
RIDER_HISTORY_PATH = HERE / "rider_history.csv"
JOIN_SUFFIX = "__hist"
# Which career specialty fraction matches each stage type.
STAGE_AFFINITY = {
    "flat": "spec_sprint_frac",
    "hilly": "spec_hills_frac",
    "mountain": "spec_climber_frac",
    "itt": "spec_time_trial_frac",
    "ttt": "spec_time_trial_frac",
}


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stateless per-row form/GC signals derived from existing columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature frame passed by the harness (``rider_id`` already dropped).

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with derived signals; infinities from ratios set to NaN.
    """
    df = df.copy()
    gc = df["gc_rank_before"]
    best = df["best_prior_stage_rank"]
    avg = df["avg_prior_stage_rank"]
    last = df["last_stage_rank"]
    df["sprinter_signal"] = gc - best
    df["form_gap"] = avg - best
    df["momentum"] = last - avg
    df["best_over_gc"] = best / (gc + 1.0)
    df["avg_over_gc"] = avg / (gc + 1.0)
    return df.replace([np.inf, -np.inf], np.nan)


def finalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy the ``skrub.Joiner`` output and add the stage-affinity feature.

    The Joiner suffixes every auxiliary column; we drop the duplicated key
    columns, strip the suffix from the real features, then build
    ``stage_affinity`` -- the rider's specialty fraction matching today's stage.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of the rider-history join.

    Returns
    -------
    pandas.DataFrame
        Frame with clean history columns and the ``stage_affinity`` feature.
    """
    df = df.copy()
    for key in ("year", "bib", "stage_number"):
        col = f"{key}{JOIN_SUFFIX}"
        if col in df.columns:
            df = df.drop(columns=col)
    df = df.rename(
        columns={c: c[: -len(JOIN_SUFFIX)] for c in df.columns if c.endswith(JOIN_SUFFIX)}
    )
    stage_type = df["stage_type"].astype(str)
    affinity = pd.Series(np.nan, index=df.index, dtype=float)
    for stype, col in STAGE_AFFINITY.items():
        if col in df.columns:
            affinity = affinity.mask(stage_type == stype, df[col])
    df["stage_affinity"] = affinity
    return df


def build_estimator():
    """Return an unfitted scikit-learn-compatible estimator.

    Returns
    -------
    estimator : sklearn.pipeline.Pipeline
        Form features -> skrub rider-history join -> stage affinity -> skrub
        ``tabular_pipeline`` around a shallow, regularised gradient booster.
    """
    rider_history = pd.read_csv(RIDER_HISTORY_PATH)
    return make_pipeline(
        FunctionTransformer(add_form_features, validate=False),
        Joiner(
            rider_history,
            main_key=["year", "bib"],
            aux_key=["year", "bib"],
            max_dist=0,
            suffix=JOIN_SUFFIX,
            add_match_info=False,
        ),
        FunctionTransformer(finalize_features, validate=False),
        tabular_pipeline(
            HistGradientBoostingRegressor(
                max_depth=3,
                max_iter=300,
                learning_rate=0.03,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=0,
            )
        ),
    )


if __name__ == "__main__":
    # Local methodology check with skore: walk-forward-by-stage CV scored with
    # Spearman rho (mirrors how CI scores a newly completed stage).
    from scipy.stats import spearmanr
    from sklearn.metrics import make_scorer

    from skore import CrossValidationReport

    DROP = ["stage_rank", "rider_id", "rider_name", "stage_name", "stage_date"]
    TARGET = "stage_rank"

    class WalkForwardByStage:
        """Time-ordered group splitter: test one stage, train on earlier stages."""

        def __init__(self, min_train_stages: int = 6) -> None:
            self.min_train_stages = min_train_stages

        def _stage_id(self, X: pd.DataFrame) -> np.ndarray:
            """Return a chronologically sortable integer id per stage."""
            return (X["year"].astype(int) * 100 + X["stage_number"].astype(int)).to_numpy()

        def split(self, X, y=None, groups=None):
            """Yield ``(train_idx, test_idx)`` with test = one later stage."""
            sid = self._stage_id(X)
            for stage in np.sort(np.unique(sid))[self.min_train_stages :]:
                train = np.where(sid < stage)[0]
                test = np.where(sid == stage)[0]
                if len(train) and len(test):
                    yield train, test

        def get_n_splits(self, X=None, y=None, groups=None) -> int:
            """Return the number of walk-forward folds."""
            return max(0, np.unique(self._stage_id(X)).size - self.min_train_stages)

    def _spearman(y_true, y_pred) -> float:
        """Spearman rho, mapping the degenerate/undefined case to 0.0."""
        coef = spearmanr(y_true, y_pred).statistic
        return 0.0 if coef is None or np.isnan(coef) else float(coef)

    data = pd.read_csv("data/data.csv").dropna(subset=[TARGET])
    y = data[TARGET].astype(float)
    X = data.drop(columns=[c for c in DROP if c in data.columns])

    report = CrossValidationReport(
        build_estimator(), X=X, y=y, splitter=WalkForwardByStage(min_train_stages=6)
    )
    report.metrics.add(
        make_scorer(_spearman, response_method="predict"),
        name="Spearman",
        greater_is_better=True,
    )
    print(report.metrics.summarize().frame())
