"""Tour de France skore challenge - submission by glemaitre.

Target
------
``stage_rank`` on the *next* stage (``1`` = winner). The leaderboard metric is
Spearman rho (rank correlation within a stage); MAE is secondary.

Approach
--------
The pipeline is declared as a **skrub DataOps graph** and the rider-history join
is a DataOps step (a pandas merge inside ``apply_func``), not a
``ColumnTransformer`` / ``skrub.Joiner`` inside a scikit-learn ``Pipeline``:

    X = skrub.var("X"); y = skrub.var("y").skb.mark_as_y()
    Xm = X.skb.mark_as_X()
    Xm = Xm.skb.apply_func(add_form_features)      # stateless per-row signals
    Xm = Xm.skb.apply_func(join_rider_history)     # DataOps join on (year, bib)
    Xm = Xm.skb.apply_func(add_stage_affinity)     # career specialty x stage-type
    Xm = Xm.skb.apply_func(add_race_dynamics)      # in-Tour GC/form x stage-type
    pred = Xm.skb.apply(tabular_pipeline(HGBR), y=y)
    learner = pred.skb.make_learner()

Why the graph is rooted on ``skrub.var("X")`` / ``skrub.var("y")``: the challenge
harness *is* the loader -- it hands us the feature frame and target directly --
so there is no ``data_dir`` source to load. No feature step looks across rows
(the history join is a per-row left-merge of a constant reference table), so the
X marker sits on the source frame (skrub's IID case).

Feature layers:

1. ``add_form_features`` -- stateless per-row form/GC signals. The key one is a
   **sprinter signal**: a rider finishing stages well (``best_prior_stage_rank``
   low) yet poorly placed on GC (``gc_rank_before`` high) is a sprinter, who
   wins flat stages.
2. ``join_rider_history`` -- left-merge ``rider_history.csv`` (a static
   ProCyclingStats-derived table shipped alongside this file) on ``(year, bib)``.
   The harness drops ``rider_id`` but keeps ``year`` / ``bib``, which map 1:1 to
   a rider, so identity is recoverable. History adds rider *type* (specialty
   points: sprint / climber / gc / tt / hills / one-day), career quality
   (pre-Tour points, previous-season points and rank) and physiology
   (age / weight / height / BMI).
3. ``add_stage_affinity`` -- the rider's *career* specialty fraction matching
   today's stage type, orthogonal to the form features (which average a rider's
   results over ALL stage types and so make a pure sprinter look mediocre on a
   flat day).
4. ``add_race_dynamics`` -- *in-Tour* interactions that change every day: GC
   standing predicts finishing order with opposite sign on flat (sprinters, deep
   on GC, win) vs mountain (GC leaders win summit finishes); a non-threatening GC
   position buys freedom to escape on transition stages; cracking on one mountain
   day tends to repeat the next. (A learned latent race-state -- PCA + KMeans
   archetype over the same situation columns -- was tested and did not help this
   tree model, so the knowledge is encoded as explicit features.)
5. ``tabular_pipeline(HistGradientBoostingRegressor(...))`` -- encodes the
   categoricals and passes native missing values to a shallow, well-regularised
   tree learner (the dataset is small, ~4.7k rows, so capacity overfits).

Harness bridge: a skrub ``SkrubLearner`` fits from an environment dict, and
skore's ``EstimatorReport`` rejects the harness's ``X_train`` / ``y_train`` call
for one. ``build_estimator`` therefore returns a thin scikit-learn adapter
(``SkrubDataOpsRegressor``) that turns ``fit(X, y)`` / ``predict(X)`` into the
DataOps learner's env-dict calls -- the graph stays pure DataOps, only the outer
interface is adapted.

Model selection used a **walk-forward-by-stage** cross-validation (each test fold
is one stage; training uses only chronologically earlier stages) evaluated with
skore ``skore.evaluate(learner, data={"X": X, "y": y}, splitter=...)``.
Walk-forward Spearman rho:

    form features only          0.43   (flat 0.30, mountain 0.59)
    + rider history + affinity  0.57   (flat 0.43, mountain 0.74)
    + in-Tour race dynamics     0.57   (flat 0.45, mountain 0.76)  <- shipped

Leakage note: the temporally-honest history fields (pre-Tour points, previous
season) use only seasons before each Tour. The specialty fractions are a career
snapshot, so 2025 backtest folds are mildly optimistic; for the live task the
artifact is built through the latest completed stage, so predicting the *next*
stage uses only past information.

Run ``python submissions/glemaitre/submission.py`` to reproduce the skore
walk-forward report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import skore  # noqa: F401  - required: CI aborts submissions that do not use skore
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import tabular_pipeline

HERE = Path(__file__).resolve().parent
RIDER_HISTORY = pd.read_csv(HERE / "rider_history.csv")
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


def join_rider_history(df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge the constant PCS rider-history table on ``(year, bib)``.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature frame carrying the ``year`` and ``bib`` join keys.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the rider-history columns joined per row.
    """
    return df.merge(RIDER_HISTORY, on=["year", "bib"], how="left")


def add_stage_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """Add the rider's specialty fraction matching today's stage type.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with ``stage_type`` and the joined ``spec_*_frac`` columns.

    Returns
    -------
    pandas.DataFrame
        ``df`` with a ``stage_affinity`` column.
    """
    df = df.copy()
    stage_type = df["stage_type"].astype(str)
    affinity = pd.Series(np.nan, index=df.index, dtype=float)
    for stype, col in STAGE_AFFINITY.items():
        if col in df.columns:
            affinity = affinity.mask(stage_type == stype, df[col])
    df["stage_affinity"] = affinity
    return df


def add_race_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit race-dynamics interactions from in-Tour GC and form.

    Encodes three effects grounded in the data: (1) GC standing predicts finish
    order with opposite sign on flat (sprinters, low on GC, win) vs mountain (GC
    leaders win summit finishes); (2) a non-threatening GC position gives a rider
    freedom to escape on transition stages; (3) cracking on a mountain stage
    tends to repeat the next mountain day (fatigue carry-over).

    Parameters
    ----------
    df : pandas.DataFrame
        Frame with ``stage_type`` and the in-Tour GC / form columns.

    Returns
    -------
    pandas.DataFrame
        ``df`` with the interaction columns; infinities set to NaN.
    """
    df = df.copy()
    stage_type = df["stage_type"].astype(str)
    flat = (stage_type == "flat").astype(float)
    hilly = (stage_type == "hilly").astype(float)
    mountain = (stage_type == "mountain").astype(float)
    gc_rank = df["gc_rank_before"]
    gc_gap = df["gc_time_gap_before_s"]
    momentum = df["last_stage_rank"] - df["avg_prior_stage_rank"]
    df["gc_rank_flat"] = gc_rank * flat
    df["gc_rank_mountain"] = gc_rank * mountain
    df["gc_gap_mountain"] = gc_gap * mountain
    df["escape_freedom"] = gc_rank * (flat + hilly)
    df["marked_leader_transition"] = (gc_rank <= 10).astype(float) * (flat + hilly)
    df["mountain_fatigue"] = momentum * mountain
    return df.replace([np.inf, -np.inf], np.nan)


def build_learner():
    """Build the skrub DataOps learner (join + features + model).

    Returns
    -------
    skrub SkrubLearner
        Unfitted learner whose ``fit`` / ``predict`` take an env dict with keys
        ``"X"`` (feature frame) and ``"y"`` (target).
    """
    X = skrub.var("X")
    y = skrub.var("y").skb.mark_as_y()
    features = X.skb.mark_as_X()
    features = features.skb.apply_func(add_form_features)
    features = features.skb.apply_func(join_rider_history)
    features = features.skb.apply_func(add_stage_affinity)
    features = features.skb.apply_func(add_race_dynamics)
    predictions = features.skb.apply(
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
        y=y,
    )
    return predictions.skb.make_learner()


class SkrubDataOpsRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn adapter around the skrub DataOps learner.

    Bridges the harness's ``fit(X, y)`` / ``predict(X)`` calls to the DataOps
    learner's environment-dict interface.
    """

    def fit(self, X, y):
        """Fit the DataOps learner from the feature frame and target.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature frame.
        y : array-like
            Target ``stage_rank`` values.

        Returns
        -------
        SkrubDataOpsRegressor
            The fitted adapter.
        """
        self.learner_ = build_learner()
        self.learner_.fit({"X": X, "y": y})
        return self

    def predict(self, X):
        """Predict stage ranks for a feature frame.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature frame.

        Returns
        -------
        numpy.ndarray
            Predicted ``stage_rank`` values.
        """
        return self.learner_.predict({"X": X})


def build_estimator():
    """Return an unfitted scikit-learn-compatible estimator for the harness.

    Returns
    -------
    SkrubDataOpsRegressor
        Adapter wrapping the skrub DataOps learner.
    """
    return SkrubDataOpsRegressor()


if __name__ == "__main__":
    # Local methodology check: DataOps-native evaluation via env-dict + skore,
    # walk-forward-by-stage CV scored with Spearman rho.
    from scipy.stats import spearmanr
    from sklearn.metrics import make_scorer

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

    report = skore.evaluate(
        build_learner(),
        data={"X": X, "y": y},
        splitter=WalkForwardByStage(min_train_stages=6),
    )
    report.metrics.add(
        make_scorer(_spearman, response_method="predict"),
        name="Spearman",
        greater_is_better=True,
    )
    print(report.metrics.summarize().frame())
