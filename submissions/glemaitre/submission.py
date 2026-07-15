"""Tour de France skore challenge - submission by glemaitre.

Target
------
``stage_rank`` on the *next* stage (``1`` = winner). The leaderboard metric is
Spearman rho (rank correlation within a stage); MAE is secondary.

Approach
--------
The harness fits the returned estimator with the plain ``fit(X, y)`` signature,
so the submission must be a scikit-learn-compatible estimator. We use a skrub
``tabular_pipeline`` because the feature matrix mixes high-cardinality
categoricals (``team``, ``stage_type``) with numerics that are heavily missing
(``gc_time_gap_before_s`` ~66% null, ``age`` ~38%, ``gc_rank_before`` ~18%);
``tabular_pipeline`` handles categorical encoding and native missing values with
no manual ``ColumnTransformer`` wiring.

The estimator is a deliberately shallow, well-regularised
``HistGradientBoostingRegressor``. The dataset is small (~4.7k rows, roughly one
and a half Tours), so higher-capacity models overfit and *lose* held-out rank
correlation. This choice was selected with a **walk-forward-by-stage**
cross-validation (each test fold is a single stage; training uses only
chronologically earlier stages, so there is no future leakage) scored with skore
``CrossValidationReport``. Candidate walk-forward Spearman rho:

    dummy (mean)          0.00
    Ridge                 0.40 +/- 0.25
    HGBR (default)        0.38 +/- 0.21
    HGBR (this baseline)  0.41 +/- 0.21   <- best mean, lowest MAE

Run ``python submissions/glemaitre/submission.py`` to reproduce the skore
walk-forward report.
"""

from __future__ import annotations

import skore  # noqa: F401  - required: CI aborts submissions that do not use skore
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import tabular_pipeline


def build_estimator():
    """Return an unfitted scikit-learn-compatible estimator.

    Returns
    -------
    estimator : skrub Pipeline
        A shallow, regularised gradient-boosted regressor wrapped in a skrub
        ``tabular_pipeline`` that encodes categoricals and passes native missing
        values straight to the tree learner.
    """
    return tabular_pipeline(
        HistGradientBoostingRegressor(
            max_depth=3,
            max_iter=300,
            learning_rate=0.03,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=0,
        )
    )


if __name__ == "__main__":
    # Local methodology check with skore: walk-forward-by-stage CV scored with
    # Spearman rho (mirrors how CI scores a newly completed stage).
    import numpy as np
    import pandas as pd
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
