"""Skeleton submission for the Tour de France skore challenge.

Copy this folder to ``submissions/<your-github-login>/`` and implement
``build_estimator``.

Recommended: install Probabl skills so your agent follows skore methodology:

    npx skills add probabl-ai/skills
    # or: pip install skore-cli && skore skills install

See https://github.com/probabl-ai/skills
"""

from __future__ import annotations

import skore  # noqa: F401  — required: CI aborts submissions that do not use skore
from sklearn.ensemble import HistGradientBoostingRegressor
from skrub import tabular_pipeline


def build_estimator():
    """Return an unfitted sklearn-compatible estimator / Pipeline.

    The CI harness will:
    1. call ``build_estimator()``
    2. fit on ``data/data.csv`` (features + ``stage_rank``)
    3. evaluate on the newly completed stage (or a holdout) via skore ``EstimatorReport``
    4. optionally publish the report to Skore Hub
    """
    # skore is imported above so the usage gate passes; use it locally when iterating:
    #   from skore import evaluate
    #   report = evaluate(build_estimator(), X, y, splitter=0.2)
    return tabular_pipeline(HistGradientBoostingRegressor(max_depth=3, max_iter=100))
