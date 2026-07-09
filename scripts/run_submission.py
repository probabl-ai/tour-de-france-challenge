#!/usr/bin/env python3
"""Load a submission estimator, evaluate on data/next_stage labels, optionally publish.

Training set is always ``data/data.csv`` (labeled history through the latest
completed stage). Scoring uses ``data/test.csv`` when the nightly job just
discovered a newly completed stage; otherwise PR dry-runs hold out the latest
stage inside ``data.csv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

DROP_COLUMNS = {
    "stage_rank",
    "rider_id",
    "rider_name",
    "stage_name",
    "stage_date",
}
TARGET = "stage_rank"


def _load_module(submission_dir: Path):
    path = submission_dir / "submission.py"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    spec = importlib.util.spec_from_file_location("challenge_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["challenge_submission"] = module
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"missing target column {TARGET!r}")
    labeled = df.dropna(subset=[TARGET]).copy()
    if labeled.empty:
        raise ValueError("no labeled rows available")
    y = labeled[TARGET].astype(float)
    X = labeled.drop(columns=[c for c in DROP_COLUMNS if c in labeled.columns])
    return X, y


def _holdout_latest_stage(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit on all but the latest stage of the latest year; score on that stage."""
    labeled = data.dropna(subset=[TARGET])
    if labeled.empty:
        raise ValueError("data.csv has no labeled rows")
    year = int(labeled["year"].max())
    year_rows = labeled[labeled["year"] == year]
    latest = int(year_rows["stage_number"].max())
    test_mask = (labeled["year"] == year) & (labeled["stage_number"] == latest)
    test = labeled.loc[test_mask].copy()
    train = labeled.loc[~test_mask].copy()
    if train.empty:
        train = labeled.sample(frac=0.8, random_state=0)
        test = labeled.drop(index=train.index)
    return train, test


def build_report(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    from skore import EstimatorReport

    return EstimatorReport(
        estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def publish_report(report: Any, key: str, workspace: str, project: str) -> None:
    from skore import Project, login

    login(mode="hub")
    hub_project = Project(name=f"{workspace}/{project}", mode="hub")
    hub_project.put(key, report)
    print(f"published report key={key!r} to {workspace}/{project}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, help="Path to submissions/<login>/")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish EstimatorReport to Skore Hub (requires SKORE_API_KEY)",
    )
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="If test.csv is missing, hold out the latest stage from data.csv",
    )
    parser.add_argument(
        "--hub-key",
        default=None,
        help="Hub put key (default: submission folder name)",
    )
    args = parser.parse_args()
    submission_dir = args.submission.resolve()
    key = args.hub_key or submission_dir.name

    module = _load_module(submission_dir)
    if not hasattr(module, "build_estimator"):
        print("error: submission.py must define build_estimator()", file=sys.stderr)
        return 1
    estimator = module.build_estimator()
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        print("error: build_estimator() must return an object with fit/predict", file=sys.stderr)
        return 1

    data_df = _read_csv(DATA_DIR / "data.csv")
    test_path = DATA_DIR / "test.csv"

    if test_path.exists() and test_path.stat().st_size > 0:
        test_df = _read_csv(test_path)
        # Fit on data.csv excluding the score stage (avoid leakage).
        if not test_df.empty and "stage_number" in test_df.columns:
            score_stage = int(test_df["stage_number"].iloc[0])
            score_year = int(test_df["year"].iloc[0])
            train_df = data_df[
                ~((data_df["year"] == score_year) & (data_df["stage_number"] == score_stage))
            ]
        else:
            train_df = data_df
        try:
            X_train, y_train = _xy(train_df)
            X_test, y_test = _xy(test_df)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        if not args.allow_holdout:
            print(
                "error: data/test.csv missing; pass --allow-holdout for PR dry-runs",
                file=sys.stderr,
            )
            return 1
        print("warning: no test.csv; holding out latest stage from data.csv")
        train_df, test_df = _holdout_latest_stage(data_df)
        X_train, y_train = _xy(train_df)
        X_test, y_test = _xy(test_df)

    X_test = X_test.reindex(columns=X_train.columns)

    report = build_report(estimator, X_train, y_train, X_test, y_test)
    metrics = report.metrics.summarize().frame()
    print(metrics.to_string())

    publish = args.publish or os.environ.get("PUBLISH_TO_HUB") == "1"
    if publish:
        workspace = os.environ.get("SKORE_WORKSPACE", "tour-de-france-challenge")
        project = os.environ.get("SKORE_PROJECT")
        if not project:
            print("error: SKORE_PROJECT is required to publish", file=sys.stderr)
            return 1
        if not os.environ.get("SKORE_API_KEY"):
            print("error: SKORE_API_KEY is required to publish", file=sys.stderr)
            return 1
        publish_report(report, key=key, workspace=workspace, project=project)

    out = {
        "submission": key,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "published": bool(publish),
    }
    summary_path = Path(os.environ.get("SUBMISSION_SUMMARY", "submission_summary.json"))
    summary_path.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
