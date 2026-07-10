"""Unit tests for challenge harness helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def discover_mod():
    return _load("discover_day_prs", SCRIPTS / "discover_day_prs.py")


@pytest.fixture(scope="module")
def skore_mod():
    return _load("check_skore_usage", SCRIPTS / "check_skore_usage.py")


@pytest.fixture(scope="module")
def fetch_mod():
    return _load("fetch_tdf_data", SCRIPTS / "fetch_tdf_data.py")


def test_safe_submission_names(discover_mod):
    assert discover_mod.is_safe_submission_name("louiscrc")
    assert discover_mod.is_safe_submission_name("user_1-test")
    assert not discover_mod.is_safe_submission_name("foo$(whoami)")
    assert not discover_mod.is_safe_submission_name("../evil")
    assert not discover_mod.is_safe_submission_name("has space")
    assert not discover_mod.is_safe_submission_name("")


def test_submission_dirs_skip_unsafe(discover_mod):
    files = [
        "submissions/ok_user/submission.py",
        "submissions/bad$(x)/submission.py",
        "README.md",
    ]
    assert discover_mod._submission_dirs_from_files(files) == ["ok_user"]


def test_check_skore_detects_import(skore_mod, tmp_path: Path):
    sub = tmp_path / "submissions" / "demo"
    sub.mkdir(parents=True)
    (sub / "submission.py").write_text("import skore\n\ndef build_estimator():\n    pass\n")
    assert skore_mod.check_submission(sub) == 0


def test_check_skore_rejects_missing(skore_mod, tmp_path: Path):
    sub = tmp_path / "submissions" / "demo"
    sub.mkdir(parents=True)
    (sub / "submission.py").write_text("def build_estimator():\n    return None\n")
    assert skore_mod.check_submission(sub) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("00h 21' 47''", 21 * 60 + 47.0),
        ("+ 00h 00' 08''", 8.0),
        ("4:12:03", 4 * 3600 + 12 * 60 + 3.0),
        ("+0:04", 4.0),
        ("01 02 03 04", None),
        ("-", None),
        (None, None),
    ],
)
def test_parse_time_to_seconds(fetch_mod, raw, expected):
    assert fetch_mod._parse_time_to_seconds(raw) == expected


def test_apply_stage_catalog_merge(fetch_mod, monkeypatch):
    catalog = {
        "2026": {
            "1": {
                "date": "2026-07-04",
                "type": "flat",
                "distance_km": 100.0,
                "name": "Stage 1",
            }
        }
    }
    monkeypatch.setattr(fetch_mod, "_STAGE_CATALOG", catalog)
    df = pd.DataFrame(
        [
            {
                "year": 2026,
                "stage_number": 1,
                "stage_date": None,
                "stage_type": "unknown",
                "distance_km": None,
                "stage_name": "old",
                "days_since_start": None,
                "rider_id": "a",
            }
        ]
    )
    out = fetch_mod.apply_stage_catalog(df)
    assert out.loc[0, "stage_date"] == "2026-07-04"
    assert out.loc[0, "stage_type"] == "flat"
    assert out.loc[0, "distance_km"] == 100.0
    assert out.loc[0, "stage_name"] == "Stage 1"
    assert out.loc[0, "days_since_start"] == 0
