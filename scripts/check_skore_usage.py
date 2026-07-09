#!/usr/bin/env python3
"""Fail unless a submission directory imports skore."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _imports_skore(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        print(f"error: cannot parse {path}: {exc}", file=sys.stderr)
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "skore" or alias.name.startswith("skore."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "skore" or node.module.startswith("skore.")):
                return True
    return False


def check_submission(submission_dir: Path) -> int:
    if not submission_dir.is_dir():
        print(f"error: not a directory: {submission_dir}", file=sys.stderr)
        return 2

    py_files = sorted(submission_dir.rglob("*.py"))
    if not py_files:
        print(f"error: no Python files under {submission_dir}", file=sys.stderr)
        return 1

    for path in py_files:
        if _imports_skore(path):
            print(f"ok: skore import found in {path.relative_to(submission_dir)}")
            return 0

    print(
        f"error: submission {submission_dir} must import skore "
        "(e.g. `import skore` or `from skore import evaluate`).",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission",
        type=Path,
        help="Path to submissions/<login>/ directory",
    )
    args = parser.parse_args()
    return check_submission(args.submission.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
