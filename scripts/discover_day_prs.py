#!/usr/bin/env python3
"""Discover open submission PRs for nightly scoring.

Every open PR that touches ``submissions/<login>/`` (non-skeleton) is eligible.
Models are retrained each CI run, so PRs stay open and are re-scored whenever
there is a new stage to evaluate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Folder / hub keys must be safe for shell paths and artifact names.
_SAFE_NAME = re.compile(r"^[a-zA-Z0-9_-]+$")


def is_safe_submission_name(name: str) -> bool:
    return bool(name) and bool(_SAFE_NAME.match(name))


def _gh_json(args: list[str]) -> object | None:
    cmd = ["gh", *args]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        print(f"error: gh {' '.join(args)} failed: {err}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("error: gh CLI not found on PATH", file=sys.stderr)
        return None
    return json.loads(result.stdout) if result.stdout.strip() else None


def _submission_dirs_from_files(files: list[str]) -> list[str]:
    found: set[str] = set()
    for path in files:
        parts = Path(path).parts
        if len(parts) >= 2 and parts[0] == "submissions" and not parts[1].startswith("_"):
            name = parts[1]
            if is_safe_submission_name(name):
                found.add(name)
            else:
                print(f"skip unsafe submission folder name: {name!r}", file=sys.stderr)
    return sorted(found)


def discover() -> list[dict]:
    """Return matrix rows for all open PRs with a submission folder."""
    prs = _gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,author,headRefOid,headRepository,url",
        ]
    )
    if not isinstance(prs, list):
        return []

    rows: list[dict] = []
    for pr in prs:
        number = pr["number"]
        raw = subprocess.run(
            ["gh", "pr", "diff", str(number), "--name-only"],
            check=False,
            capture_output=True,
            text=True,
        )
        file_list = [line.strip() for line in raw.stdout.splitlines() if line.strip()]

        subs = _submission_dirs_from_files(file_list)
        if not subs:
            print(
                f"skip PR #{number}: no submissions/<login>/ changes",
                file=sys.stderr,
            )
            continue

        author = (pr.get("author") or {}).get("login") or "unknown"
        if author in subs:
            submission = author
        else:
            submission = subs[0]
        if not is_safe_submission_name(submission):
            print(
                f"skip PR #{number}: unsafe submission name {submission!r}",
                file=sys.stderr,
            )
            continue

        head_repo = None
        head = pr.get("headRepository") or {}
        if isinstance(head, dict) and head.get("name"):
            owner = head.get("owner")
            if isinstance(owner, dict):
                head_repo = f"{owner.get('login')}/{head.get('name')}"
            elif head.get("nameWithOwner"):
                head_repo = head["nameWithOwner"]

        if not head_repo:
            detail = _gh_json(
                ["pr", "view", str(number), "--json", "headRepositoryOwner,headRepository"]
            )
            if isinstance(detail, dict):
                owner = (detail.get("headRepositoryOwner") or {}).get("login")
                repo = (detail.get("headRepository") or {}).get("name")
                if owner and repo:
                    head_repo = f"{owner}/{repo}"

        rows.append(
            {
                "pr_number": number,
                "submission": submission,
                "hub_key": submission,
                "author": author,
                "title": pr.get("title") or "",
                "head_sha": pr.get("headRefOid"),
                "head_repo": head_repo or "",
                "url": pr.get("url") or "",
            }
        )
        print(
            f"eligible PR #{number} ({author}) submission={submission}",
            file=sys.stderr,
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=... to $GITHUB_OUTPUT",
    )
    # Kept for backward compatibility with older workflow inputs; ignored.
    parser.add_argument("--score-day", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--meta", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    print("discovering all open submission PRs", file=sys.stderr)
    rows = discover()
    matrix = {"include": rows} if rows else {"include": []}
    text = json.dumps(matrix, separators=(",", ":"))
    print(text)
    Path("matrix.json").write_text(text + "\n")

    if args.github_output:
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"matrix={text}\n")
                fh.write(f"has_prs={'true' if rows else 'false'}\n")
                fh.write(f"n_prs={len(rows)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
