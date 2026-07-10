#!/usr/bin/env python3
"""Discover open PRs eligible for tonight's scoring.

A PR is eligible when it is open and was created or last updated on the scoring
stage's calendar day (Europe/Paris). Submissions are not merged to main; the
nightly job checks out each PR head and evaluates it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")


def _gh_json(args: list[str]) -> object:
    cmd = ["gh", *args]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def _paris_day(iso_ts: str) -> str:
    # gh returns timestamps like 2026-07-08T15:26:26Z
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).astimezone(PARIS)
    return dt.date().isoformat()


def _submission_dirs_from_files(files: list[str]) -> list[str]:
    found: set[str] = set()
    for path in files:
        parts = Path(path).parts
        if len(parts) >= 2 and parts[0] == "submissions" and not parts[1].startswith("_"):
            found.add(parts[1])
    return sorted(found)


def discover(score_day: str) -> list[dict]:
    """Return matrix rows for open PRs touched on ``score_day`` (YYYY-MM-DD Paris)."""
    prs = _gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,author,createdAt,updatedAt,headRefOid,headRepository,url",
        ]
    )
    if not isinstance(prs, list):
        return []

    rows: list[dict] = []
    for pr in prs:
        created_day = _paris_day(pr["createdAt"])
        updated_day = _paris_day(pr["updatedAt"])
        if score_day not in {created_day, updated_day}:
            continue

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
        # Prefer folder matching author; else first submission dir in the PR
        submission = author if author in subs else subs[0]
        head_repo = None
        head = pr.get("headRepository") or {}
        if isinstance(head, dict) and head.get("name"):
            # owner/name — gh json may only have name; fall back to pr view
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
                "created_day": created_day,
                "updated_day": updated_day,
            }
        )
        print(
            f"eligible PR #{number} ({author}) submission={submission} "
            f"created={created_day} updated={updated_day}",
            file=sys.stderr,
        )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-day",
        default=None,
        help="Calendar day in Europe/Paris (YYYY-MM-DD). Default: score_stage_date from meta.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("data/latest_score_meta.json"),
        help="Path to latest_score_meta.json",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write matrix=... to $GITHUB_OUTPUT",
    )
    args = parser.parse_args()

    score_day = args.score_day
    if not score_day and args.meta.exists():
        info = json.loads(args.meta.read_text())
        score_day = info.get("score_stage_date")
    if not score_day:
        # Fall back to today in Paris
        score_day = datetime.now(tz=PARIS).date().isoformat()

    # Normalize date-only
    score_day = str(score_day)[:10]
    print(f"discovering open PRs for score day {score_day} (Europe/Paris)", file=sys.stderr)

    rows = discover(score_day)
    matrix = {"include": rows} if rows else {"include": []}
    # Compact JSON so GHA if-conditions can match '{"include":[]}' reliably
    text = json.dumps(matrix, separators=(",", ":"))
    print(text)
    Path("matrix.json").write_text(text + "\n")

    if args.github_output:
        out = os.environ.get("GITHUB_OUTPUT")
        if out:
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(f"matrix={text}\n")
                fh.write(f"has_prs={'true' if rows else 'false'}\n")
                fh.write(f"score_day={score_day}\n")
                fh.write(f"n_prs={len(rows)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
