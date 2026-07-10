#!/usr/bin/env python3
"""Fetch Tour de France data and refresh data/data.csv + data/next_stage.csv.

Sources
-------
- Current season: official rankings on letour.fr (no Cloudflare).
- Previous season (2025): ProCyclingStats pages via the Wayback Machine.

Semantics
---------
- ``data.csv`` is the training set: all labeled stages through the latest
  completed stage (e.g. night of 9 July → includes 9 July).
- ``next_stage.csv`` is the prediction target for the *next* stage (no labels).
- When a new stage completes, CI scores submissions by fitting on the previous
  ``data.csv`` and evaluating on the newly labeled stage, then appends that
  stage into ``data.csv`` and rolls ``next_stage.csv`` forward.
"""

from __future__ import annotations

import argparse
import html as htmlmod
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import pandas as pd
import requests
from selectolax.parser import HTMLParser

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
# Challenge is locked to the 2026 Tour (history = 2025 via Wayback PCS).
CURRENT_YEAR = 2026
HISTORY_YEAR = 2025
STAGE_CATALOG_PATH = DATA_DIR / "stage_catalog.json"
LETOUR = "https://www.letour.fr"
WAYBACK = "https://web.archive.org/web"
PCS = "https://www.procyclingstats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Homogeneous schema across 2025 (PCS) and 2026 (letour). Dropped always-empty /
# source-asymmetric fields: nationality, pcs_points, uci_points, profile_icon.
COLUMNS = [
    "year",
    "stage_number",
    "stage_date",
    "rider_id",
    "rider_name",
    "team",
    "bib",
    "age",
    "stage_type",
    "distance_km",
    "stage_name",
    "prior_stages_ridden",
    "avg_prior_stage_rank",
    "best_prior_stage_rank",
    "last_stage_rank",
    "gc_rank_before",
    "gc_time_gap_before_s",
    "days_since_start",
    "stage_rank",
]
ID_KEY = ["year", "stage_number", "rider_id"]

_STAGE_CATALOG: dict[str, dict[str, dict[str, Any]]] | None = None


def _load_stage_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    global _STAGE_CATALOG
    if _STAGE_CATALOG is not None:
        return _STAGE_CATALOG
    if not STAGE_CATALOG_PATH.exists():
        _STAGE_CATALOG = {}
        return _STAGE_CATALOG
    raw = json.loads(STAGE_CATALOG_PATH.read_text())
    _STAGE_CATALOG = {str(y): {str(k): v for k, v in stages.items()} for y, stages in raw.items()}
    return _STAGE_CATALOG


def catalog_stage(year: int, stage_number: int) -> dict[str, Any] | None:
    """Official stage meta (date / type / distance / name) shared by both seasons."""
    return _load_stage_catalog().get(str(year), {}).get(str(stage_number))


def _race_start_date(year: int) -> str | None:
    stages = _load_stage_catalog().get(str(year), {})
    dates = [s["date"] for s in stages.values() if s.get("date")]
    return min(dates) if dates else None


def _days_since_start(year: int, stage_date: str | None) -> int | None:
    start = _race_start_date(year)
    if not start or not stage_date:
        return None
    try:
        return (
            datetime.strptime(str(stage_date)[:10], "%Y-%m-%d").date()
            - datetime.strptime(start, "%Y-%m-%d").date()
        ).days
    except ValueError:
        return None


def apply_stage_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite stage-level fields from the shared catalog (both years)."""
    if df.empty:
        return df
    catalog = _load_stage_catalog()
    rows: list[dict[str, Any]] = []
    for year, stages in catalog.items():
        for stage_number, meta in stages.items():
            date = meta.get("date")
            rows.append(
                {
                    "year": int(year),
                    "stage_number": int(stage_number),
                    "cat_stage_date": date,
                    "cat_stage_type": meta.get("type", "unknown"),
                    "cat_distance_km": meta.get("distance_km"),
                    "cat_stage_name": meta.get("name"),
                    "cat_days_since_start": _days_since_start(int(year), date),
                }
            )
    if not rows:
        return df

    cat_df = pd.DataFrame(rows)
    out = df.merge(cat_df, on=["year", "stage_number"], how="left")
    matched = out["cat_stage_date"].notna() | out["cat_stage_type"].notna()
    out.loc[matched, "stage_date"] = out.loc[matched, "cat_stage_date"]
    out.loc[matched, "stage_type"] = out.loc[matched, "cat_stage_type"].fillna("unknown")
    dist_ok = matched & out["cat_distance_km"].notna()
    out.loc[dist_ok, "distance_km"] = out.loc[dist_ok, "cat_distance_km"]
    name_ok = matched & out["cat_stage_name"].notna()
    out.loc[name_ok, "stage_name"] = out.loc[name_ok, "cat_stage_name"]
    days_ok = matched & out["cat_days_since_start"].notna()
    out.loc[days_ok, "days_since_start"] = out.loc[days_ok, "cat_days_since_start"]
    return out.drop(
        columns=[
            "cat_stage_date",
            "cat_stage_type",
            "cat_distance_km",
            "cat_stage_name",
            "cat_days_since_start",
        ],
        errors="ignore",
    )


def _apply_catalog_to_row(row: dict[str, Any], year: int, stage_number: int) -> dict[str, Any]:
    meta = catalog_stage(year, stage_number)
    if not meta:
        return row
    row["stage_date"] = meta.get("date") or row.get("stage_date")
    row["stage_type"] = meta.get("type") or row.get("stage_type") or "unknown"
    if meta.get("distance_km") is not None:
        row["distance_km"] = meta["distance_km"]
    if meta.get("name"):
        row["stage_name"] = meta["name"]
    days = _days_since_start(year, row.get("stage_date"))
    if days is not None:
        row["days_since_start"] = days
    return row


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get(session: requests.Session, url: str, retries: int = 3, sleep_s: float = 1.0) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"GET failed for {url}: {last}")


def _parse_time_to_seconds(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text in {"-", "–", "—"}:
        return None
    # letour: 00h 21' 47''  or + 00h 00' 08''
    text = text.replace("''", "").replace("'", " ").replace("h", " ")
    text = text.replace("+", " ").replace(",", "")
    parts = [p for p in re.split(r"[:\s]+", text) if p]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        # PCS style 4:12:03 or +0:04
        text2 = str(value).strip().lstrip("+")
        parts2 = text2.split(":")
        try:
            nums = [int(p) for p in parts2]
        except ValueError:
            return None
    if len(nums) == 3:
        h, m, s = nums
        return float(h * 3600 + m * 60 + s)
    if len(nums) == 2:
        m, s = nums
        return float(m * 60 + s)
    if len(nums) == 1:
        return float(nums[0])
    # Malformed inputs with 0 or 4+ numeric parts are treated as missing.
    return None


def _slugify(name: str) -> str:
    text = name.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def _normalize_stage_type(stage_name: str | None, distance_km: float | None) -> str:
    name = (stage_name or "").lower()
    if "itt" in name or "individual time trial" in name or "contre-la-montre" in name:
        return "itt"
    if "ttt" in name or "team time trial" in name:
        return "ttt"
    if distance_km is not None and distance_km < 50:
        return "itt"
    if any(k in name for k in ("alpe", "tourmalet", "mountain", "summit", "col ")):
        return "mountain"
    return "unknown"


def _add_form_features(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return _empty_frame()
    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "stage_number", "stage_rank"], kind="mergesort")
    enriched: list[dict[str, Any]] = []
    history: dict[tuple[int, str], list[float]] = {}
    for _, row in df.iterrows():
        key = (int(row["year"]), str(row["rider_id"]))
        prior = history.get(key, [])
        rec = row.to_dict()
        rec["prior_stages_ridden"] = len(prior)
        rec["avg_prior_stage_rank"] = float(sum(prior) / len(prior)) if prior else pd.NA
        rec["best_prior_stage_rank"] = float(min(prior)) if prior else pd.NA
        rec["last_stage_rank"] = float(prior[-1]) if prior else pd.NA
        enriched.append(rec)
        if pd.notna(row.get("stage_rank")):
            history.setdefault(key, []).append(float(row["stage_rank"]))
    out = pd.DataFrame(enriched)
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[COLUMNS]


# ---------------------------------------------------------------------------
# letour.fr (current season)
# ---------------------------------------------------------------------------


def _extract_ajax_stacks(page_html: str) -> list[dict[str, str]]:
    stacks = re.findall(r"data-ajax-stack\s*=\s*(\{.*?\})", page_html)
    out: list[dict[str, str]] = []
    for raw in stacks:
        try:
            out.append(json.loads(htmlmod.unescape(raw)))
        except json.JSONDecodeError:
            continue
    return out


def _parse_letour_table(html: str) -> list[dict[str, Any]]:
    tree = HTMLParser(html)
    table = tree.css_first("table")
    if table is None:
        return []
    rows = table.css("tr")
    if len(rows) < 2:
        return []
    headers = [c.text(strip=True).lower() for c in rows[0].css("th,td")]
    results = []
    for tr in rows[1:]:
        cells = [c.text(strip=True) for c in tr.css("td")]
        if not cells or not cells[0].isdigit():
            continue
        data = {headers[i]: cells[i] if i < len(cells) else None for i in range(len(headers))}
        rider_link = None
        team_link = None
        for a in tr.css("a"):
            href = a.attributes.get("href") or ""
            if "/rider/" in href and rider_link is None:
                rider_link = href
            if "/team/" in href and team_link is None:
                team_link = href
        rider_id = None
        if rider_link:
            # /en/rider/11/team-.../jonas-vingegaard-hansen
            parts = [p for p in rider_link.strip("/").split("/") if p]
            if "rider" in parts:
                idx = parts.index("rider")
                if idx + 1 < len(parts):
                    bib_from_url = parts[idx + 1]
                    slug = parts[-1] if len(parts) > idx + 2 else bib_from_url
                    rider_id = slug
        results.append(
            {
                "rank": int(cells[0]),
                "rider_name": data.get("rider"),
                "bib": int(data["rider no."]) if data.get("rider no.") and data["rider no."].isdigit() else None,
                "team": data.get("team"),
                "time": data.get("times") or data.get("time"),
                "gap": data.get("gap"),
                "rider_id": rider_id or _slugify(data.get("rider") or "unknown"),
                "rider_url": rider_link,
                "team_url": team_link,
            }
        )
    return results


def _parse_stage_nav(page_html: str, year: int) -> list[dict[str, Any]]:
    """Parse stage list + dates from rankings page links."""
    tree = HTMLParser(page_html)
    stages: dict[int, dict[str, Any]] = {}
    for a in tree.css("a"):
        href = a.attributes.get("href") or ""
        text = a.text(strip=True)
        m = re.search(r"/rankings/stage-(\d+)", href)
        if not m:
            continue
        num = int(m.group(1))
        # Stage 6- 07/09 -Pau > Gavarnie-Gèdre
        dm = re.search(r"(\d{2})/(\d{2})", text)
        stage_date = None
        if dm:
            mm, dd = dm.group(1), dm.group(2)
            stage_date = f"{year}-{mm}-{dd}"
        name_m = re.search(r"-\s*(.+)$", text)
        stage_name = name_m.group(1).strip() if name_m else f"Stage {num}"
        stages[num] = {
            "year": year,
            "stage_number": num,
            "stage_date": stage_date,
            "stage_name": stage_name,
            "stage_url": href if href.startswith("http") else f"{LETOUR}{href}",
        }
    return [stages[k] for k in sorted(stages)]


def fetch_letour_year(session: requests.Session, year: int) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return labeled rows for completed stages + stage meta (including upcoming)."""
    home = _get(session, f"{LETOUR}/en/rankings/stage-1")
    parsed = {s["stage_number"]: s for s in _parse_stage_nav(home, year)}
    # Always cover 1..21; nav on a stage page often omits the current stage link.
    stage_meta: list[dict[str, Any]] = []
    for n in range(1, 22):
        if n in parsed:
            stage_meta.append(parsed[n])
        else:
            stage_meta.append(
                {
                    "year": year,
                    "stage_number": n,
                    "stage_date": None,
                    "stage_name": f"Stage {n}",
                    "stage_url": f"{LETOUR}/en/rankings/stage-{n}",
                }
            )
    # Fill stage-1 date from page title / nav of another stage if missing
    if stage_meta[0].get("stage_date") is None:
        # Stage 1- 07/04 appears on later pages
        later = _get(session, f"{LETOUR}/en/rankings/stage-2")
        for s in _parse_stage_nav(later, year):
            if s["stage_number"] in parsed or s["stage_number"] == 1:
                for meta in stage_meta:
                    if meta["stage_number"] == s["stage_number"]:
                        meta.update({k: v for k, v in s.items() if v})

    labeled_rows: list[dict[str, Any]] = []
    completed_meta: list[dict[str, Any]] = []

    for meta in stage_meta:
        n = meta["stage_number"]
        url = f"{LETOUR}/en/rankings/stage-{n}"
        try:
            page = _get(session, url)
        except RuntimeError as exc:
            print(f"  warn: stage {n} page failed: {exc}")
            meta["has_results"] = False
            continue
        stage_results = _parse_letour_table(page)
        stacks = _extract_ajax_stacks(page)
        gc_results: list[dict[str, Any]] = []
        if stacks:
            # first stack is usually general classifications; itg = individual general
            itg = stacks[0].get("itg")
            if itg and str(itg).startswith("/"):
                try:
                    gc_html = _get(session, f"{LETOUR}{itg}")
                    gc_results = _parse_letour_table(gc_html)
                except RuntimeError:
                    gc_results = []

        meta["has_results"] = bool(stage_results)
        meta["n_results"] = len(stage_results)
        if not stage_results:
            print(f"  stage {n}: no results yet")
            time.sleep(0.2)
            continue

        # distance from stage page if possible
        distance_km = None
        try:
            stage_page = _get(session, f"{LETOUR}/en/stage-{n}")
            dm = re.search(r"(\d+(?:[.,]\d+)?)\s*km", stage_page, flags=re.I)
            if dm:
                distance_km = float(dm.group(1).replace(",", "."))
        except RuntimeError:
            pass

        stage_type = _normalize_stage_type(meta.get("stage_name"), distance_km)
        cat = catalog_stage(year, n)
        if cat:
            stage_type = cat.get("type") or stage_type
            if cat.get("distance_km") is not None:
                distance_km = cat["distance_km"]
            if cat.get("date"):
                meta["stage_date"] = cat["date"]
            if cat.get("name"):
                meta["stage_name"] = cat["name"]
        gc_by_rider = {r["rider_id"]: r for r in gc_results}
        # GC *before* this stage ≈ GC after previous stage; approximate with
        # current GC rank shifted via previous labeled rows later. Here store
        # post-stage GC as gc_rank_after proxy in a temp field, then shift.
        print(f"  stage {n}: {len(stage_results)} riders (letour)")

        days_since_start = _days_since_start(year, meta.get("stage_date"))

        for res in stage_results:
            gc = gc_by_rider.get(res["rider_id"], {})
            labeled_rows.append(
                {
                    "year": year,
                    "stage_number": n,
                    "stage_date": meta.get("stage_date"),
                    "rider_id": res["rider_id"],
                    "rider_name": res.get("rider_name"),
                    "team": res.get("team"),
                    "bib": res.get("bib"),
                    "age": pd.NA,
                    "stage_type": stage_type,
                    "distance_km": distance_km,
                    "stage_name": meta.get("stage_name"),
                    "gc_rank_after": gc.get("rank"),
                    "gc_time_after_s": _parse_time_to_seconds(gc.get("time")),
                    "days_since_start": days_since_start,
                    "stage_rank": float(res["rank"]),
                }
            )
        completed_meta.append(meta)
        time.sleep(0.35)

    # Snapshot GC *after* the latest completed stage for next_stage.csv
    gc_after_latest: dict[str, dict[str, Any]] = {}
    if labeled_rows:
        max_stage = max(int(r["stage_number"]) for r in labeled_rows)
        for row in labeled_rows:
            if int(row["stage_number"]) != max_stage:
                continue
            gc_after_latest[row["rider_id"]] = {
                "gc_rank_before": row.get("gc_rank_after"),
                "gc_time_gap_before_s": None,
            }
        times = [
            r["gc_time_after_s"]
            for r in labeled_rows
            if int(r["stage_number"]) == max_stage
            and r.get("gc_time_after_s") is not None
            and not pd.isna(r.get("gc_time_after_s"))
        ]
        if times:
            lead = min(times)
            for row in labeled_rows:
                if int(row["stage_number"]) != max_stage:
                    continue
                t = row.get("gc_time_after_s")
                if t is not None and not pd.isna(t):
                    gc_after_latest[row["rider_id"]]["gc_time_gap_before_s"] = float(t) - float(
                        lead
                    )

    labeled_rows = _shift_gc_before(labeled_rows)
    return _add_form_features(labeled_rows), stage_meta, gc_after_latest


def _shift_gc_before(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Snapshot GC-after per stage *before* mutating rows (rows are shared refs).
    gc_after: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        stage = int(row["stage_number"])
        gc_after.setdefault(stage, {})[row["rider_id"]] = {
            "gc_rank_after": row.get("gc_rank_after"),
            "gc_time_after_s": row.get("gc_time_after_s"),
        }

    leader_time: dict[int, float] = {}
    for stage, riders in gc_after.items():
        times = [
            r["gc_time_after_s"]
            for r in riders.values()
            if r.get("gc_time_after_s") is not None and not pd.isna(r.get("gc_time_after_s"))
        ]
        if times:
            leader_time[stage] = min(times)

    out = []
    for row in rows:
        stage = int(row["stage_number"])
        prev = gc_after.get(stage - 1, {}).get(row["rider_id"])
        if prev is None:
            row["gc_rank_before"] = pd.NA
            row["gc_time_gap_before_s"] = pd.NA
        else:
            row["gc_rank_before"] = prev.get("gc_rank_after")
            prev_t = prev.get("gc_time_after_s")
            lead = leader_time.get(stage - 1)
            if prev_t is not None and lead is not None and not pd.isna(prev_t):
                row["gc_time_gap_before_s"] = float(prev_t) - float(lead)
            else:
                row["gc_time_gap_before_s"] = pd.NA
        row.pop("gc_rank_after", None)
        row.pop("gc_time_after_s", None)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Wayback + PCS (previous season)
# ---------------------------------------------------------------------------


def _wayback_url(pcs_path: str, timestamp: str = "20250715000000") -> str:
    return f"{WAYBACK}/{timestamp}/{PCS}/{pcs_path.lstrip('/')}"


def _parse_pcs_wayback_stage(html: str, year: int, stage_number: int) -> list[dict[str, Any]]:
    tree = HTMLParser(html)
    # distance / date
    distance_km = None
    dm = re.search(r"\((\d+(?:\.\d+)?)\s*km\)", html)
    if dm:
        distance_km = float(dm.group(1))
    stage_date = None
    # try common PCS date formats in page
    for pat in [
        rf"({year}-\d{{2}}-\d{{2}})",
        r"(\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})",
    ]:
        m = re.search(pat, html)
        if m:
            raw = m.group(1)
            try:
                if "-" in raw:
                    stage_date = raw
                else:
                    stage_date = datetime.strptime(raw, "%d %b %Y").strftime("%Y-%m-%d")
            except ValueError:
                pass
            break

    stage_name = f"Stage {stage_number}"
    title = tree.css_first("title")
    if title:
        stage_name = title.text(strip=True)

    results = []
    for tr in tree.css("table tr"):
        tds = tr.css("td")
        if len(tds) < 8:
            continue
        rank_txt = tds[0].text(strip=True)
        if not rank_txt.isdigit():
            continue
        links = tr.css("a")
        rider_url = None
        team_url = None
        rider_name = None
        team_name = None
        for a in links:
            href = unquote(a.attributes.get("href") or "")
            # strip wayback prefix
            href = re.sub(r"^https?://web\.archive\.org/web/\d+/", "", href)
            href = href.replace(f"{PCS}/", "").lstrip("/")
            text = a.text(strip=True)
            if href.startswith("rider/") and rider_url is None:
                rider_url = href
                rider_name = text
            elif href.startswith("team/") and team_url is None:
                team_url = href
                team_name = text
        if rider_url is None:
            continue
        rider_id = rider_url.split("/")[-1]
        bib = None
        age = None
        # PCS columns often: Rnk GC Timelag BIB H2H Specialty Age Rider Team UCI
        try:
            bib = int(tds[3].text(strip=True))
        except ValueError:
            pass
        try:
            age = int(tds[6].text(strip=True))
        except ValueError:
            pass
        gc_rank = None
        try:
            gc_rank = int(tds[1].text(strip=True))
        except ValueError:
            pass
        results.append(
            {
                "year": year,
                "stage_number": stage_number,
                "stage_date": stage_date,
                "rider_id": rider_id,
                "rider_name": rider_name,
                "team": team_name,
                "bib": bib,
                "age": age,
                "stage_type": _normalize_stage_type(stage_name, distance_km),
                "distance_km": distance_km,
                "stage_name": stage_name,
                "gc_rank_after": gc_rank,
                "gc_time_after_s": pd.NA,
                "days_since_start": pd.NA,
                "stage_rank": float(rank_txt),
            }
        )
    # Prefer shared catalog for stage-level fields (homogeneous with 2026).
    return [_apply_catalog_to_row(r, year, stage_number) for r in results]


def fetch_pcs_wayback_year(
    session: requests.Session,
    year: int = 2025,
    max_stage: int = 21,
) -> pd.DataFrame:
    """Fetch a full previous Tour via Wayback snapshots of PCS stage pages."""
    # Timestamps roughly after each stage; a mid/late-tour snapshot often still
    # serves historical stage pages.
    timestamps = [
        "20250801000000",
        "20250728000000",
        "20250725000000",
        "20250722000000",
        "20250720000000",
        "20250718000000",
        "20250715000000",
        "20250713000000",
        "20250710000000",
        "20250708000000",
        "20250706000000",
    ]
    all_rows: list[dict[str, Any]] = []
    for n in range(1, max_stage + 1):
        path = f"race/tour-de-france/{year}/stage-{n}"
        got = False
        for ts in timestamps:
            url = _wayback_url(path, ts)
            try:
                html = _get(session, url, retries=2, sleep_s=1.5)
            except RuntimeError:
                continue
            if "Just a moment" in html or "Page not found" in html:
                continue
            rows = _parse_pcs_wayback_stage(html, year=year, stage_number=n)
            # Deduplicate: PCS pages embed multiple ranking tables; keep first
            # occurrence per rider (stage result table comes first).
            seen = set()
            unique = []
            for row in rows:
                if row["rider_id"] in seen:
                    continue
                seen.add(row["rider_id"])
                unique.append(row)
            if len(unique) >= 50:
                print(f"  stage {n}: {len(unique)} riders (wayback PCS @ {ts})")
                all_rows.extend(unique)
                got = True
                break
        if not got:
            print(f"  stage {n}: unavailable via Wayback")
        time.sleep(0.5)

    if not all_rows:
        return _empty_frame()

    # Fill days_since_start from min stage_date
    dates = [r["stage_date"] for r in all_rows if r.get("stage_date")]
    if dates:
        start = min(dates)
        for r in all_rows:
            if r.get("stage_date"):
                try:
                    r["days_since_start"] = (
                        datetime.strptime(r["stage_date"], "%Y-%m-%d").date()
                        - datetime.strptime(start, "%Y-%m-%d").date()
                    ).days
                except ValueError:
                    pass

    all_rows = _shift_gc_before(all_rows)
    return _add_form_features(all_rows)


# ---------------------------------------------------------------------------
# Splits / next stage
# ---------------------------------------------------------------------------


def build_next_stage(
    data: pd.DataFrame,
    stage_meta: list[dict[str, Any]],
    year: int,
    gc_after_latest: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if data.empty:
        return _empty_frame()
    current = data[data["year"] == year]
    if current.empty:
        current = data
        year = int(data["year"].max())

    last_stage = int(current["stage_number"].max())
    upcoming = [s for s in stage_meta if s["stage_number"] > last_stage]
    if not upcoming:
        next_num = last_stage + 1
        next_meta = {
            "year": year,
            "stage_number": next_num,
            "stage_date": None,
            "stage_name": f"Stage {next_num}",
            "stage_type": "unknown",
            "distance_km": pd.NA,
        }
    else:
        nxt = upcoming[0]
        next_meta = {
            "year": year,
            "stage_number": nxt["stage_number"],
            "stage_date": nxt.get("stage_date"),
            "stage_name": nxt.get("stage_name"),
            "stage_type": _normalize_stage_type(nxt.get("stage_name"), None),
            "distance_km": pd.NA,
        }

    cat = catalog_stage(year, int(next_meta["stage_number"]))
    if cat:
        next_meta["stage_date"] = cat.get("date") or next_meta.get("stage_date")
        next_meta["stage_type"] = cat.get("type") or next_meta.get("stage_type")
        next_meta["distance_km"] = cat.get("distance_km", next_meta.get("distance_km"))
        next_meta["stage_name"] = cat.get("name") or next_meta.get("stage_name")

    days = _days_since_start(year, next_meta.get("stage_date"))

    last_rows = current[current["stage_number"] == last_stage]
    rows = []
    for _, last in last_rows.iterrows():
        rider_hist = current[current["rider_id"] == last["rider_id"]]
        prior = rider_hist["stage_rank"].dropna().astype(float).tolist()
        rows.append(
            {
                "year": next_meta["year"],
                "stage_number": next_meta["stage_number"],
                "stage_date": next_meta["stage_date"],
                "rider_id": last["rider_id"],
                "rider_name": last["rider_name"],
                "team": last["team"],
                "bib": last.get("bib"),
                "age": last.get("age"),
                "stage_type": next_meta["stage_type"],
                "distance_km": next_meta["distance_km"],
                "stage_name": next_meta["stage_name"],
                "prior_stages_ridden": len(prior),
                "avg_prior_stage_rank": float(sum(prior) / len(prior)) if prior else pd.NA,
                "best_prior_stage_rank": float(min(prior)) if prior else pd.NA,
                "last_stage_rank": float(prior[-1]) if prior else pd.NA,
                "gc_rank_before": (gc_after_latest or {}).get(last["rider_id"], {}).get(
                    "gc_rank_before", last.get("gc_rank_before")
                ),
                "gc_time_gap_before_s": (gc_after_latest or {})
                .get(last["rider_id"], {})
                .get("gc_time_gap_before_s", last.get("gc_time_gap_before_s")),
                "days_since_start": days if days is not None else pd.NA,
                "stage_rank": pd.NA,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS) if rows else _empty_frame()


def hub_project_name(stage_date: str | None, stage_number: int | None = None) -> str:
    """Hub project slug: ``{day}_juillet`` (e.g. ``8_juillet`` for 8 July)."""
    if stage_date:
        try:
            day = datetime.strptime(str(stage_date)[:10], "%Y-%m-%d").day
            return f"{day}_juillet"
        except ValueError:
            pass
    if stage_number is not None:
        return f"stage-{stage_number}_juillet"
    return "unknown_juillet"


def write_outputs(
    data: pd.DataFrame,
    next_stage: pd.DataFrame,
    score_stage: pd.DataFrame | None = None,
    *,
    has_new_stage: bool = False,
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = apply_stage_catalog(data)
    data = data.sort_values(["year", "stage_number", "stage_rank"], kind="mergesort").reset_index(
        drop=True
    )
    # Ensure homogeneous column set (drop legacy empty fields if present).
    for col in COLUMNS:
        if col not in data.columns:
            data[col] = pd.NA
    data = data[COLUMNS]
    data.to_csv(DATA_DIR / "data.csv", index=False)

    next_stage = apply_stage_catalog(next_stage)
    for col in COLUMNS:
        if col not in next_stage.columns:
            next_stage[col] = pd.NA
    next_stage = next_stage[COLUMNS] if not next_stage.empty else _empty_frame()
    next_stage.to_csv(DATA_DIR / "next_stage.csv", index=False)

    # Remove legacy train.csv if present
    legacy_train = DATA_DIR / "train.csv"
    if legacy_train.exists():
        legacy_train.unlink()

    info: dict[str, Any] = {
        "n_data": len(data),
        "n_next": len(next_stage),
        "latest_stage_number": None,
        "latest_stage_date": None,
        "latest_year": None,
        "next_stage_number": int(next_stage["stage_number"].iloc[0]) if not next_stage.empty else None,
        "next_stage_date": None,
        "score_stage_number": None,
        "score_stage_date": None,
        "n_score": 0,
        "has_new_stage": bool(has_new_stage),
        "commit_day": None,
    }
    if not data.empty:
        # Prefer the most recent season present in data.csv
        latest_year = int(data["year"].max())
        year_rows = data[data["year"] == latest_year]
        latest = int(year_rows["stage_number"].max())
        latest_rows = year_rows[year_rows["stage_number"] == latest]
        dates = latest_rows["stage_date"].dropna()
        info["latest_year"] = latest_year
        info["latest_stage_number"] = latest
        info["latest_stage_date"] = str(dates.iloc[0]) if not dates.empty else None

    if not next_stage.empty:
        nd = next_stage["stage_date"].dropna()
        info["next_stage_date"] = str(nd.iloc[0]) if not nd.empty else None

    if score_stage is not None and not score_stage.empty:
        score_stage = apply_stage_catalog(score_stage)
        for col in COLUMNS:
            if col not in score_stage.columns:
                score_stage[col] = pd.NA
        score_stage = score_stage[COLUMNS]
        score_stage.to_csv(DATA_DIR / "test.csv", index=False)
        info["n_score"] = len(score_stage)
        info["score_stage_number"] = int(score_stage["stage_number"].iloc[0])
        sd = score_stage["stage_date"].dropna()
        info["score_stage_date"] = str(sd.iloc[0]) if not sd.empty else None
        info["skore_project"] = hub_project_name(
            info["score_stage_date"], info["score_stage_number"]
        )
    else:
        # No dedicated score file: PR dry-runs hold out the latest data stage.
        test_path = DATA_DIR / "test.csv"
        if test_path.exists():
            test_path.unlink()
        info["skore_project"] = hub_project_name(
            info.get("next_stage_date"), info.get("next_stage_number")
        )

    # Calendar day for the CI commit message ("New data for the {day} july stage")
    day_src = info.get("score_stage_date") or info.get("latest_stage_date")
    if day_src:
        try:
            info["commit_day"] = datetime.strptime(str(day_src)[:10], "%Y-%m-%d").day
        except ValueError:
            info["commit_day"] = None

    (DATA_DIR / "latest_score_meta.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def load_existing_data() -> pd.DataFrame:
    path = DATA_DIR / "data.csv"
    if not path.exists() or path.stat().st_size == 0:
        return _empty_frame()
    df = pd.read_csv(path)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS]


def enrich_existing_csvs() -> int:
    """Re-apply stage catalog to committed CSVs without a live fetch."""
    data = load_existing_data()
    if data.empty:
        print("No data.csv to enrich")
        return 1
    next_path = DATA_DIR / "next_stage.csv"
    next_stage = pd.read_csv(next_path) if next_path.exists() else _empty_frame()
    test_path = DATA_DIR / "test.csv"
    score_stage = None
    if test_path.exists() and test_path.stat().st_size > 0:
        score_stage = pd.read_csv(test_path)
    # Preserve prior meta flags when only enriching.
    prev_meta: dict[str, Any] = {}
    meta_path = DATA_DIR / "latest_score_meta.json"
    if meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            prev_meta = {}
    info = write_outputs(
        data,
        next_stage,
        score_stage=score_stage,
        has_new_stage=bool(prev_meta.get("has_new_stage")),
    )
    if prev_meta.get("n_score") and score_stage is not None:
        info["n_score"] = int(prev_meta["n_score"])
        info["score_stage_number"] = prev_meta.get("score_stage_number")
        info["score_stage_date"] = prev_meta.get("score_stage_date")
        info["skore_project"] = prev_meta.get("skore_project") or info.get("skore_project")
        meta_path.write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"Enriched CSVs via stage_catalog.json "
        f"(data={info['n_data']}, next={info['n_next']}, score={info.get('n_score', 0)})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only fetch the 2026 season from letour.fr (skip 2025 Wayback backfill)",
    )
    parser.add_argument(
        "--enrich-only",
        action="store_true",
        help="Re-apply data/stage_catalog.json to existing CSVs (no network fetch)",
    )
    args = parser.parse_args()
    if args.enrich_only:
        return enrich_existing_csvs()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()

    print(f"Fetching current season {CURRENT_YEAR} from letour.fr…")
    current_df, stage_meta, gc_after_latest = fetch_letour_year(session, CURRENT_YEAR)
    print(f"  labeled rows: {len(current_df)}")

    history_df = _empty_frame()
    if not args.skip_history:
        print(f"Fetching history year {HISTORY_YEAR} via Wayback PCS…")
        try:
            history_df = fetch_pcs_wayback_year(session, year=HISTORY_YEAR)
            print(f"  history rows: {len(history_df)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: history fetch failed: {exc}", file=sys.stderr)

    # Merge: keep non-overlapping years from existing file, replace fetched years
    existing = load_existing_data()
    fetched_years: set[int] = set()
    if not current_df.empty:
        if not existing.empty:
            existing_cur = existing[existing["year"] == CURRENT_YEAR]
            if not existing_cur.empty and len(current_df) < len(existing_cur) * 0.8:
                print(
                    "error: fetched fewer CURRENT_YEAR rows than existing "
                    f"({len(current_df)} < 80% of {len(existing_cur)}); aborting overwrite",
                    file=sys.stderr,
                )
                return 1
        fetched_years.add(CURRENT_YEAR)
    elif not existing.empty and (existing["year"] == CURRENT_YEAR).any():
        print(
            "warn: current-year fetch returned no rows; keeping existing CURRENT_YEAR data",
            file=sys.stderr,
        )
    if not history_df.empty:
        fetched_years.add(HISTORY_YEAR)
    if not existing.empty and fetched_years:
        keep = existing[~existing["year"].isin(fetched_years)]
    else:
        keep = existing if not existing.empty else _empty_frame()

    parts = [p for p in (keep, history_df, current_df) if not p.empty]
    data = pd.concat(parts, ignore_index=True) if parts else _empty_frame()
    data = data.drop_duplicates(subset=ID_KEY, keep="last")

    # New stage = newly completed stage vs previous data.csv (rest days → no advance)
    prev = existing
    score_stage = None
    has_new_stage = False
    if not current_df.empty and not prev.empty:
        prev_cur = prev[prev["year"] == CURRENT_YEAR]
        new_cur = current_df
        if not prev_cur.empty:
            prev_max = int(prev_cur["stage_number"].max())
            new_max = int(new_cur["stage_number"].max())
            if new_max > prev_max:
                has_new_stage = True
                score_stage = new_cur[new_cur["stage_number"] == new_max].copy()
                print(f"  new stage available for scoring: {new_max}")
            else:
                print(
                    f"  no new stage (latest still {new_max}) — "
                    "rest day or results not posted yet"
                )
        else:
            # First time we see 2026 rows in an existing file
            has_new_stage = True
    elif not current_df.empty and prev.empty:
        # Bootstrap empty repo: write data, but nothing to score yet
        has_new_stage = True
        print("  bootstrap: writing initial data.csv (no score stage)")

    if not has_new_stage and not existing.empty:
        # Leave committed CSVs untouched so CI does not push or re-score.
        info_path = DATA_DIR / "latest_score_meta.json"
        info: dict[str, Any] = {}
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
            except json.JSONDecodeError:
                info = {}
        info["has_new_stage"] = False
        info["n_score"] = 0
        info.pop("score_stage_number", None)
        info.pop("score_stage_date", None)
        # Keep skore_project from last real stage for reference; scoring is gated off.
        info_path.write_text(json.dumps(info, indent=2) + "\n")
        print("No new stage data tonight; leaving data/*.csv unchanged.")
        return 0

    next_stage = build_next_stage(
        data, stage_meta, CURRENT_YEAR, gc_after_latest=gc_after_latest
    )
    info = write_outputs(
        data, next_stage, score_stage=score_stage, has_new_stage=has_new_stage
    )
    print(
        f"Wrote data.csv ({info['n_data']} rows through stage "
        f"{info['latest_stage_number']} / {info['latest_stage_date']}), "
        f"next_stage={info['n_next']} (stage {info['next_stage_number']})"
    )
    if info["n_score"]:
        print(
            f"Wrote test.csv for scoring stage {info['score_stage_number']} "
            f"({info['score_stage_date']}), project={info['skore_project']}"
        )
    return 0 if not data.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
