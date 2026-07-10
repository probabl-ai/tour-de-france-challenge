# Dataset

Labeled history for the Tour de France stage-rank challenge.

## Files

| File | Role |
| --- | --- |
| `data.csv` | **Training set** — all labeled rider–stage rows through the latest completed stage (e.g. night of 9 July → includes 9 July). Same file participants fit on. |
| `next_stage.csv` | **Prediction target** — features for the *next* stage only, **without** `stage_rank` (e.g. 10 July). |
| `test.csv` | Written only by the evening job when a **new** stage just completed; used to score that day's open PRs. Absent during the day. |

There is **no** separate `train.csv`. `data.csv` *is* the train set.

### Example (night of 9 July 2026)

1. Fetch completes stage 6 (9 July) results from letour.fr.
2. If stage 6 was not yet in the previous `data.csv`, write `test.csv` = stage 6 (for scoring).
3. Append stage 6 into `data.csv` (now history through 9 July).
4. Write `next_stage.csv` = stage 7 (10 July) features, no labels.
5. Score **open PRs** created/updated on 9 July (Paris); publish to Hub project `9_juillet`.

## Target

`stage_rank` — finishing position on that stage (`1` = stage winner). Treated as regression.

## Columns

| Column | Description |
| --- | --- |
| `year` | Tour season year |
| `stage_number` | Stage number (1–21) |
| `stage_date` | Stage date (`YYYY-MM-DD`) |
| `rider_id` | Stable rider slug |
| `rider_name` | Display name |
| `team` | Team name |
| `nationality` | Country code when available |
| `bib` | Race number |
| `age` | Rider age when available |
| `stage_type` | `flat` / `hilly` / `mountain` / `itt` / `ttt` / `unknown` |
| `distance_km` | Stage distance in km |
| `profile_icon` | Profile hint when available |
| `stage_name` | Stage title |
| `pcs_points` / `uci_points` | Points when available |
| `prior_stages_ridden` | Count of prior stages finished this Tour |
| `avg_prior_stage_rank` | Mean of prior stage ranks |
| `best_prior_stage_rank` | Best (min) prior stage rank |
| `last_stage_rank` | Rank on the previous stage |
| `gc_rank_before` | GC rank entering the stage |
| `gc_time_gap_before_s` | GC time gap to leader before the stage (seconds) |
| `days_since_start` | Days since race start |
| `stage_rank` | **Target** (null in `next_stage.csv`) |

## Sources

- **Current season:** [letour.fr](https://www.letour.fr/en/rankings) official rankings (HTML + ajax GC tables).
- **Previous season (default year−1):** [ProCyclingStats](https://www.procyclingstats.com) stage pages via the [Wayback Machine](https://web.archive.org/) (PCS itself is Cloudflare-blocked).

```bash
python scripts/fetch_tdf_data.py              # 2025 history + current year through today
python scripts/fetch_tdf_data.py --skip-history
```

Some 2025 stages may be missing when Wayback has no usable snapshot; the fetch logs
those stages and continues. Re-run later to backfill.
