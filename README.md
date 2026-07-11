# Tour de France Skore Challenge

Predict the **next-stage finishing rank** of every rider still in the race.
Each evening with a new stage result, open submission PRs are scored and published
to [Skore Hub](https://skore.probabl.ai/).

**Leave your PR open** to continue participating.

## What you predict

| | |
| --- | --- |
| Target | `stage_rank` — finishing position on the **next** stage (`1` = winner) |
| Metric | **Spearman ρ** (primary; higher is better), MAE secondary |
| Train on | `data/data.csv` — labeled history through the latest completed stage |
| Predict for | `data/next_stage.csv` — next stage features, **no** `stage_rank` |

If `data.csv` includes **9 July**, you are predicting **10 July**.

## Mandatory: Probabl skore

Submissions **must** use the [skore](https://docs.skore.probabl.ai/) library
(Probabl). CI rejects any `submission.py` that does not import / use skore.

## Recommended: Probabl skills

Install the [probabl-ai/skills](https://github.com/probabl-ai/skills) pack so your
agent follows good ML methodology with skore:

```bash
pip install skore-cli && skore skills install
# or
npx skills add probabl-ai/skills
```

Useful: `explore-ml-data`, `build-ml-pipeline`, `evaluate-ml-pipeline`,
`iterate-ml-experiment`.

## Submit

1. Branch from `main` and copy the skeleton:

   ```bash
   cp -r submissions/_skeleton submissions/<your-github-login>
   ```

   Use your GitHub username as the folder name (Hub report key).

2. Implement `build_estimator()` in `submissions/<login>/submission.py`:

   ```python
   def build_estimator():
       """Return an unfitted sklearn-compatible estimator or Pipeline."""
       ...
   ```

   - Must expose `fit` / `predict`
   - Must use **skore** (see above)
   - Optional `requirements.txt` for extra packages

3. Open a PR and **leave it open**. Every evening with new stage data, CI retrains
   your estimator on the updated `data.csv` and publishes to Hub project
   `{day}_juillet` (e.g. `9_juillet`).

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Fail if submission.py does not use skore
python scripts/check_skore_usage.py submissions/<login>
# Dry-run: hold out the latest stage in data.csv (no test.csv during the day)
python scripts/run_submission.py submissions/<login> --allow-holdout
```

## Data columns

| Column | Description |
| --- | --- |
| `year` | Tour season year |
| `stage_number` | Stage number (1–21) |
| `stage_date` | Stage date (`YYYY-MM-DD`) |
| `rider_id` | Stable rider slug |
| `rider_name` | Display name |
| `team` | Team name |
| `bib` | Race number |
| `age` | Rider age when available |
| `stage_type` | `flat` / `hilly` / `mountain` / `itt` / `ttt` |
| `distance_km` | Stage distance (km) |
| `stage_name` | Stage title |
| `prior_stages_ridden` | Prior stages finished this Tour |
| `avg_prior_stage_rank` | Mean of prior stage ranks |
| `best_prior_stage_rank` | Best (min) prior stage rank |
| `last_stage_rank` | Rank on the previous stage |
| `gc_rank_before` | GC rank entering the stage |
| `gc_time_gap_before_s` | GC time gap to leader before the stage (seconds) |
| `days_since_start` | Days since race start |
| `stage_rank` | **Target** (null in `next_stage.csv`) |
