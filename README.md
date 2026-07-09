# Tour de France Skore Challenge

Predict the **next stage finishing rank** of every rider still in the Tour de France.
Each evening, CI refreshes the dataset from [letour.fr](https://www.letour.fr/en/rankings)
(current season) plus the previous season via Wayback/PCS, scores every accepted submission
with [skore](https://docs.skore.probabl.ai/), and publishes reports to a private
[Skore Hub](https://skore.probabl.ai/) project for that race day.

## How it works

1. **Nightly fetch** — `scripts/fetch_tdf_data.py` updates `data/data.csv` through the latest
   completed stage and writes blind features for the upcoming stage to `next_stage.csv`.
2. **Score submissions** — fit on `data.csv` (excluding the newly completed stage), evaluate on
   that stage’s labels, push an `EstimatorReport` to Hub.
3. **Iterate during the day** — develop against `data/next_stage.csv` (no labels) and open a PR.

```text
evening cron
  → fetch letour.fr (+ Wayback history) → update data.csv / next_stage.csv
  → for each submission: isolated venv → skore check → evaluate → Hub put
```

### `data.csv` vs prediction day

`data.csv` **is** the training set (there is no separate `train.csv`).

If `data.csv` includes **9 July**, the prediction target is **10 July** (`next_stage.csv`).

See [`data/README.md`](data/README.md).

## Predict

| Field | Meaning |
| --- | --- |
| Target | `stage_rank` — finishing position on the **next** stage (1 = winner) |
| Task | Regression (rank as a continuous target so skore reports work out of the box) |
| Rows | One row per rider still in the race for that stage |

## Submit

1. Copy the skeleton:

   ```bash
   cp -r submissions/_skeleton submissions/<your-github-login>
   ```

2. Implement `build_estimator()` in `submissions/<login>/submission.py` (must use **skore**).
3. Open a PR. [`validate-pr.yml`](.github/workflows/validate-pr.yml) checks the contract and
   runs a dry evaluation (no Hub publish).
4. After merge, the evening job scores your model and publishes to Hub.

Details: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Recommended: Probabl skills

Use the [probabl-ai/skills](https://github.com/probabl-ai/skills) pack so your agent follows
good ML methodology with skore:

```bash
# via skore-cli
pip install skore-cli && skore skills install

# or
npx skills add probabl-ai/skills
```

Useful skills: `explore-ml-data`, `build-ml-pipeline`, `evaluate-ml-pipeline`,
`iterate-ml-experiment`.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Fetch / refresh data:

```bash
python scripts/fetch_tdf_data.py                 # history year + current season
python scripts/fetch_tdf_data.py --skip-history  # letour.fr only (faster)
```

Dry-run a submission:

```bash
python scripts/check_skore_usage.py submissions/_skeleton
python scripts/run_submission.py submissions/_skeleton --allow-holdout
```

## Repo secrets (maintainers)

| Secret | Purpose |
| --- | --- |
| `SKORE_API_KEY` | Private workspace API key for `skore.login(mode="hub")` |

Hub workspace is public config: `tour-de-france-challenge` (projects are named `tdf-YYYY-MM-DD`).

Optional: `TDF_YEAR` repository variable to override the Tour season year.

## Layout

```text
data/                 # data.csv + next_stage.csv (+ ephemeral test.csv)
scripts/              # fetch, skore gate, evaluation harness
submissions/          # one folder per participant (+ _skeleton)
.github/workflows/    # nightly fetch+score, PR validation
```

## License

MIT
