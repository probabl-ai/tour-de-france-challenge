# Tour de France Skore Challenge

Predict the **next stage finishing rank** of every rider still in the Tour de France.
Each evening, CI refreshes the dataset from [letour.fr](https://www.letour.fr/en/rankings),
scores **open PRs** that were opened or updated that calendar day (Europe/Paris), and
publishes reports to [Skore Hub](https://skore.probabl.ai/).

## How it works

1. **During the day** — develop against `data/next_stage.csv` (no labels) and open a PR with
   your code under `submissions/<github-login>/`. **Do not merge** — leave the PR open.
2. **Nightly fetch** — `scripts/fetch_tdf_data.py` updates `data/data.csv` through the stage
   that just finished and writes the next blind `next_stage.csv`.
3. **Nightly score** — only open PRs **created or last updated on that stage’s calendar day**
   (Paris) are checked out and evaluated; reports go to Hub project `{day}_juillet`
   (e.g. `8_juillet` for 8 July).

```text
evening cron
  → fetch letour.fr → update data.csv / next_stage.csv / test.csv
  → discover open PRs touched today (Paris)
  → for each PR: overlay submissions/ → skore check → evaluate → Hub put
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
3. Open a PR **the same calendar day** as the stage you are predicting (Europe/Paris).
   [`validate-pr.yml`](.github/workflows/validate-pr.yml) checks the contract (no Hub publish).
4. Leave the PR open. That evening, if your PR was opened/updated that day, CI scores it
   and publishes to Hub project `{day}_juillet`.

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
| `SKORE_API_KEY` | Mapped to `SKORE_HUB_API_KEY` for `skore.login(mode="hub")` |
| `DATA_PUSH_TOKEN` | Fine-grained PAT (your user) with access to this repo, **Contents: Read and write**, and org SSO authorized if required — used only to push `data/` past protected `main` |

Hub workspace: `tour-de-france-challenge`. Projects are named `{day}_juillet` (e.g. `9_juillet`).

Optional: `TDF_YEAR` repository variable to override the Tour season year.

## Layout

```text
data/                 # data.csv + next_stage.csv (+ ephemeral test.csv)
scripts/              # fetch, PR discovery, skore gate, evaluation harness
submissions/          # skeleton (+ PR branches carry participant folders)
.github/workflows/    # nightly fetch+score, PR validation, manual backfill
```

## License

MIT
