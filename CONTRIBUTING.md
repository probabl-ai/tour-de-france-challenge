# Contributing a submission

## 1. Fork / branch

Create a branch from `main` and copy the skeleton:

```bash
cp -r submissions/_skeleton submissions/<your-github-login>
```

Use your GitHub username as the folder name. That name is the Hub report key when CI publishes.

## 2. Implement `build_estimator`

Edit `submissions/<login>/submission.py`:

```python
def build_estimator():
    """Return an unfitted sklearn-compatible estimator or Pipeline."""
    ...
```

Requirements:

- Must return an object with `fit` and `predict`.
- Must use the **skore** library somewhere in your submission (import is checked by AST).
- Optional `requirements.txt` in your folder for extra packages (installed on top of the
  challenge baseline: `skore[hub]`, `scikit-learn`, `pandas`, `skrub`, …).

Recommended workflow with [probabl-ai/skills](https://github.com/probabl-ai/skills):

```bash
npx skills add probabl-ai/skills
# or: skore skills install
```

## 3. Train / predict locally

- **Fit on** `data/data.csv` (labeled history through the latest completed stage).
- **Predict for** `data/next_stage.csv` (upcoming stage, no `stage_rank`).
- If `data.csv` contains 9 July, you are predicting 10 July.

ID-like columns are dropped by the harness before `fit` / `predict`. Target column is
`stage_rank`.

```bash
python scripts/check_skore_usage.py submissions/<login>
python scripts/run_submission.py submissions/<login> --allow-holdout
```

## 4. Open a PR

PRs that touch `submissions/**` run `validate-pr.yml`:

- Isolated venv + your `requirements.txt`
- Skore usage check (fails if skore is not imported)
- Dry-run evaluation (no Hub publish)

Do not commit secrets, Hub API keys, or large binary artifacts.

## 5. After merge

The nightly `fetch-and-score` workflow:

1. Refreshes `data/data.csv` and `next_stage.csv` from letour.fr (+ Wayback history)
2. Scores every `submissions/*/` folder (except `_skeleton`)
3. Publishes each report to Skore Hub project `{SKORE_WORKSPACE}/tdf-YYYY-MM-DD`
