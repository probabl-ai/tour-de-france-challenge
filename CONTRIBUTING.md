# Contributing a submission

## 1. Branch (do not merge)

Create a branch from `main` and copy the skeleton:

```bash
cp -r submissions/_skeleton submissions/<your-github-login>
```

Use your GitHub username as the folder name. That name is the Hub report key when CI publishes.

**PRs are not meant to be merged.** Leave them open so the nightly job can check them out.

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

```bash
python scripts/check_skore_usage.py submissions/<login>
python scripts/run_submission.py submissions/<login> --allow-holdout
```

## 4. Open a PR the same day as the stage

Open (or push an update to) your PR on the **calendar day of the stage you predict**, in
**Europe/Paris** time. Example: to be scored for the 8 July stage, open/update the PR on
8 July (Paris).

[`validate-pr.yml`](.github/workflows/validate-pr.yml) runs on the PR:

- Isolated venv + your `requirements.txt`
- Skore usage check
- Dry-run evaluation (no Hub publish)

Do not commit secrets or Hub API keys.

## 5. Nightly scoring (no merge)

The evening `fetch-and-score` workflow:

1. Refreshes `data/*.csv` from letour.fr (+ Wayback history)
2. Finds **open** PRs created or last updated on that stage’s calendar day (Paris)
3. Checks out each PR’s `submissions/` onto main’s data/harness
4. Publishes each report to Skore Hub project `{day}_juillet`  
   (e.g. `tour-de-france-challenge/8_juillet`)
