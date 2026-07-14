"""Hierarchical Bayesian latent-score ("rankit") model in NumPyro.

Within-stage ranks are mapped to normal scores z_is = Phi^-1((rank - 0.5) / n_s)
and modeled as z_is ~ StudentT(nu, mu_is, sigma_is) with
mu_is = sum_k beta[k, type] * x_kis + u[rider, type] + theta[team, gate] and
log sigma_is = tau[type] + kappa * 1[year == 2025].

Rider aptitude u is a per-rider 4-vector over stage types (flat/hilly/mountain/itt)
drawn from MVN(0, diag(sigma_u) Omega diag(sigma_u)) with an LKJ(2) correlation
prior — a rider's ITT aptitude is imputed through their hilly/mountain aptitude
via Omega, since 2026 has no ITT before stage 16.

Team intercepts theta are type-varying and gated to mountain/itt stages only —
team effects are stage-type-graded (mountain/itt ICC ~0.14 vs flat 0.03), so a
scalar team term would spend its budget where teams don't matter. Teams are
keyed (year, team) like riders. Flat/hilly rows get no team term.

The pooled avg_prior_stage_rank covariate is replaced by the same-context prior
mean: the rider's mean prior within-stage outcome normal score restricted to
same-context stages (flat stages use flat history; hilly/mountain/itt share the
climb-manifold history). Pooled history contaminates flat predictions with
climbing form. Missing history (a year's first same-context stage) maps to the
field average via the within-stage normal-score transform.

Self-contained on purpose: the nightly scoring workflow overlays only the
submissions/ folder from the PR onto main, so no code outside this directory
can be imported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import skore  # noqa: F401  (required by CI static check)
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, RegressorMixin

STAGE_KEY = ["year", "stage_number"]
COVARIATES = ["gc_rank_before", "same_context_prior_mean", "last_stage_rank"]
STAGE_TYPE_COL = "stage_type"
NUM_CHAINS = 4

# flat history informs flat stages only; hilly/mountain/itt share the climb manifold.
CONTEXT = {"flat": "flat", "hilly": "climb", "mountain": "climb", "itt": "climb"}
# team intercepts fire only on these types; index = gate dim.
GATED_TYPES = {"mountain": 0, "itt": 1}

numpyro.set_host_device_count(NUM_CHAINS)


def normal_score(values: pd.Series) -> pd.Series:
    """Within one stage: rank values (average ties), z = norm.ppf((rank - 0.5) / n_valid).

    NaN maps to 0.0 (field average) and is excluded from n_valid and from the
    ranking of the other values.
    """
    valid = values.notna()
    n_valid = int(valid.sum())

    scores = pd.Series(0.0, index=values.index)
    if n_valid == 0:
        return scores

    ranks = rankdata(values[valid])
    scores[valid] = norm.ppf((ranks - 0.5) / n_valid)
    return scores


def covariate_scores(X: pd.DataFrame) -> pd.DataFrame:
    """Within-stage normal scores of COVARIATES. X must already carry the raw
    same_context_prior_mean column."""
    grouped = X.groupby(STAGE_KEY, sort=False)
    scored = {col: grouped[col].transform(normal_score) for col in COVARIATES}
    return pd.DataFrame(scored, index=X.index)[COVARIATES]


def outcome_score(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """z_is = norm.ppf((rank_within_stage(y) - 0.5) / n_s) grouped by X[STAGE_KEY]."""
    stage_id = X[STAGE_KEY].apply(tuple, axis=1)
    return y.groupby(stage_id).transform(normal_score)


def training_mask(X: pd.DataFrame) -> pd.Series:
    """True for usable rows.

    Excludes (year == 2026) & (stage_number == 1) — a TTT with fabricated
    individual ranks, not a real within-stage ordering.
    """
    excluded = (X["year"] == 2026) & (X["stage_number"] == 1)
    return ~excluded


def same_context_prior_mean(z: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Per row: mean of the rider's outcome normal scores over strictly earlier
    same-context stages of the same year. NaN where no such history exists."""
    d = pd.DataFrame(
        {
            "year": X["year"],
            "bib": X["bib"],
            "ctx": X[STAGE_TYPE_COL].map(CONTEXT),
            "stage": X["stage_number"],
            "z": z,
        }
    ).sort_values("stage", kind="stable")
    prior = d.groupby(["year", "bib", "ctx"])["z"].transform(
        lambda s: s.expanding().mean().shift(1)
    )
    return prior.reindex(X.index)


def model(
    covariates,
    type_idx,
    rider_idx,
    team_idx,
    gate_idx,
    is_2025,
    num_types,
    num_riders,
    num_teams,
    obs=None,
):
    """Rankit likelihood. covariates: (n, 3); indices: (n,); is_2025: (n,) float.

    gate_idx is the GATED_TYPES dim for mountain/itt rows, -1 elsewhere (no
    team term).
    """
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    # Normal(0, 0.5) regularizes the correlated prior-rank covariates.
    beta = numpyro.sample("beta", dist.Normal(0.0, 0.5).expand([len(COVARIATES), num_types]))
    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(0.5).expand([num_types]))
    # LKJ(2) mildly concentrates toward 0 but supports the observed positive
    # manifold of terrain aptitudes (hilly-mountain corr ~0.8).
    L_omega = numpyro.sample("L_omega", dist.LKJCholesky(num_types, concentration=2.0))
    # flat stages are near-lotteries, mountain deterministic; kappa (2025 noise
    # inflation) is weakly supported, keep near-zero-centered.
    tau = numpyro.sample("tau", dist.Normal(jnp.log(0.7), 0.5).expand([num_types]))
    kappa = numpyro.sample("kappa", dist.Normal(0.0, 0.3))
    # team ICC is small on average; shrinkage prior lets the likelihood earn it.
    sigma_team = numpyro.sample("sigma_team", dist.HalfNormal(0.25).expand([len(GATED_TYPES)]))

    scale_tril = sigma_u[:, None] * L_omega
    with numpyro.plate("riders", num_riders):
        z_u = numpyro.sample("z_u", dist.Normal(0.0, 1.0).expand([num_types]).to_event(1))
    u = numpyro.deterministic("u", z_u @ scale_tril.T)  # (num_riders, num_types)

    with numpyro.plate("teams", num_teams):
        z_theta = numpyro.sample(
            "z_theta", dist.Normal(0.0, 1.0).expand([len(GATED_TYPES)]).to_event(1)
        )
    theta = numpyro.deterministic("theta", z_theta * sigma_team)  # (num_teams, n_gated)

    gated = gate_idx >= 0
    safe_gate = jnp.where(gated, gate_idx, 0)
    theta_is = jnp.where(gated, theta[team_idx, safe_gate], 0.0)

    beta_is = beta[:, type_idx].T  # (n, 3)
    mu = jnp.sum(beta_is * covariates, axis=-1) + u[rider_idx, type_idx] + theta_is
    sigma = jnp.exp(tau[type_idx] + kappa * is_2025)

    with numpyro.plate("obs", covariates.shape[0]):
        numpyro.sample("z", dist.StudentT(nu, mu, sigma), obs=obs)


class RankitEstimator(BaseEstimator, RegressorMixin):
    """Rankit mixed model: type-specific beta, rider-by-terrain aptitude, gated
    team intercepts, type dispersion.

    The same-context history covariate is built leakage-free at fit time (expanding
    mean over strictly earlier same-context stages); the history is stored so
    predict-time rows draw the same quantity from training data.

    Predictions are float expected within-stage ranks (monotone with finishing
    rank). Ranks are computed within each posterior draw and averaged over draws
    (Shen & Louis 1998; never rank posterior-mean mu).
    """

    def __init__(self, seed: int = 0, num_warmup: int = 1500, num_samples: int = 1000):
        self.seed = seed
        self.num_warmup = num_warmup
        self.num_samples = num_samples

    def _design(self, X: pd.DataFrame, rider_index=None, type_index=None, team_index=None):
        """X must already carry the raw same_context_prior_mean column."""
        covariates = covariate_scores(X)[COVARIATES].to_numpy(dtype="float32")

        riders = list(zip(X["year"], X["bib"]))
        if rider_index is None:
            rider_index = {r: i for i, r in enumerate(dict.fromkeys(riders))}
        rider_idx = np.array([rider_index.get(r, -1) for r in riders])

        if type_index is None:
            type_index = {t: i for i, t in enumerate(sorted(X[STAGE_TYPE_COL].unique()))}
        type_idx = np.array([type_index.get(t, -1) for t in X[STAGE_TYPE_COL]])

        teams = list(zip(X["year"], X["team"]))
        if team_index is None:
            team_index = {t: i for i, t in enumerate(dict.fromkeys(teams))}
        team_idx = np.array([team_index.get(t, -1) for t in teams])
        gate_idx = np.array([GATED_TYPES.get(t, -1) for t in X[STAGE_TYPE_COL]])
        # unseen team → no team term, like flat/hilly rows
        gate_idx = np.where(team_idx >= 0, gate_idx, -1)
        team_idx = np.where(team_idx >= 0, team_idx, 0)

        is_2025 = (X["year"] == 2025).to_numpy(dtype="float32")
        return (
            covariates,
            type_idx,
            rider_idx,
            team_idx,
            gate_idx,
            is_2025,
            rider_index,
            type_index,
            team_index,
        )

    def _history_prior_mean(self, X: pd.DataFrame) -> pd.Series:
        """Predict-time covariate: mean stored z over same-context training stages
        strictly before each row's stage. NaN where no history."""
        key = pd.DataFrame(
            {
                "year": X["year"],
                "bib": X["bib"],
                "ctx": X[STAGE_TYPE_COL].map(CONTEXT),
                "stage": X["stage_number"],
                "row": X.index,
            }
        )
        merged = key.merge(self.history_, on=["year", "bib", "ctx"], how="left")
        merged = merged[merged["hist_stage"] < merged["stage"]]
        return merged.groupby("row")["z"].mean().reindex(X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        mask = training_mask(X)
        X_train = X.loc[mask]
        y_train = (y.loc[mask] if hasattr(y, "loc") else pd.Series(y, index=X.index).loc[mask])

        z = outcome_score(y_train, X_train)
        self.history_ = pd.DataFrame(
            {
                "year": X_train["year"],
                "bib": X_train["bib"],
                "ctx": X_train[STAGE_TYPE_COL].map(CONTEXT),
                "hist_stage": X_train["stage_number"],
                "z": z,
            }
        ).dropna(subset=["ctx"])

        X_train = X_train.assign(same_context_prior_mean=same_context_prior_mean(z, X_train))
        (
            covariates,
            type_idx,
            rider_idx,
            team_idx,
            gate_idx,
            is_2025,
            rider_index,
            type_index,
            team_index,
        ) = self._design(X_train)
        self.rider_index_ = rider_index
        self.type_index_ = type_index
        self.team_index_ = team_index

        args = (
            jnp.asarray(covariates),
            jnp.asarray(type_idx),
            jnp.asarray(rider_idx),
            jnp.asarray(team_idx),
            jnp.asarray(gate_idx),
            jnp.asarray(is_2025),
            len(type_index),
            len(rider_index),
            len(team_index),
        )
        key = jax.random.PRNGKey(self.seed)

        mcmc = MCMC(
            # init_to_median avoids a degenerate "no-rider-structure" basin
            # (sigma_u -> 0, nu -> inf) reachable from default prior-sampled inits
            NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=NUM_CHAINS,
            progress_bar=False,
        )
        mcmc.run(key, *args, obs=jnp.asarray(z.to_numpy(dtype="float32")))
        self.posterior_samples_ = mcmc.get_samples()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.assign(same_context_prior_mean=self._history_prior_mean(X))
        covariates, type_idx, rider_idx, team_idx, gate_idx, is_2025, _, _, _ = self._design(
            X, self.rider_index_, self.type_index_, self.team_index_
        )
        beta = np.asarray(self.posterior_samples_["beta"])  # (m, 3, types)
        u = np.asarray(self.posterior_samples_["u"])  # (m, num_riders, types)
        theta = np.asarray(self.posterior_samples_["theta"])  # (m, num_teams, n_gated)
        sigma_u = np.asarray(self.posterior_samples_["sigma_u"])  # (m, types)
        L_omega = np.asarray(self.posterior_samples_["L_omega"])  # (m, types, types)
        num_draws = beta.shape[0]

        beta_mean_type = beta.mean(axis=2)  # (m, 3), fallback for unseen types
        seen_type = type_idx >= 0
        safe_type = np.where(seen_type, type_idx, 0)
        beta_is = beta[:, :, safe_type]  # (m, 3, n)
        beta_is = np.where(seen_type, beta_is, beta_mean_type[:, :, None])

        cov = covariates.T[None, :, :]  # (1, 3, n)
        mu = np.sum(beta_is * cov, axis=1)  # (m, n)

        seen_rider = rider_idx >= 0
        safe_rider = np.where(seen_rider, rider_idx, 0)
        u_seen = u[:, safe_rider, safe_type]  # (m, n), aptitude for this stage's terrain
        u_seen_avg = u[:, safe_rider, :].mean(axis=2)  # (m, n), avg over types, unseen-type fallback
        u_seen = np.where(seen_type, u_seen, u_seen_avg)

        scale_tril = sigma_u[:, :, None] * L_omega  # (m, types, types)
        rng = np.random.default_rng(self.seed)
        eps = rng.standard_normal((num_draws, seen_rider.size, sigma_u.shape[1]))  # (m, n, types)
        u_unseen_full = np.einsum("mij,mnj->mni", scale_tril, eps)  # (m, n, types)
        u_unseen_type = np.take_along_axis(
            u_unseen_full, safe_type[None, :, None], axis=2
        )[:, :, 0]  # (m, n)
        u_unseen = np.where(seen_type, u_unseen_type, u_unseen_full.mean(axis=2))
        u_is = np.where(seen_rider, u_seen, u_unseen)
        mu = mu + u_is

        gated = gate_idx >= 0
        safe_gate = np.where(gated, gate_idx, 0)
        theta_is = np.where(gated, theta[:, team_idx, safe_gate], 0.0)  # (m, n)
        mu = mu + theta_is

        expected_rank = np.empty(len(X), dtype=float)
        stage_id = list(zip(X["year"], X["stage_number"]))
        for stage in dict.fromkeys(stage_id):
            cols = np.array([s == stage for s in stage_id])
            ranks = rankdata(mu[:, cols], axis=1)  # rank within draw within stage
            expected_rank[cols] = ranks.mean(axis=0)
        return expected_rank


def build_estimator():
    """Return an unfitted sklearn-compatible estimator."""
    return RankitEstimator()
