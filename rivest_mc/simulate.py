"""
Monte Carlo machinery for evaluating the Rivest post-calving estimator.

Three independent pieces, each swappable:

  1. a group-size process        -- how T animals are split into groups
  2. a collar-allocation process -- how n collars land among those groups,
                                    with a tunable departure from random mixing
  3. a phase-II detection process -- which collared groups get photographed

The estimator then sees only what a survey crew would see: the detected groups
and their collar counts.  Because the generator knows T, bias, RMSE and
interval coverage are all directly measurable -- which is the whole point of a
simulation study and is not possible from field data alone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

__all__ = [
    "fit_powerlaw_alpha", "group_sizes_powerlaw", "group_sizes_lognormal",
    "allocate_collars", "detect_groups", "Scenario", "simulate_once", "run_scenario",
]


# --------------------------------------------------------------------------
# 1. group sizes
# --------------------------------------------------------------------------

def fit_powerlaw_alpha(sizes, xmin: float | None = None) -> tuple[float, float]:
    """MLE of the continuous power-law exponent (Newman 2005, eq. 5).

        alpha_hat = 1 + n / sum(log(x_i / xmin))

    Returns (alpha_hat, xmin).  Using this on the observed group sizes is
    strictly better than fixing alpha at an arbitrary value: it makes the
    simulated group-size distribution match the herd being studied.
    """
    x = np.asarray(sizes, dtype=float)
    x = x[x > 0]
    if xmin is None:
        xmin = float(x.min())
    x = x[x >= xmin]
    if x.size < 2:
        return np.nan, xmin
    return float(1.0 + x.size / np.sum(np.log(x / xmin))), float(xmin)


def group_sizes_powerlaw(T: int, n_groups: int, alpha: float,
                         xmin: float = 1.0, rng=None) -> np.ndarray:
    """Split T animals into ~n_groups groups with power-law relative sizes.

    Draws continuous power-law variates by inverse transform,
    x = xmin * (1-u)^(-1/(alpha-1)), rescales them to total T, rounds, and
    drops empty groups.

    NOTE ON alpha.  The continuous power law has infinite mean for alpha <= 2
    and infinite variance for alpha <= 3.  At alpha = 1.4 -- a value that has
    been used in this literature -- a single group typically absorbs almost the
    whole herd while every other group sits near xmin.  That is not what
    post-calving aggregations look like, and it is a regime where the Rivest
    inclusion probabilities degenerate.  Prefer alpha fitted from the data
    (see fit_powerlaw_alpha), and treat alpha < 2 as a stress test rather than
    a baseline.
    """
    rng = np.random.default_rng() if rng is None else rng
    u = rng.random(n_groups)
    rel = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
    sizes = np.round(T * rel / rel.sum()).astype(np.int64)
    return sizes[sizes > 0]


def group_sizes_lognormal(T: int, n_groups: int, sigma: float = 1.5,
                          rng=None) -> np.ndarray:
    """Lognormal alternative to the power law.

    Worth running alongside the power law: group-size distributions are often
    equally well or better described by a lognormal, and showing that the
    estimator's behaviour does not hinge on that choice is a cheap robustness
    result.
    """
    rng = np.random.default_rng() if rng is None else rng
    rel = rng.lognormal(mean=0.0, sigma=sigma, size=n_groups)
    sizes = np.round(T * rel / rel.sum()).astype(np.int64)
    return sizes[sizes > 0]


# --------------------------------------------------------------------------
# 2. collar allocation, with a clustering dial
# --------------------------------------------------------------------------

def allocate_collars(sizes: np.ndarray, n_collars: int,
                     kappa: float = np.inf, rng=None) -> np.ndarray:
    """Distribute n collars among groups, optionally violating random mixing.

    kappa = inf   exact multinomial with probabilities proportional to group
                  size.  This IS the random-mixing assumption the Rivest
                  estimator is derived under, so it is the null case.
    kappa finite  Dirichlet-multinomial: p ~ Dirichlet(kappa * w) with
                  w = size proportions, then collars ~ Multinomial(n, p).
                  Smaller kappa means collars clump into fewer groups than
                  their sizes warrant -- collar clustering.  kappa -> inf
                  recovers the multinomial, so the null is nested inside the
                  alternative and the dial is continuous.

    Returns an integer vector of collar counts, one per group (zeros included).
    """
    rng = np.random.default_rng() if rng is None else rng
    w = sizes / sizes.sum()
    if not np.isfinite(kappa):
        p = w
    else:
        p = rng.dirichlet(np.maximum(kappa * w, 1e-12))
    counts = rng.multinomial(n_collars, p)
    # a group cannot hold more collars than animals
    return np.minimum(counts, sizes)


# --------------------------------------------------------------------------
# 3. phase-II detection
# --------------------------------------------------------------------------

def detect_groups(collars: np.ndarray, model: str = "H", r: float = 0.9,
                  B: int = 2, rng=None) -> np.ndarray:
    """Boolean mask of which groups the survey actually photographs.

    Only groups holding at least one collar can be detected at all -- that is
    the sampling frame.  Then:

      H  each such group is detected with common probability r
      I  each collar is independently detected w.p. (1-r), so a group with x
         collars is detected w.p. 1 - r^x
      T  groups with >= B collars are always detected; others w.p. r
    """
    rng = np.random.default_rng() if rng is None else rng
    has = collars >= 1
    prob = np.zeros(collars.size)
    if model == "H":
        prob[has] = r
    elif model == "I":
        prob[has] = 1.0 - r ** collars[has]
    elif model == "T":
        prob[has] = np.where(collars[has] >= B, 1.0, r)
    else:
        raise ValueError("model must be 'H', 'I' or 'T'")
    return rng.random(collars.size) < prob


# --------------------------------------------------------------------------
# scenario definition and the replicate loop
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Scenario:
    """One cell of the simulation design."""

    T: int = 100_000                # true herd size
    n_groups: int = 30              # groups the herd splits into
    n_collars: int = 100            # ACTIVE collars
    alpha: float = 2.5              # power-law exponent (group-size dispersion)
    size_model: str = "powerlaw"    # 'powerlaw' | 'lognormal'
    sigma: float = 1.5              # lognormal dispersion, if used
    kappa: float = np.inf           # collar clustering; inf = random mixing
    detect_model: str = "H"         # truth: how detection really works
    r: float = 0.9                  # detection parameter
    B: int = 2                      # threshold, if detect_model == 'T'
    fit_model: str = "H"            # what the ANALYST assumes when estimating

    def label(self) -> str:
        k = "inf" if not np.isfinite(self.kappa) else f"{self.kappa:g}"
        return (f"T={self.T} G={self.n_groups} n={self.n_collars} "
                f"a={self.alpha} kappa={k} true={self.detect_model} "
                f"fit={self.fit_model} r={self.r}")


def simulate_once(sc: Scenario, rng, *,
                  include_zero_collar_groups: bool = False,
                  n_for_estimator: str = "active") -> dict:
    """Generate one survey and estimate T from it.

    include_zero_collar_groups
        Reproduces the most common implementation error: passing every
        simulated group to the estimator instead of only the detected,
        collared ones.  Groups with zero collars are outside the support of
        the conditional likelihood, and each one added contributes roughly
        T/(r*n) to the estimating equation, forcing T_hat upward.
    n_for_estimator
        'active' (correct) uses the number of collars deployed; 'found' uses
        only the collars seen, another common error which inflates r_hat to 1
        and biases T_hat downward.
    """
    from .estimator import abundance

    if sc.size_model == "powerlaw":
        sizes = group_sizes_powerlaw(sc.T, sc.n_groups, sc.alpha, rng=rng)
    else:
        sizes = group_sizes_lognormal(sc.T, sc.n_groups, sc.sigma, rng=rng)
    if sizes.size < 2:
        return {"ok": False, "reason": "degenerate group sizes"}

    T_true = int(sizes.sum())        # the realised herd size -- the estimand
    collars = allocate_collars(sizes, sc.n_collars, sc.kappa, rng=rng)
    seen = detect_groups(collars, sc.detect_model, sc.r, sc.B, rng=rng)

    if include_zero_collar_groups:
        keep = np.ones(sizes.size, dtype=bool)      # the bug: everything goes in
    else:
        keep = seen & (collars >= 1)                # what a crew actually records

    if keep.sum() < 2:
        return {"ok": False, "reason": "fewer than two detected groups",
                "T_true": T_true}

    mat = np.column_stack([collars[keep], sizes[keep]])
    n_est = sc.n_collars if n_for_estimator == "active" else int(collars[keep].sum())
    if n_est < 1:
        return {"ok": False, "reason": "no collars", "T_true": T_true}

    try:
        res = abundance(mat, n=n_est, model=sc.fit_model,
                        B=sc.B if sc.fit_model == "T" else None,
                        allow_zero_collar_groups=include_zero_collar_groups)
    except Exception as exc:                                   # noqa: BLE001
        return {"ok": False, "reason": f"estimator: {exc}", "T_true": T_true}

    lo_log, hi_log = res.ci(method="log")
    lo_nrm, hi_nrm = res.ci(method="normal")
    return {
        "ok": True,
        "T_true": T_true,
        "T_hat": res.T_hat,
        "se": res.se_T_hat,
        "counted": res.gnt,
        "mp": res.mp,
        "converged": res.converged,
        "cover_log": bool(lo_log <= T_true <= hi_log),
        "cover_normal": bool(lo_nrm <= T_true <= hi_nrm),
        "neg_lower": bool(lo_nrm < 0),
        "z_random": res.z_random,
        "reject_random": bool(np.isfinite(res.p_random) and res.p_random < 0.05),
        "rr": res.rr,
    }


def run_scenario(sc: Scenario, n_rep: int = 1000, seed: int = 20260822,
                 **kwargs) -> dict:
    """Replicate a scenario and summarise the sampling distribution of T_hat.

    This is the step the original analysis omitted.  Bias, RMSE and coverage
    are properties of a distribution; a single replicate estimates none of
    them.  n_rep >= 1000 keeps the Monte Carlo error on coverage under about
    1.4 percentage points.
    """
    rng = np.random.default_rng(seed)
    T_hat, T_true, cov_l, cov_n, neg, rej, ses = [], [], [], [], [], [], []
    n_fail = n_degen = 0

    for _ in range(n_rep):
        out = simulate_once(sc, rng, **kwargs)
        if not out["ok"]:
            n_fail += 1
            continue
        rej.append(out["reject_random"])
        if not out["converged"]:
            # The estimating equation had no interior root, so the routine
            # returned a bracket endpoint. These are not estimates and must not
            # be averaged in with real ones -- doing so is what makes a broken
            # run look unbiased. They are counted and reported instead.
            n_degen += 1
            continue
        T_hat.append(out["T_hat"]); T_true.append(out["T_true"])
        cov_l.append(out["cover_log"]); cov_n.append(out["cover_normal"])
        neg.append(out["neg_lower"]); ses.append(out["se"])

    degen_pct = 100.0 * n_degen / n_rep
    if not T_hat:
        return {**asdict(sc), "n_rep": n_rep, "n_ok": 0,
                "fail_rate": n_fail / n_rep, "degenerate_pct": degen_pct,
                "power_random_test_pct": (100 * float(np.mean(rej))) if rej else np.nan}

    Th, Tt = np.array(T_hat, float), np.array(T_true, float)
    err = Th - Tt
    return {
        **asdict(sc),
        "n_rep": n_rep,
        "n_ok": Th.size,
        "fail_rate": n_fail / n_rep,
        "degenerate_pct": degen_pct,
        "mean_T_true": float(Tt.mean()),
        "mean_T_hat": float(Th.mean()),
        "median_T_hat": float(np.median(Th)),
        "rel_bias_pct": float(100 * err.mean() / Tt.mean()),
        "median_rel_bias_pct": float(100 * np.median(err / Tt)),
        "rel_rmse_pct": float(100 * np.sqrt((err ** 2).mean()) / Tt.mean()),
        "cover_log_pct": float(100 * np.mean(cov_l)),
        "cover_normal_pct": float(100 * np.mean(cov_n)),
        "neg_lower_pct": float(100 * np.mean(neg)),
        "power_random_test_pct": float(100 * np.mean(rej)),
        "mean_se": float(np.mean(ses)),
        "sd_T_hat": float(Th.std(ddof=1)),
    }
