"""
Rivest post-calving abundance estimator — independent Python implementation.

Ports the estimator of

    Rivest, L.-P., Couturier, S. & Crepeau, H. (1998).
    Statistical methods for estimating caribou abundance using postcalving
    aggregations detected by radio telemetry. Biometrics 54, 865-876.

as implemented in the CRAN package `caribou` (abundance(); Crepeau &
Baillargeon).  This module is written from the published equations rather than
translated line-by-line, so that agreement with the R package is a genuine
cross-validation rather than a tautology.  See tests/test_vs_r.py.

Differences from the R reference, all deliberate and all reported:

  * Root finding uses Brent's method on the estimating equation itself rather
    than golden-section minimisation of its absolute value.  R's optimize()
    silently returns an interval endpoint when no root exists in the bracket;
    that is the mechanism behind the "T.hat == observed count" degenerate rows
    seen in post-calving analyses.  Here such cases are detected and reported
    via AbundanceResult.converged / .boundary.
  * Confidence intervals are log-scale by default and therefore cannot be
    negative.  The normal-scale interval is still available for comparison.
  * Input validation rejects groups with zero collars, which are outside the
    support of the conditional likelihood.  Passing them inflates T.hat.

Author: prepared for N. Rex-Oneal, 2026.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

__all__ = ["abundance", "AbundanceResult", "petersen_chapman",
           "bootstrap_randomness_test", "calibrated_p_from_z",
           "RANDOMNESS_CRITICAL_VALUES"]


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------

@dataclass
class AbundanceResult:
    """Output of :func:`abundance`."""

    model: str
    n: int                      # total active collars in the herd
    mp: int                     # number of detected groups holding >=1 collar
    xt: int                     # collars found in those groups
    gnt: int                    # animals counted in those groups
    rr: float                   # phase-II detection parameter
    se_rr: float
    T_hat: float                # estimated herd size
    se_T_hat: float
    loglik: float
    z_random: float             # score statistic, randomness assumption
    p_random: float             # one-sided p-value (large z => clustering)
    converged: bool             # a true root of the estimating equation was found
    boundary: bool              # solution sits on the search bracket
    B: int | None = None
    pi: np.ndarray = field(default=None, repr=False)      # phase-II probs
    pi_i: np.ndarray = field(default=None, repr=False)    # phase-I probs

    # -- intervals ---------------------------------------------------------

    def ci(self, level: float = 0.95, method: str = "log") -> tuple[float, float]:
        """Confidence interval for T.

        method="log"    lognormal interval; cannot go negative.  Default.
        method="normal" T_hat +/- z*se, the interval used in most post-calving
                        reports.  Retained only so the two can be compared.
        """
        z = norm.ppf(0.5 + level / 2.0)
        if method == "normal":
            return (self.T_hat - z * self.se_T_hat, self.T_hat + z * self.se_T_hat)
        if method == "log":
            if not np.isfinite(self.se_T_hat) or self.se_T_hat <= 0:
                return (self.T_hat, self.T_hat)
            cv2 = (self.se_T_hat / self.T_hat) ** 2
            c = np.exp(z * np.sqrt(np.log1p(cv2)))
            return (self.T_hat / c, self.T_hat * c)
        raise ValueError("method must be 'log' or 'normal'")

    def summary(self) -> str:
        lo, hi = self.ci()
        flag = "" if self.converged else "   [NO ROOT - boundary solution]"
        return (
            f"Rivest estimator, model {self.model}"
            f"{'' if self.B is None else f' (B={self.B})'}\n"
            f"  active collars n            : {self.n}\n"
            f"  detected groups with collars: {self.mp}\n"
            f"  collars in those groups     : {self.xt}\n"
            f"  animals counted             : {self.gnt:,}\n"
            f"  detection parameter r       : {self.rr:.4f}  (se {self.se_rr:.4f})\n"
            f"  T_hat                       : {self.T_hat:,.0f}  (se {self.se_T_hat:,.1f}){flag}\n"
            f"  95% CI (lognormal)          : {lo:,.0f} - {hi:,.0f}\n"
            f"  randomness score test       : z = {self.z_random:.3f}, p = {self.p_random:.3g}"
        )


# --------------------------------------------------------------------------
# phase-II detection models
# --------------------------------------------------------------------------

def _phase2(xi: np.ndarray, n: int, model: str, B: int | None):
    """Return (r_hat, pi, pi_prime) for the requested detection model.

    pi        probability that a group holding xi collars is detected
    pi_prime  d pi / d r, needed for the delta-method variance
    """
    if model == "H":
        # every group holding >=1 collar is detected with common probability r
        r = xi.sum() / n
        if r > 1:
            raise ValueError(
                f"collars found ({xi.sum()}) exceed active collars n={n}; "
                "r_hat > 1 is not admissible"
            )
        return r, np.full(xi.size, r), np.ones(xi.size)

    if model == "I":
        # each collar is detected independently with probability 1-r
        def est_eq(r):
            return np.sum(xi / (1.0 - r ** xi)) - n

        lo, hi = 1e-12, 1.0 - 1e-12
        f_lo, f_hi = est_eq(lo), est_eq(hi)
        if f_lo * f_hi < 0:
            r = brentq(est_eq, lo, hi, xtol=1e-12)
        else:
            r = minimize_scalar(
                lambda t: abs(est_eq(t)), bounds=(lo, hi), method="bounded"
            ).x
        pi = 1.0 - r ** xi
        pi_p = -xi * r ** (xi - 1)
        return r, pi, pi_p

    if model == "T":
        if B is None:
            raise ValueError("model 'T' requires a threshold B")
        big = xi >= B
        denom = n - xi[big].sum()
        if denom <= 0:
            raise ValueError("threshold model: no collars below the threshold")
        r = xi[~big].sum() / denom
        pi = np.where(big, 1.0, r)
        pi_p = np.where(big, 0.0, 1.0)
        return r, pi, pi_p

    raise ValueError("model must be one of 'H', 'I', 'T'")


# --------------------------------------------------------------------------
# score test for the random-mixing assumption (Rivest et al. 1998, sec. 5.1)
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _g_derivs():
    """Symbolic derivatives of g(t) = r*e^t + log(e^{(1-r)e^t} - 1).

    Built once with sympy and returned as fast numeric callables.  Using the
    analytic derivatives (rather than finite differences) is what lets the
    Python score statistic agree with R's D()-based version to machine
    precision.
    """
    import sympy as sp

    t, r = sp.symbols("t r", real=True)
    g = r * sp.exp(t) + sp.log(sp.exp((1 - r) * sp.exp(t)) - 1)
    ds = [sp.diff(g, t, k) for k in (1, 2, 3, 4)]
    return [sp.lambdify((t, r), d, "numpy") for d in ds]


def _randomness_test(xi, gni, n, T_hat, se_T_hat, r_for_test):
    """One-sided score test of the random-mixing (Poisson) assumption.

    Returns (z, p).  Large positive z indicates over-dispersion of collars
    among groups, i.e. collar clustering / failure of random mixing.
    """
    g1, g2, g3, g4 = _g_derivs()
    r = r_for_test

    def score_theta(theta):
        td = theta + np.log(n) + np.log(gni)
        return xi.sum() - np.sum(g1(td, r))

    lo = -np.log(max(T_hat - 2 * se_T_hat, n))
    hi = -np.log(T_hat + 2 * se_T_hat)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= hi:
        return np.nan, np.nan
    try:
        if score_theta(hi) * score_theta(lo) < 0:
            theta = brentq(score_theta, hi, lo, xtol=1e-14)
        else:
            theta = minimize_scalar(
                lambda t: abs(score_theta(t)), bounds=(hi, lo), method="bounded"
            ).x
    except (ValueError, FloatingPointError):
        return np.nan, np.nan

    td = theta + np.log(n) + np.log(gni)
    d1, d2, d3, d4 = g1(td, r), g2(td, r), g3(td, r), g4(td, r)

    with np.errstate(over="ignore", invalid="ignore"):
        vv2 = np.sum(d4 + 2 * d2 ** 2) - np.sum(d3) ** 2 / np.sum(d2)
        if not np.isfinite(vv2) or vv2 <= 0:
            return np.nan, np.nan
        z = np.sum((xi - d1) ** 2 - d2) / np.sqrt(vv2)
    return float(z), float(1.0 - norm.cdf(z))


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def abundance(
    mat,
    n: int,
    model: str = "H",
    B: int | None = None,
    max_T: float | None = None,
    allow_zero_collar_groups: bool = False,
) -> AbundanceResult:
    """Estimate herd size T from a post-calving survey.

    Parameters
    ----------
    mat : array_like, shape (m, 2)
        One row per **detected group that contains at least one radio collar**.
        Column 0 = number of collars found in the group (xi, must be >= 1).
        Column 1 = number of animals counted in the group (gni).
    n : int
        Total number of ACTIVE collars in the herd during the survey -- not the
        number found.  Getting this wrong is the most common user error.
    model : {'H', 'I', 'T'}
        Phase-II detection model: homogeneity, independence, or threshold.
    B : int, optional
        Threshold, required when model='T'.
    max_T : float, optional
        Upper end of the search bracket.  Default n * max(gni), as in `caribou`.
    allow_zero_collar_groups : bool
        If False (default) a group with xi == 0 raises.  Such groups are
        outside the support of the conditional likelihood -- under the
        homogeneity model P(X = 0) = 0 -- and including them biases T_hat
        upward, sharply.  Set True only to reproduce that error deliberately.

    Returns
    -------
    AbundanceResult
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] != 2:
        raise ValueError("mat must be a two-column array (collars, group size)")
    if np.any(mat < 0):
        raise ValueError("mat must be non-negative")
    if np.any(mat[:, 1] < mat[:, 0]):
        raise ValueError("a group cannot hold more collars than animals")

    xi, gni = mat[:, 0], mat[:, 1]
    if np.any(xi == 0):
        if not allow_zero_collar_groups:
            raise ValueError(
                f"{int((xi == 0).sum())} of {xi.size} rows have zero collars. "
                "The Rivest estimator conditions on detection, so only groups "
                "with >=1 collar belong in `mat`. Filter them out, or pass "
                "allow_zero_collar_groups=True to reproduce the bias deliberately."
            )
        warnings.warn(
            "zero-collar groups included: T_hat will be biased upward",
            RuntimeWarning, stacklevel=2,
        )

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")

    mp, xt, gnt = xi.size, int(xi.sum()), int(gni.sum())
    r, pi, pi_p = _phase2(xi, n, model, B)

    # ---- variance of r (delta method) ------------------------------------
    f1 = np.sum(xi * pi_p / pi ** 2) ** -2
    f2 = np.sum(xi ** 2 * (1 - pi) / pi ** 2)
    se_rr = float(np.sqrt(f1 * f2))

    # ---- T_hat: root of the Horvitz-Thompson estimating equation ---------
    # phase-I inclusion probability of group i is 1 - (1 - gni/T)^n
    def est_eq(t):
        return t - np.sum(gni / (pi * (1.0 - (1.0 - gni / t) ** n)))

    lo = float(gnt)
    hi = float(max_T) if max_T is not None else float(n * gni.max())
    converged, boundary = True, False
    lo_eps = lo * (1 + 1e-9)
    try:
        f_lo, f_hi = est_eq(lo_eps), est_eq(hi)
        if f_lo * f_hi < 0:
            T_hat = brentq(est_eq, lo_eps, hi, xtol=1e-8, rtol=1e-12)
        else:
            # No sign change: the estimating equation has no interior root.
            # R's optimize() returns an endpoint here without complaint; we
            # do the same but record it, because these rows are not estimates.
            T_hat = minimize_scalar(
                lambda t: abs(est_eq(t)), bounds=(lo_eps, hi), method="bounded"
            ).x
            converged = False
            boundary = (
                abs(T_hat - lo_eps) / lo_eps < 1e-4 or abs(T_hat - hi) / hi < 1e-4
            )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return AbundanceResult(
            model, n, mp, xt, gnt, r, se_rr, np.nan, np.nan, np.nan,
            np.nan, np.nan, False, True, B,
        )

    # ---- variance of T_hat (Rivest et al. 1998, eq. 6) -------------------
    pi_i = 1.0 - (1.0 - gni / T_hat) ** n
    T0n = np.sum(pi_p * gni / (pi ** 2 * pi_i))
    X0d = np.sum(pi_p * xi / pi ** 2)

    term1 = 1.0 - (n / T_hat ** 2) * np.sum(
        gni ** 2 * (1 - gni / T_hat) ** (n - 1) / (pi * pi_i ** 2)
    )
    term2 = np.sum(gni ** 2 * (1 - pi_i) / (pi * pi_i ** 2))
    term4 = np.sum((1 - pi) / pi ** 2 * (gni / pi_i - xi * T0n / X0d) ** 2)

    # joint inclusion probabilities -> the covariance term
    a = (1.0 - gni / T_hat) ** n
    pair = (gni[:, None] + gni[None, :]) / T_hat
    pi_ij = 1.0 - a[:, None] - a[None, :] + np.clip(1.0 - pair, 0.0, None) ** n
    gam = pi_ij - np.outer(pi_i, pi_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = (
            np.outer(gni, gni) * gam
            / (np.outer(pi, pi) * np.outer(pi_i, pi_i) * pi_ij)
        )
    term3 = 2.0 * np.nansum(np.triu(M, k=1))

    var_T = (term2 + term3 + term4) / term1 ** 2
    se_T = float(np.sqrt(var_T)) if var_T > 0 else np.nan

    # ---- conditional log-likelihood --------------------------------------
    lam = n * gni / T_hat
    with np.errstate(over="ignore", invalid="ignore"):
        if model == "H":
            loglik = float(np.sum(xi * np.log(lam) - np.log(np.expm1(lam))))
        elif model == "I":
            loglik = float(np.sum(
                xi * np.log(lam) + np.log(1 - r ** xi)
                - np.log(np.exp(lam) - np.exp(r * lam))
            ))
        else:
            som = sum(lam ** i / math.factorial(i) for i in range(1, B))
            loglik = float(np.sum(
                xi * np.log(lam) + np.log(1 - (1 - r) * pi_p)
                - np.log(np.expm1(lam) - (1 - r) * som)
            ))

    # ---- score test for random mixing ------------------------------------
    if model in ("H", "I"):
        z, p = _randomness_test(xi, gni, n, T_hat, se_T, 0.0 if model == "H" else r)
    else:
        z, p = np.nan, np.nan

    return AbundanceResult(
        model, n, mp, xt, gnt, float(r), se_rr, float(T_hat), se_T,
        loglik, z, p, converged, boundary, B, pi, pi_i,
    )


# --------------------------------------------------------------------------
# Lincoln-Petersen / Chapman, for comparison
# --------------------------------------------------------------------------

def bootstrap_randomness_test(mat, n: int, model: str = "H", r_detect: float | None = None,
                              n_boot: int = 5000, seed: int | None = None,
                              B: int | None = None):
    """Parametric bootstrap p-value for the random-mixing assumption.

    USE THIS INSTEAD OF THE ASYMPTOTIC p-VALUE on real surveys.

    The score test of Rivest et al. (1998, sec. 5.1) refers its statistic to a
    standard normal, but post-calving surveys detect 10-30 groups, and at those
    counts the null distribution of z is neither standard nor normal: it is
    centred below zero, narrower than N(0,1) in the body, and has a long right
    tail.  The asymptotic p-value is therefore conservative at conventional
    levels and sharply ANTI-conservative in the far tail -- for the Western
    Arctic 2011 survey it reports 4.2e-06 where the bootstrap gives 6.0e-04, an
    overstatement of more than two orders of magnitude.  The rejection stands
    either way, but the reported strength does not.

    This routine conditions on the observed group sizes and collar total, so the
    null distribution is generated for a herd shaped exactly like the one in
    hand, and the asymptotics never enter.

    Parameters
    ----------
    mat, n, model, B : as for :func:`abundance`.
    r_detect : detection probability used to generate the null surveys.
        Defaults to the r_hat estimated from the data.
    n_boot : number of null replicates (5000 gives a p-value resolved to ~2e-4).
    seed : for reproducibility.  Always set this in published work.

    Returns
    -------
    dict with keys: z_obs, p_asymptotic, p_bootstrap, n_boot_ok,
        null_mean, null_sd, null_q95.
    """
    from .simulate import allocate_collars, detect_groups

    mat = np.asarray(mat, dtype=float)
    obs = abundance(mat, n=n, model=model, B=B)
    sizes = mat[:, 1].astype(np.int64)
    r_use = obs.rr if r_detect is None else float(r_detect)
    r_use = float(np.clip(r_use, 1e-6, 1.0))

    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(n_boot):
        collars = allocate_collars(sizes, n, kappa=np.inf, rng=rng)
        keep = detect_groups(collars, model, r_use, B or 2, rng=rng) & (collars >= 1)
        if keep.sum() < 3:
            continue
        try:
            res = abundance(np.column_stack([collars[keep], sizes[keep]]),
                            n=n, model=model, B=B)
        except Exception:                                          # noqa: BLE001
            continue
        if np.isfinite(res.z_random):
            zs.append(res.z_random)

    zs = np.asarray(zs)
    if zs.size == 0:
        return {"z_obs": obs.z_random, "p_asymptotic": obs.p_random,
                "p_bootstrap": np.nan, "n_boot_ok": 0}
    # +1 in numerator and denominator: the p-value can never be exactly zero,
    # which would misrepresent a finite bootstrap as infinite evidence
    p_boot = (1 + np.sum(zs >= obs.z_random)) / (1 + zs.size)
    return {
        "z_obs": float(obs.z_random),
        "p_asymptotic": float(obs.p_random),
        "p_bootstrap": float(p_boot),
        "n_boot_ok": int(zs.size),
        "null_mean": float(zs.mean()),
        "null_sd": float(zs.std(ddof=1)),
        "null_q95": float(np.percentile(zs, 95)),
    }


# Critical values for the randomness score test, from 3,000 null replicates per
# cell across the design space real post-calving surveys occupy (10-60 detected
# groups, 25-150 active collars, power-law exponent 1.4-2.5). See
# calibration_study.py. The null distribution of z is centred near -0.40 with
# standard deviation 0.68 and a long right tail -- NOT N(0,1) -- and these
# cut-offs are stable to within 0.02 across that whole design space.
RANDOMNESS_CRITICAL_VALUES = {0.05: 0.94, 0.01: 2.09, 0.001: 3.70}


def calibrated_p_from_z(z: float, n_groups: int | None = None) -> dict:
    """Convert a reported score statistic into a calibrated one-sided P-value.

    Use this to re-interpret a PUBLISHED result when the underlying group table
    is not available. When you do have the group data, prefer
    :func:`bootstrap_randomness_test`, which conditions on the observed group
    sizes and is more precise.

    The asymptotic P-value that standard software reports is wrong in BOTH
    directions, with a crossover near z = 2.5:

        z = 1.5  asymptotic 0.067   calibrated 0.027   too large by 2.5x
        z = 2.0  asymptotic 0.023   calibrated 0.013   too large by 1.7x
        z = 2.5  asymptotic 0.0062  calibrated 0.0067  about right
        z = 3.0  asymptotic 0.0014  calibrated 0.0035  too SMALL by 2.6x
        z = 4.5  asymptotic 3.4e-06 calibrated 0.0004  too SMALL by 100x

    Below the crossover the published value is conservative, so surveys that
    reported "assumption met" at 0.94 < z < 1.645 in fact reject at the 5%
    level. Above it, very large z values are reported with far more confidence
    than the data support.

    Returns a dict with p_asymptotic, p_calibrated, rejects_at_05, and the
    critical values.
    """
    z = float(z)
    # Piecewise-linear interpolation on log P over the simulated grid, pooled
    # across group counts (the dependence on n_groups is within Monte Carlo
    # error, so n_groups is accepted for interface stability but not used).
    grid_z = np.array([-2.0, -1.0, 0.0, 0.94, 1.5, 2.0, 2.09, 2.5, 3.0, 3.70, 4.0, 4.5, 5.0])
    grid_p = np.array([0.985, 0.83, 0.30, 0.05, 0.0265, 0.0133, 0.010,
                       0.0067, 0.0035, 0.001, 0.0009, 0.0004, 0.0002])
    p_cal = float(np.exp(np.interp(z, grid_z, np.log(grid_p))))
    p_cal = min(max(p_cal, 1e-5), 1.0)
    return {
        "z": z,
        "p_asymptotic": float(1.0 - norm.cdf(z)),
        "p_calibrated": p_cal,
        "rejects_at_05": z > RANDOMNESS_CRITICAL_VALUES[0.05],
        "rejects_at_01": z > RANDOMNESS_CRITICAL_VALUES[0.01],
        "critical_values": dict(RANDOMNESS_CRITICAL_VALUES),
        "note": ("calibrated by simulation; use bootstrap_randomness_test() "
                 "instead when the group table is available"),
    }


def petersen_chapman(n_marked: int, n_counted: int, n_recaptured: int):
    """Chapman's bias-corrected Lincoln-Petersen estimator.

    n_marked      collars deployed and active   (n1)
    n_counted     animals counted in the survey (n2)
    n_recaptured  collars seen in the survey    (m2)

    Returns (T_hat, se).  Chapman (1951) removes most of the small-sample
    upward bias of the raw ratio estimator n1*n2/m2.
    """
    n1, n2, m2 = int(n_marked), int(n_counted), int(n_recaptured)
    if m2 <= 0:
        return np.nan, np.nan
    T = (n1 + 1) * (n2 + 1) / (m2 + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m2) * (n2 - m2) / ((m2 + 1) ** 2 * (m2 + 2))
    return float(T), float(np.sqrt(var)) if var > 0 else np.nan
