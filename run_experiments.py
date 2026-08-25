"""
The simulation experiments.

Four studies, each answering one question the original analysis could not:

  A. DIAGNOSIS   Do the two implementation errors reproduce the reported
                 overestimation, and which one dominates?
  B. CLUSTERING  How does the estimator degrade as collar mixing departs from
                 random, and can the built-in score test detect it?
  C. EFFORT      How many collars are needed for usable bias and coverage?
  D. MISSPEC     What does assuming the wrong detection model cost?

Everything runs on simulated data.  No field data is read, written, or needed.

Usage:  python3 run_experiments.py [--reps 1000] [--out results/]
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from rivest_mc.simulate import Scenario, run_scenario

warnings.filterwarnings("ignore")

SEED = 20260822

# Median power-law exponent fitted by the Clauset-Shalizi-Newman procedure to the
# group sizes of 12 PUBLICLY AVAILABLE post-calving surveys (CRAN caribou package
# plus the Boulanger et al. 2018 appendix). Using a fitted value rather than a
# round number matters: at alpha < 1.5 a single aggregation dominates the herd
# and the clustering effect is damped roughly threefold.
ALPHA_FIT = 2.14


def _job(args):
    sc, reps, seed, kwargs = args
    return run_scenario(sc, n_rep=reps, seed=seed, **kwargs)


def run_all(jobs, workers=None):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, jobs))


# --------------------------------------------------------------------------
# A. diagnosis of the reported overestimation
# --------------------------------------------------------------------------

def study_A(reps: int) -> pd.DataFrame:
    """Reproduce the two implementation errors and measure what each costs.

    The reported analysis showed a median +44.5% overestimate and 23% interval
    coverage under a collar-allocation process that was pure multinomial --
    i.e. exactly the random-mixing assumption the estimator is derived under,
    where it should be unbiased.  If the errors below reproduce those numbers,
    the finding is a bug, not a property of the estimator.
    """
    rows, jobs, meta = [], [], []
    for n_groups, alpha, bug in product(
        (15, 30, 60, 120), (1.4, ALPHA_FIT), ("none", "zero_collar", "n_found")
    ):
        sc = Scenario(T=100_000, n_groups=n_groups, n_collars=100,
                      alpha=alpha, kappa=np.inf, r=0.9)
        kw = {}
        if bug == "zero_collar":
            kw = dict(include_zero_collar_groups=True)
        elif bug == "n_found":
            kw = dict(n_for_estimator="found")
        jobs.append((sc, reps, SEED, kw))
        meta.append({"n_groups": n_groups, "alpha": alpha, "error": bug})

    for m, res in zip(meta, run_all(jobs)):
        rows.append({**m, **{k: res.get(k) for k in (
            "rel_bias_pct", "median_rel_bias_pct", "rel_rmse_pct",
            "cover_log_pct", "cover_normal_pct", "degenerate_pct", "n_ok")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# B. departure from random mixing -- the scientific question
# --------------------------------------------------------------------------

def study_B(reps: int) -> pd.DataFrame:
    """Sweep collar clustering from perfect random mixing to severe.

    kappa is the Dirichlet concentration: infinity is the multinomial (the
    assumption holds exactly), and smaller values mean collars clump into
    fewer groups than group sizes warrant.  Reported alongside the estimator's
    performance is the POWER of the package's built-in score test, which tells
    a practitioner whether they could have detected the violation from their
    own data.
    """
    kappas = [np.inf, 50, 20, 10, 5, 2, 1, 0.5, 0.25]
    jobs, meta = [], []
    for kappa, n_collars in product(kappas, (50, 100, 200, 400)):
        sc = Scenario(T=100_000, n_groups=30, n_collars=n_collars,
                      alpha=ALPHA_FIT, kappa=kappa, r=0.9)
        jobs.append((sc, reps, SEED, {}))
        meta.append({"kappa": kappa, "n_collars": n_collars})

    rows = []
    for m, res in zip(meta, run_all(jobs)):
        rows.append({**m, **{k: res.get(k) for k in (
            "rel_bias_pct", "median_rel_bias_pct", "rel_rmse_pct",
            "cover_log_pct", "power_random_test_pct", "degenerate_pct",
            "mean_se", "sd_T_hat", "n_ok")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# C. survey effort
# --------------------------------------------------------------------------

def study_C(reps: int) -> pd.DataFrame:
    """How many collars, and how many groups, are needed?

    The practical question a manager asks.  Run under mild clustering
    (kappa=10) as well as the null, because the null flatters the estimator.
    """
    jobs, meta = [], []
    for n_collars, n_groups, kappa in product(
        (25, 50, 75, 100, 150, 200, 300, 400), (15, 30, 60), (np.inf, 10)
    ):
        sc = Scenario(T=100_000, n_groups=n_groups, n_collars=n_collars,
                      alpha=ALPHA_FIT, kappa=kappa, r=0.9)
        jobs.append((sc, reps, SEED, {}))
        meta.append({"n_collars": n_collars, "n_groups": n_groups, "kappa": kappa})

    rows = []
    for m, res in zip(meta, run_all(jobs)):
        rows.append({**m, **{k: res.get(k) for k in (
            "rel_bias_pct", "rel_rmse_pct", "cover_log_pct",
            "degenerate_pct", "power_random_test_pct", "n_ok")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D. detection-model misspecification
# --------------------------------------------------------------------------

def study_D(reps: int) -> pd.DataFrame:
    """Simulate under one detection model, estimate under another.

    The original analysis used only the homogeneity model while describing
    three.  This quantifies what that choice costs when it is wrong.
    """
    jobs, meta = [], []
    for true_m, fit_m, r in product(("H", "I", "T"), ("H", "I", "T"), (0.7, 0.9)):
        sc = Scenario(T=100_000, n_groups=30, n_collars=100, alpha=ALPHA_FIT,
                      kappa=np.inf, detect_model=true_m, fit_model=fit_m,
                      r=r, B=2)
        jobs.append((sc, reps, SEED, {}))
        meta.append({"true_model": true_m, "fitted_model": fit_m, "r": r})

    rows = []
    for m, res in zip(meta, run_all(jobs)):
        rows.append({**m, **{k: res.get(k) for k in (
            "rel_bias_pct", "rel_rmse_pct", "cover_log_pct",
            "degenerate_pct", "n_ok")}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--only", default=None, help="comma-separated: A,B,C,D")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    which = args.only.split(",") if args.only else ["A", "B", "C", "D"]
    studies = {"A": study_A, "B": study_B, "C": study_C, "D": study_D}
    titles = {
        "A": "A. Diagnosis of the reported overestimation",
        "B": "B. Departure from random mixing",
        "C": "C. Survey effort",
        "D": "D. Detection-model misspecification",
    }

    for key in which:
        print(f"\n{'='*78}\n{titles[key]}   ({args.reps} replicates per cell)\n{'='*78}")
        df = studies[key](args.reps)
        path = args.out / f"study_{key}.csv"
        df.to_csv(path, index=False)
        with pd.option_context("display.width", 200, "display.max_columns", 40,
                               "display.float_format", lambda v: f"{v:,.2f}"):
            print(df.to_string(index=False))
        print(f"-> {path}")


if __name__ == "__main__":
    main()
