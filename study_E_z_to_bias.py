"""Study E: what does an observed score statistic imply about the bias?

The four published studies answer "how bad is the bias at a given kappa?".  A
practitioner never observes kappa.  They observe z.  This study inverts the
relationship: across survey configurations resembling real post-calving surveys,
it records (z, relative error of T-hat) for every replicate, then reports the
distribution of the error conditional on the z the survey actually returned.

Output: results/study_E_pairs.csv  (one row per replicate)
        results/study_E_bins.csv   (error distribution by z band)

Usage:  python3 study_E_z_to_bias.py [--reps 400]
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from rivest_mc.simulate import Scenario, simulate_once

warnings.filterwarnings("ignore")

SEED = 20261022
ALPHA_FIT = 2.14

# Survey configurations resembling the published post-calving record: the six
# public surveys detected 10-28 collared groups and held 33-120 active collars.
GROUPS = (10, 15, 20, 30)
COLLARS = (50, 100, 150)
# Severe cells (kappa < 2) produce z values far beyond anything in the
# published record (observed range about -1.1 to +4.5) and are excluded so
# that the simulated distribution of z resembles the observed one.
KAPPAS = (np.inf, 50, 30, 20, 15, 10, 7, 5, 3, 2)

# z bands.  0.94 is the calibrated 5% critical value; 2.09 the 1%; 3.70 the 0.1%.
BANDS = [(-np.inf, 0.94), (0.94, 2.09), (2.09, 3.70), (3.70, np.inf)]
BAND_LABELS = ["z < 0.94 (not rejected at 5%)", "0.94 <= z < 2.09 (5% level)",
               "2.09 <= z < 3.70 (1% level)", "z >= 3.70 (0.1% level)"]


def _cell(args):
    n_groups, n_collars, kappa, reps, seed = args
    sc = Scenario(T=100_000, n_groups=n_groups, n_collars=n_collars,
                  alpha=ALPHA_FIT, kappa=kappa, r=0.9)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        out = simulate_once(sc, rng)
        if not out.get("ok") or not out.get("converged"):
            continue
        z = out.get("z_random")
        if z is None or not np.isfinite(z):
            continue
        rows.append({
            "n_groups": n_groups, "n_collars": n_collars, "kappa": kappa,
            "z": float(z),
            "rel_err_pct": 100.0 * (out["T_hat"] - out["T_true"]) / out["T_true"],
            "cover_log": out["cover_log"],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(g, c, k, args.reps, SEED + i)
            for i, (g, c, k) in enumerate(product(GROUPS, COLLARS, KAPPAS))]
    print(f"{len(jobs)} cells x {args.reps} replicates")

    with ProcessPoolExecutor() as ex:
        rows = [r for part in ex.map(_cell, jobs) for r in part]

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "study_E_pairs.csv", index=False)
    print(f"{len(df):,} usable replicates -> {args.out/'study_E_pairs.csv'}")

    recs = []
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        sub = df[(df.z >= lo) & (df.z < hi)]
        if sub.empty:
            continue
        recs.append({
            "band": label,
            "n": len(sub),
            "share_pct": 100.0 * len(sub) / len(df),
            "median_err_pct": sub.rel_err_pct.median(),
            "q25_err_pct": sub.rel_err_pct.quantile(0.25),
            "q75_err_pct": sub.rel_err_pct.quantile(0.75),
            "q05_err_pct": sub.rel_err_pct.quantile(0.05),
            "q95_err_pct": sub.rel_err_pct.quantile(0.95),
            "coverage_pct": 100.0 * sub.cover_log.mean(),
        })
    bins = pd.DataFrame(recs)
    bins.to_csv(args.out / "study_E_bins.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.1f}"):
        print()
        print(bins.to_string(index=False))


if __name__ == "__main__":
    main()
