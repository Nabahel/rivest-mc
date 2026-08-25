"""
Calibrating the randomness score test for real post-calving survey sizes.

The score test of Rivest et al. (1998, sec. 5.1) refers its statistic to a
standard normal. Post-calving surveys detect 10-40 groups, and at that scale the
null distribution of z is not N(0,1): it is centred below zero, narrower in the
body, and carries a long right tail. Every published application we are aware of
-- the Boulanger et al. (2018) meta-analysis of 20 surveys, and the Alaskan
management reports -- uses the asymptotic reference distribution, so every
published P-value in this literature is on the wrong scale.

This script quantifies that. It simulates the null distribution of z across the
design space real surveys occupy, and produces:

  1. a table of the true null distribution by (groups, collars, dispersion);
  2. the actual size of the nominal 5%, 1% and 0.1% tests;
  3. a lookup that converts a published asymptotic P-value into a calibrated one,
     so existing results can be re-interpreted without re-running anyone's data.

The third output is the practically useful one: it does not require access to the
original group tables, only the reported z and the number of groups.

Usage:  python3 calibration_study.py [--reps 4000] [--out results/]
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from rivest_mc.estimator import abundance
from rivest_mc.simulate import (allocate_collars, detect_groups,
                                group_sizes_powerlaw)

warnings.filterwarnings("ignore")
SEED = 20260822

# The design space real post-calving surveys occupy, taken from the published
# record: 7-70 detected groups, 17-120 active collars.
N_GROUPS = [10, 15, 20, 25, 30, 40, 60]
N_COLLARS = [25, 50, 75, 100, 150]
ALPHAS = [1.4, 1.84, 2.5]


def null_draw(args) -> dict:
    """Null distribution of z for one design cell."""
    n_groups, n_collars, alpha, reps, seed = args
    rng = np.random.default_rng(seed)
    zs = []
    for _ in range(reps):
        sizes = group_sizes_powerlaw(100_000, n_groups, alpha, rng=rng)
        if sizes.size < 3:
            continue
        collars = allocate_collars(sizes, n_collars, kappa=np.inf, rng=rng)
        keep = detect_groups(collars, "H", r=0.95, rng=rng) & (collars >= 1)
        if keep.sum() < 3:
            continue
        try:
            res = abundance(np.column_stack([collars[keep], sizes[keep]]),
                            n=n_collars, model="H")
        except Exception:                                          # noqa: BLE001
            continue
        if np.isfinite(res.z_random):
            zs.append(res.z_random)
    zs = np.asarray(zs)
    if zs.size < 100:
        return {"n_groups": n_groups, "n_collars": n_collars, "alpha": alpha,
                "n_ok": int(zs.size)}
    # size of the nominal tests: how often the asymptotic rule rejects under H0
    return {
        "n_groups": n_groups, "n_collars": n_collars, "alpha": alpha,
        "n_ok": int(zs.size),
        "null_mean": float(zs.mean()), "null_sd": float(zs.std(ddof=1)),
        "q90": float(np.percentile(zs, 90)), "q95": float(np.percentile(zs, 95)),
        "q99": float(np.percentile(zs, 99)),
        "q999": float(np.percentile(zs, 99.9)),
        "size_at_5pct": float(np.mean(zs >= norm.ppf(0.95)) * 100),
        "size_at_1pct": float(np.mean(zs >= norm.ppf(0.99)) * 100),
        "size_at_0.1pct": float(np.mean(zs >= norm.ppf(0.999)) * 100),
        # what a reported asymptotic P really corresponds to
        "true_p_at_asym_05": float(np.mean(zs >= norm.ppf(0.95))),
        "true_p_at_asym_01": float(np.mean(zs >= norm.ppf(0.99))),
        "true_p_at_asym_001": float(np.mean(zs >= norm.ppf(0.999))),
        "z_grid": zs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    jobs = [(g, c, al, a.reps, SEED + 1000 * i)
            for i, (g, c, al) in enumerate(product(N_GROUPS, N_COLLARS, ALPHAS))]
    with ProcessPoolExecutor(max_workers=2) as ex:
        res = list(ex.map(null_draw, jobs))

    zgrids = {(r["n_groups"], r["n_collars"], r["alpha"]): r.pop("z_grid")
              for r in res if "z_grid" in r}
    D = pd.DataFrame(res)
    D.to_csv(a.out / "null_calibration.csv", index=False)

    print("NULL DISTRIBUTION OF THE SCORE STATISTIC (asymptotically N(0,1))")
    print("=" * 96)
    print("Pooled across the design space real surveys occupy:")
    print(f"  mean   {D.null_mean.mean():+.3f}   (asymptotic: 0)")
    print(f"  sd     {D.null_sd.mean():.3f}    (asymptotic: 1)")
    print(f"  97.5th percentile of z ranges {D.q95.min():+.2f} to {D.q95.max():+.2f}"
          f"   (asymptotic 95th: {norm.ppf(0.95):+.2f})")

    print("\nACTUAL SIZE OF THE NOMINAL TESTS  (%)")
    print("-" * 96)
    print(f"{'groups':>7}{'collars':>9}{'alpha':>7}"
          f"{'nominal 5%':>13}{'nominal 1%':>13}{'nominal 0.1%':>15}")
    print("-" * 96)
    for _, r in D.dropna(subset=["null_mean"]).sort_values(
            ["alpha", "n_groups", "n_collars"]).iterrows():
        print(f"{int(r.n_groups):>7}{int(r.n_collars):>9}{r.alpha:>7.2f}"
              f"{r['size_at_5pct']:>13.1f}{r['size_at_1pct']:>13.1f}"
              f"{r['size_at_0.1pct']:>15.2f}")
    print("-" * 96)
    ok = D.dropna(subset=["null_mean"])
    print(f"nominal  5%  -> actual {ok['size_at_5pct'].min():.1f}-"
          f"{ok['size_at_5pct'].max():.1f}%  (median {ok['size_at_5pct'].median():.1f})")
    print(f"nominal  1%  -> actual {ok['size_at_1pct'].min():.1f}-"
          f"{ok['size_at_1pct'].max():.1f}%  (median {ok['size_at_1pct'].median():.1f})")
    print(f"nominal 0.1% -> actual {ok['size_at_0.1pct'].min():.2f}-"
          f"{ok['size_at_0.1pct'].max():.2f}% (median {ok['size_at_0.1pct'].median():.2f})")

    # ---- the practically useful output ---------------------------------
    print("\n\nCONVERTING A PUBLISHED z INTO A CALIBRATED P-VALUE")
    print("=" * 96)
    print("Rows: number of detected groups. Columns: the z a paper reports.")
    print("Cell: calibrated one-sided P, pooled over collar numbers and dispersion.\n")
    zshow = [1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
    print(f"{'groups':>7}" + "".join(f"{'z='+str(z):>12}" for z in zshow)
          + f"{'  (asymptotic P)':>0}")
    print("-" * 96)
    lookup = []
    for g in N_GROUPS:
        pooled = np.concatenate([v for k, v in zgrids.items() if k[0] == g])
        row = f"{g:>7}"
        for z in zshow:
            p = (1 + np.sum(pooled >= z)) / (1 + pooled.size)
            row += f"{p:>12.4f}"
            lookup.append({"n_groups": g, "z": z, "p_calibrated": p,
                           "p_asymptotic": 1 - norm.cdf(z)})
        print(row)
    print("-" * 96)
    print("asymptotic".rjust(7) + "".join(f"{1-norm.cdf(z):>12.4f}" for z in zshow))
    pd.DataFrame(lookup).to_csv(a.out / "z_to_p_lookup.csv", index=False)

    print(f"\n-> {a.out/'null_calibration.csv'}, {a.out/'z_to_p_lookup.csv'}")


if __name__ == "__main__":
    main()
