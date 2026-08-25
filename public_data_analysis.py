"""
The real-data component of the paper, built entirely on public data.

Every herd-year here ships with the CRAN `caribou` package and is freely
redistributable, so this analysis needs no permissions and can be published as
it stands.  Restricted survey data, if and when it is cleared, extends this
table; it does not enable it.

Two results come out:

  1. The random-mixing assumption is testable, and it holds in most published
     post-calving surveys but fails decisively in one.
  2. The asymptotic p-value of that test cannot be trusted at the group counts
     these surveys actually have.  Report a parametric bootstrap instead.

Usage:  python3 public_data_analysis.py [--boot 5000] [--out results/]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rivest_mc.estimator import abundance, bootstrap_randomness_test

warnings.filterwarnings("ignore")

# Active-collar counts and herd identities taken VERBATIM from each dataset's
# help page in the caribou package (man/*.Rd). These are easy to guess wrong:
# BEH00 is the Bluenose-EAST herd, not Beverly, and LRH01 had 120 collars
# active of which only 23 (19%) were photographed.
DATASETS = {
    "GRH93": (92,  "George River",   "Quebec/Labrador", 1993),
    "GRH01": (109, "George River",   "Quebec",          2001),
    "GRH10": (71,  "George River",   "Quebec",          2010),
    "BEH00": (33,  "Bluenose-East",  "NWT",             2000),
    "LRH01": (120, "Leaf River",     "Quebec",          2001),
    "WAH11": (97,  "Western Arctic", "Alaska",          2011),
}
# Patterson et al. (2004) also computed BEH00 assuming 30 active collars, on the
# grounds that 3 collared caribou were never located after collaring.
ALT_N = {"BEH00": 30}

SEED = 20260822


def load(name: str, data_dir: Path) -> np.ndarray:
    """Group matrix for one dataset: column 0 collars, column 1 group size."""
    return np.loadtxt(data_dir / f"{name}.csv", delimiter=",", skiprows=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--data", type=Path, default=Path("/tmp/caribou_data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for key, (n, herd, region, year) in DATASETS.items():
        mat = load(key, a.data)
        res = abundance(mat, n=n, model="H")
        lo, hi = res.ci(method="log")
        bt = bootstrap_randomness_test(mat, n=n, model="H",
                                       n_boot=a.boot, seed=SEED)
        rows.append({
            "dataset": key, "herd": herd, "region": region, "year": year,
            "groups_detected": res.mp,
            "collars_active": n, "collars_found": res.xt,
            "animals_counted": res.gnt,
            "r_hat": round(res.rr, 4),
            "T_hat": round(res.T_hat),
            "se_T_hat": round(res.se_T_hat, 1),
            "ci_lo": round(lo), "ci_hi": round(hi),
            "z_random": round(bt["z_obs"], 3),
            "p_asymptotic": bt["p_asymptotic"],
            "p_bootstrap": bt["p_bootstrap"],
            "null_mean_z": round(bt["null_mean"], 3),
            "null_sd_z": round(bt["null_sd"], 3),
            "mixing_rejected": bt["p_bootstrap"] < 0.05,
        })

    df = pd.DataFrame(rows)
    df.to_csv(a.out / "public_herd_years.csv", index=False)

    print("Rivest estimates and tests of random mixing, published post-calving surveys")
    print("=" * 100)
    hdr = (f"{'herd':<15}{'year':>6}{'grps':>6}{'n':>5}{'T_hat':>10}"
           f"{'95% CI (lognormal)':>24}{'z':>7}{'p (boot)':>11}")
    print(hdr); print("-" * 100)
    for r in rows:
        ci = f"{r['ci_lo']:,} - {r['ci_hi']:,}"
        star = " *" if r["mixing_rejected"] else ""
        print(f"{r['herd']:<15}{r['year']:>6}{r['groups_detected']:>6}"
              f"{r['collars_active']:>5}{r['T_hat']:>10,}{ci:>24}"
              f"{r['z_random']:>7.2f}{r['p_bootstrap']:>11.1e}{star}")
    print("-" * 100)
    nrej = int(df.mixing_rejected.sum())
    print(f"* random mixing rejected at p < 0.05 in {nrej} of {len(df)} surveys "
          f"({a.boot} bootstrap replicates each)\n")

    print("Asymptotic vs bootstrap p-values for the score test")
    print("-" * 74)
    print(f"{'dataset':<10}{'groups':>8}{'z':>8}{'p asympt.':>13}{'p bootstrap':>14}{'ratio':>12}")
    for r in rows:
        ratio = r["p_asymptotic"] / r["p_bootstrap"] if r["p_bootstrap"] else np.nan
        print(f"{r['dataset']:<10}{r['groups_detected']:>8}{r['z_random']:>8.2f}"
              f"{r['p_asymptotic']:>13.2e}{r['p_bootstrap']:>14.2e}{ratio:>12.3g}")
    print("-" * 74)
    print("The asymptotic p-value refers z to N(0,1). The bootstrap columns show\n"
          "what the null distribution actually is at these group counts: the ratio\n"
          "is how far the asymptotic value misstates the evidence.")
    print(f"\n-> {a.out / 'public_herd_years.csv'}")


if __name__ == "__main__":
    main()
