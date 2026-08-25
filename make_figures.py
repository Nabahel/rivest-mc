"""
Publication figures from the simulation results.

Rendered at 300 dpi in a print-safe palette; each figure is written as both PNG
(for review) and PDF (vector, for submission).  Categorical hues are assigned in
a fixed, colour-vision-deficiency-validated order and never cycled; the
misspecification heat map uses a diverging blue-red scale with a neutral grey at
zero bias, because bias is signed and zero is the meaningful midpoint.

Usage:  python3 make_figures.py [--results results/] [--out figures/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

# validated categorical order (blue, orange, aqua, yellow, magenta, green, ...)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e6e5e1", "#ffffff"
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
DIVERGING = LinearSegmentedColormap.from_list(
    "bias", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f3b4b4", "#e34948", "#8f1f1f"]
)

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def _grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {stem}.png / .pdf")


# --------------------------------------------------------------------------

def fig_A(df: pd.DataFrame, out: Path):
    """Diagnosis: what each implementation error costs, by number of groups."""
    labels = {"none": "Correct usage",
              "zero_collar": "Zero-collar groups passed to estimator",
              "n_found": "n set to collars found, not collars active"}
    order = ["none", "zero_collar", "n_found"]
    colors = {"none": SERIES[2], "zero_collar": SERIES[1], "n_found": SERIES[0]}

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, alpha in zip(axes, (2.14, 1.4)):
        sub = df[df.alpha == alpha]
        groups = sorted(sub.n_groups.unique())
        x = np.arange(len(groups)); w = 0.26
        for k, err in enumerate(order):
            s = sub[sub.error == err].set_index("n_groups").reindex(groups)
            vals = s.rel_bias_pct.to_numpy(dtype=float)
            ax.bar(x + (k - 1) * w, np.nan_to_num(vals), w,
                   color=colors[err], zorder=3,
                   label=labels[err] if ax is axes[0] else None)
            for xi, v in zip(x + (k - 1) * w, vals):
                if np.isnan(v):
                    # every replicate returned a boundary solution
                    ax.plot([xi], [0], marker="x", ms=7, mew=2, color=CRITICAL,
                            zorder=5)
                    ax.text(xi, 40, "all\nreplicates\nfailed", ha="center",
                            va="bottom", fontsize=6.5, color=CRITICAL,
                            fontweight="bold", linespacing=1.15)
                else:
                    # label every bar, including the near-zero ones: an
                    # invisible bar is the result, and it needs to be readable
                    up, small = v >= 0, abs(v) <= 3
                    # near-zero bars are invisible, so their labels sit close
                    # together; stagger them by series to stop them colliding
                    y = (14 + 13 * k) if small else v + (7 if up else -7)
                    ax.text(xi, y, f"{v:+.1f}", ha="center",
                            va="bottom" if (up or small) else "top",
                            fontsize=6.8,
                            color=INK2 if not small else colors[err],
                            fontweight="normal" if not small else "bold")
        ax.axhline(0, color=INK, lw=1, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_xlabel("groups in the herd")
        ax.set_title(f"$\\alpha$ = {alpha}"
                     + ("   (fitted to the survey data)" if alpha != 1.4
                        else "   (value used in the original analysis)"))
        _grid(ax)
    axes[0].set_ylabel("relative bias of $\\hat{T}$  (%)")
    axes[0].legend(loc="upper left", ncol=1)
    fig.suptitle(
        "Under true random mixing the estimator is unbiased; the reported "
        "overestimation is reproduced only by the implementation errors",
        y=1.04, fontsize=10.5, fontweight="bold", color=INK, ha="center")
    fig.text(0.5, -0.06,
             "1,000 replicates per cell. True herd size 100,000; 100 active "
             "collars; collars allocated multinomially, i.e. random mixing holds exactly.",
             ha="center", fontsize=7.5, color=MUTED)
    _save(fig, out, "fig1_diagnosis")


def fig_B(df: pd.DataFrame, out: Path):
    """Clustering: bias, coverage, and whether the score test can catch it."""
    df = df.copy()
    df["kx"] = df.kappa.replace(np.inf, 100.0)
    ns = sorted(df.n_collars.unique())

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    panels = [
        ("rel_bias_pct", "relative bias of $\\hat{T}$  (%)",
         "Clustering biases the estimate downward"),
        ("cover_log_pct", "coverage of the nominal 95% CI  (%)",
         "and the interval stops covering"),
        ("power_random_test_pct", "power of the score test  (%)",
         "the built-in test detects it, within limits"),
    ]
    for ax, (col, ylab, title) in zip(axes, panels):
        for k, n in enumerate(ns):
            s = df[df.n_collars == n].sort_values("kx")
            solid = s.degenerate_pct < 20
            ax.plot(s.kx, s[col], "-", color=SERIES[k], lw=2, zorder=3,
                    label=f"{n} collars")
            # filled markers where the estimator converged in >80% of
            # replicates; hollow where it mostly failed, so the reader can see
            # that those points summarise a surviving subset, not the design
            ax.plot(s.kx[solid], s[col][solid], "o", color=SERIES[k], ms=5,
                    mec=SURFACE, mew=1.2, zorder=4)
            ax.plot(s.kx[~solid], s[col][~solid], "o", mfc=SURFACE, ms=5,
                    mec=SERIES[k], mew=1.6, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 1, 5, 20, 100])
        ax.set_xticklabels(["0.25", "1", "5", "20", "$\\infty$"])
        ax.set_xlabel("$\\kappa$   (clustering → random mixing)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
        _grid(ax)
    axes[0].axhline(0, color=INK, lw=1, zorder=5)
    axes[1].axhline(95, color=CRITICAL, lw=1.2, ls="--", zorder=5)
    axes[1].text(0.26, 88, "nominal 95%", fontsize=7.5, color=CRITICAL)
    axes[2].axhline(80, color=MUTED, lw=1, ls=":", zorder=5)
    axes[2].text(0.26, 83, "80% power", fontsize=7.5, color=MUTED)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, -0.02))
    # the hollow-marker key goes in the middle panel, which has clear space
    axes[1].legend(
        handles=[Line2D([], [], marker="o", ls="none", mfc=SURFACE, mec=INK2,
                        mew=1.4, ms=5, label=">20% of replicates degenerate")],
        loc="upper left", bbox_to_anchor=(0.01, 0.86), fontsize=7)
    fig.suptitle(
        "Departure from random mixing, not sampling error, is what breaks the "
        "estimator — and more collars do not fix it",
        y=1.05, fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.5, -0.20,
             "1,000 replicates per cell. $\\kappa$ is the Dirichlet concentration of the "
             "collar allocation: $\\kappa=\\infty$ is exact random mixing, smaller values clump "
             "collars into fewer groups.\nThe $x$ axis is logarithmic in $\\kappa$ with "
             "$\\infty$ placed at the right; hollow markers summarise only the replicates "
             "that converged, so the apparent recovery\nat the smallest $\\kappa$ is "
             "survivorship, not improvement.",
             ha="center", fontsize=7.5, color=MUTED, linespacing=1.5)
    _save(fig, out, "fig2_clustering")


def fig_B2(df: pd.DataFrame, out: Path):
    """The reported standard error stops tracking the real one."""
    s = df[(df.n_collars == 100) & (df.degenerate_pct < 20)].copy()
    s["kx"] = s.kappa.replace(np.inf, 100.0)
    s = s.sort_values("kx")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.fill_between(s.kx, s.mean_se, s.sd_T_hat, color=SERIES[1], alpha=0.10,
                    zorder=2)
    ax.plot(s.kx, s.sd_T_hat, "-o", color=SERIES[1], lw=2, ms=5.5, mec=SURFACE,
            mew=1.2, zorder=3,
            label="actual spread of $\\hat{T}$ across replicates")
    ax.plot(s.kx, s.mean_se, "-o", color=SERIES[0], lw=2, ms=5.5, mec=SURFACE,
            mew=1.2, zorder=3,
            label="mean standard error the method reports")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "$\\infty$"])
    ax.set_xlabel("$\\kappa$   (clustering → random mixing)")
    ax.set_ylabel("caribou")
    ax.set_ylim(0, None)
    ax.set_title("Under clustering the method reports a precision it does not have",
                 fontsize=10)
    ax.legend(loc="upper right")
    _grid(ax)

    worst = s.loc[s.kx.idxmin()]
    ratio = worst.sd_T_hat / worst.mean_se
    ax.annotate(
        f"at $\\kappa$ = {worst.kappa:g} the true spread is\n"
        f"{ratio:.1f}× the reported standard error",
        xy=(worst.kx * 1.25, (worst.sd_T_hat + worst.mean_se) / 2),
        xytext=(4.2, 2100), fontsize=8, color=INK2, ha="left",
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                        connectionstyle="arc3,rad=-0.2"))
    fig.text(0.5, -0.05,
             "100 active collars, 30 groups, 1,000 replicates per cell. "
             "Cells where over 20% of replicates returned a boundary solution "
             "are omitted.",
             ha="center", fontsize=7.5, color=MUTED)
    _save(fig, out, "fig3_variance_understated")


def fig_C(df: pd.DataFrame, out: Path):
    """Survey effort, under the null and under mild clustering."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    gs = sorted(df.n_groups.unique())
    for ax, (kappa, tag) in zip(axes, [(np.inf, "random mixing holds ($\\kappa=\\infty$)"),
                                       (10.0, "mild clustering ($\\kappa=10$)")]):
        sub = df[df.kappa == kappa] if np.isfinite(kappa) else df[~np.isfinite(df.kappa)]
        for k, g in enumerate(gs):
            s = sub[sub.n_groups == g].sort_values("n_collars")
            ax.plot(s.n_collars, s.cover_log_pct, "-o", color=SERIES[k], lw=2,
                    ms=5, mec=SURFACE, mew=1.2, zorder=3, label=f"{g} groups")
        ax.axhline(95, color=CRITICAL, lw=1.2, ls="--", zorder=4)
        ax.set_ylim(0, 105)
        ax.set_xlabel("active collars")
        ax.set_title(tag, fontsize=9.5)
        _grid(ax)
    axes[0].set_ylabel("coverage of the nominal 95% CI  (%)")
    axes[0].text(30, 97, "nominal 95%", fontsize=7.5, color=CRITICAL)
    axes[0].legend(loc="lower right")
    fig.suptitle("Adding collars buys precision only while the mixing assumption holds",
                 y=1.04, fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.5, -0.06, "1,000 replicates per cell; true herd size 100,000.",
             ha="center", fontsize=7.5, color=MUTED)
    _save(fig, out, "fig4_effort")


def fig_D(df: pd.DataFrame, out: Path):
    """Cost of assuming the wrong phase-II detection model."""
    names = {"H": "Homogeneity", "I": "Independence", "T": "Threshold"}
    # one shared colour scale across both panels: per-panel scaling with a
    # single colourbar would make equal colours mean different numbers
    vmax = float(np.nanmax(np.abs(df.rel_bias_pct.to_numpy(dtype=float))))
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for ax, r in zip(axes, (0.7, 0.9)):
        sub = df[df.r == r]
        M = sub.pivot(index="true_model", columns="fitted_model",
                      values="rel_bias_pct").reindex(index=list("HIT"),
                                                     columns=list("HIT"))
        im = ax.imshow(M.values, cmap=DIVERGING, norm=norm)
        for i in range(3):
            for j in range(3):
                v = M.values[i, j]
                if np.isnan(v):
                    continue
                shade = INK if abs(v) < 0.55 * vmax else "#ffffff"
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=9, color=shade,
                        fontweight="bold" if i == j else "normal")
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               ec=INK, lw=2))
        ax.set_xticks(range(3), [names[c] for c in "HIT"])
        ax.set_yticks(range(3), [names[c] for c in "HIT"])
        ax.set_xlabel("model the analyst fits")
        ax.set_title(f"detection probability r = {r}", fontsize=9.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[1].set_yticklabels([])          # row labels belong to the left panel only
    axes[0].set_ylabel("model that actually generated detection")
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label("relative bias of $\\hat{T}$  (%)", fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("Choosing the detection model matters more than sampling error",
                 y=1.02, fontsize=10.5, fontweight="bold", color=INK)
    fig.text(0.5, -0.03,
             "Boxed cells are correctly specified. 1,000 replicates per cell; "
             "100 active collars, 30 groups, random mixing.",
             ha="center", fontsize=7.5, color=MUTED)
    _save(fig, out, "fig5_misspecification")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    print("writing figures:")
    fig_A(pd.read_csv(a.results / "study_A.csv"), a.out)
    dfB = pd.read_csv(a.results / "study_B.csv")
    fig_B(dfB, a.out); fig_B2(dfB, a.out)
    fig_C(pd.read_csv(a.results / "study_C.csv"), a.out)
    fig_D(pd.read_csv(a.results / "study_D.csv"), a.out)


if __name__ == "__main__":
    main()
