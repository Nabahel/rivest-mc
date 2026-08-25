# Rivest post-calving estimator — simulation study

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22099960.svg)](https://doi.org/10.5281/zenodo.22099960)

An independent Python implementation of the Rivest post-calving abundance
estimator, validated against the CRAN `caribou` package, plus the Monte Carlo
machinery to measure its bias, RMSE, and interval coverage under controlled
departures from its assumptions.

**No field data is used anywhere in this repository.** Every number is either
simulated or comes from the example datasets published in the `caribou` package.
The restricted survey data can be analysed with `rivest_mc.estimator.abundance()`
without any of it entering this codebase.

## Why this exists

A Monte Carlo study needs three things the original analysis did not have:
replication (bias and coverage are properties of a distribution, not of one
draw), a known truth (so error is measurable), and a dial that turns the
assumption of interest on and off (so failure can be attributed). This package
supplies all three.

## Layout

```
rivest_mc/estimator.py    the estimator: H/I/T detection models, variance,
                          log-scale intervals, the randomness score test
rivest_mc/simulate.py     group-size generators, collar allocation with a
                          clustering parameter, detection, the replicate loop
validate_vs_R.py          cross-validation gate against caribou::abundance()
run_experiments.py        the four simulation studies (A-D)
public_data_analysis.py   the real-data component, on public data only
make_figures.py           publication figures (PNG + vector PDF)
results/                  CSV output, one file per study
figures/                  rendered figures
```

## Requirements

```bash
pip install -r requirements.txt     # python >= 3.10
```

## Tests

```bash
python3 -m pytest test_estimator.py -q     # 10 tests, ~5 s
```

The two that matter most are external validations: `test_matches_caribou_package_on_WAH11`
checks T̂ and its standard error against `caribou::abundance()`, and
`test_reproduces_boulanger_cape_bathurst_2005` reproduces the estimate published
in Boulanger et al. (2018) for that survey.

`validate_vs_R.py` additionally needs R with the `caribou` package installed.
If CRAN is unreachable, the package source mirrors to GitHub:

```bash
git clone --depth 1 https://github.com/cran/caribou
R CMD INSTALL -l ~/Rlib caribou
```

## Run it

```bash
python3 -m pytest test_estimator.py -q     # fast regression check
python3 validate_vs_R.py                   # full gate against the R package
python3 run_experiments.py --reps 1000     # ~2 minutes on two cores
python3 calibration_study.py --reps 3000   # ~2 minutes
python3 public_data_analysis.py            # ~40 s
python3 make_figures.py
```

## Citing

See `CITATION.cff`. Released under the MIT licence.

> Rex-Oneal, N. 2026. *rivest-mc: simulation and calibration tools for the Rivest
> post-calving caribou abundance estimator.* Version 1.0.0. Zenodo.
> https://doi.org/10.5281/zenodo.22099960

## The validation gate

`validate_vs_R.py` runs both implementations over all six datasets shipped with
`caribou`, for all three detection models, and compares T̂, its standard error,
the detection parameter, and the score statistic.

Current status: **agreement to 1.4e-12** across all 18 dataset x model
combinations.

One caveat worth knowing about, surfaced by the comparison: `caribou::abundance()`
solves both of its estimating equations with `optimize()` — golden-section
minimisation of `|f|` at R's default tolerance — rather than by root finding.
For the independence model this leaves a relative error of order 1e-4 in the
published estimates (e.g. GRH93 returns 602,396.5 where the root is 602,388.7).
Re-solving the identical equations in R with `uniroot()` reproduces the Python
values exactly, so this is the package's optimiser tolerance and not a difference
in method. It is immaterial for field reporting and material for a simulation
study running thousands of replicates.

## Correct use of the estimator

Two errors dominate everything else, and both are easy to make:

1. **`mat` must contain only detected groups holding at least one collar.**
   Groups with zero collars lie outside the support of the conditional
   likelihood. Each one included adds roughly `T/(r·n)` to the estimating
   equation, so T̂ inflates with the number of such groups: +15% at 30 groups and
   +77% at 60, and at 120 every replicate hits the boundary (study A). `abundance()` in this
   package raises on zero-collar rows rather than silently accepting them.
2. **`n` is the number of collars ACTIVE in the herd, not the number found.**
   Passing the number found drives r̂ to 1, and T̂ collapses toward the raw
   count — a roughly −9% bias with coverage near 45%.

## What the studies found

| Study | Question | Headline |
|---|---|---|
| A | Do implementation errors explain the reported overestimation? | Yes. Correct usage gives +0.1% to +0.8% bias with 91–95% coverage. The zero-collar error alone reproduces the reported magnitude. |
| B | What does collar clustering cost? | Severe **under**estimation: −11% at κ=50, −28% at κ=10, −55% at κ=1, with coverage falling from 94% to 7%. The reported standard error does not grow to match, so precision is overstated. |
| C | How many collars are needed? | Under random mixing, 50 suffice for near-nominal coverage. Under mild clustering (κ=10), 400 collars still leave −20% bias and 22% coverage. **Collars do not buy your way out of clustering.** |
| D | What does the wrong detection model cost? | Up to +57% bias. The threshold and independence models are the dangerous ones to over-apply; homogeneity is markedly robust (worst case −6.4%). |

## Two findings from the public herd-years

`public_data_analysis.py` runs the estimator and the randomness test over all six
real post-calving surveys published with the `caribou` package. **No permissions
are needed for any of it.**

**1. The assumption usually holds — and when it fails, it fails hard.** Random
mixing is not rejected in five of the six surveys (George River 1993/2001/2010,
Bluenose-East 2000, Leaf River 2001). It is rejected decisively in Western Arctic 2011
(z = 4.45, bootstrap p = 6.0e-04), where one group of 135,556 animals holds 57 of
the 96 collars found while a group of 12 animals holds one. That the test usually
passes is what makes it worth running: a practitioner cannot tell which case they
are in without it.

**2. Do not use the asymptotic p-value.** The score test refers z to N(0,1), but
these surveys detect 10-30 groups, and at those counts the null distribution of z
is centred near -0.4, has standard deviation about 0.7, and carries a long right
tail. The asymptotic value is therefore too large in the body (by a factor of
1.1-2.5 across these datasets) and far too small in the tail: for WAH11 it reports
4.2e-06 where the bootstrap gives 6.0e-04, overstating the evidence by roughly
140x. Use `bootstrap_randomness_test()`, which conditions on the observed group
sizes and never invokes the asymptotics.

| dataset | groups | z | p asymptotic | p bootstrap | ratio |
|---|---|---|---|---|---|
| GRH93 | 28 | -0.13 | 5.5e-01 | 3.4e-01 | 1.61 |
| GRH01 | 27 | +0.49 | 3.1e-01 | 1.2e-01 | 2.55 |
| GRH10 | 13 | -0.94 | 8.3e-01 | 7.6e-01 | 1.09 |
| BEH00 | 16 | -0.12 | 5.5e-01 | 2.9e-01 | 1.86 |
| LRH01 | 17 | -1.10 | 8.6e-01 | 9.5e-01 | 0.91 |
| WAH11 | 10 | +4.45 | 4.2e-06 | 6.0e-04 | **0.007** |

## The practical recommendation

Run the score test, and bootstrap its p-value. `abundance()` already returns the
statistic, study B shows 80-93% power against the clustering levels that produce a
20%+ bias once 100-200 collars are active (weaker against mild clustering), and
`bootstrap_randomness_test()` calibrates it for the group counts real surveys
have. An estimate published without it is an estimate whose central assumption
was never checked.
