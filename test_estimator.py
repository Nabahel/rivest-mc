"""Regression tests. Run with: python3 -m pytest test_estimator.py -q"""
import numpy as np
import pytest
from rivest_mc.estimator import (abundance, bootstrap_randomness_test,
                                 calibrated_p_from_z, petersen_chapman)
from rivest_mc.simulate import Scenario, run_scenario

WAH11 = np.array([[1,1498],[1,12730],[14,83619],[1,3587],[2,3701],[12,43607],
                  [57,135556],[5,18858],[2,7593],[1,12]], float)

def test_matches_caribou_package_on_WAH11():
    """caribou::abundance(WAH11, n=97) gives T.hat 324953.2, se 10540.31."""
    r = abundance(WAH11, n=97, model="H")
    assert abs(r.T_hat - 324953.2) < 1.0
    assert abs(r.se_T_hat - 10540.31) < 0.5

def test_reproduces_boulanger_cape_bathurst_2005():
    """Boulanger et al. (2018) publish T-hat = 3,566 for this survey."""
    cb = np.array([[10,492],[3,174],[2,7],[2,228],[2,138],[1,22],[1,9],[1,73],
                   [1,1],[1,173],[1,6],[1,421],[1,453],[1,15],[1,1]], float)
    assert abs(abundance(cb, n=32, model="H").T_hat - 3566) < 5

def test_zero_collar_groups_are_rejected():
    with pytest.raises(ValueError, match="zero collars"):
        abundance(np.array([[0,100],[2,500]], float), n=10)

def test_collars_found_cannot_exceed_active():
    with pytest.raises(ValueError, match="exceed active"):
        abundance(WAH11, n=50, model="H")

def test_log_interval_is_never_negative():
    lo, _ = abundance(np.array([[1,120],[1,90],[2,300]], float), n=15).ci(method="log")
    assert lo > 0

def test_unbiased_under_random_mixing():
    """The core validity check: <2% bias and near-nominal coverage at the null."""
    r = run_scenario(Scenario(T=100_000, n_groups=30, n_collars=100,
                              alpha=2.14, kappa=np.inf), n_rep=300, seed=1)
    assert abs(r["rel_bias_pct"]) < 2.0
    assert 88 < r["cover_log_pct"] < 99

def test_clustering_biases_downward():
    r = run_scenario(Scenario(T=100_000, n_groups=30, n_collars=100,
                              alpha=2.14, kappa=5), n_rep=300, seed=1)
    assert r["rel_bias_pct"] < -15

def test_calibration_crossover():
    """Asymptotic P too large below z~2.5, too small above."""
    assert calibrated_p_from_z(1.5)["p_calibrated"] < calibrated_p_from_z(1.5)["p_asymptotic"]
    assert calibrated_p_from_z(4.5)["p_calibrated"] > calibrated_p_from_z(4.5)["p_asymptotic"]
    assert calibrated_p_from_z(1.0)["rejects_at_05"]
    assert not calibrated_p_from_z(0.5)["rejects_at_05"]

def test_bootstrap_agrees_with_lookup_on_WAH11():
    bt = bootstrap_randomness_test(WAH11, n=97, n_boot=800, seed=1)
    assert bt["p_bootstrap"] < 0.01
    assert bt["p_bootstrap"] > bt["p_asymptotic"]   # asymptotic overstates in the tail

def test_chapman():
    t, _ = petersen_chapman(79, 13755, 59)
    assert abs(t - 18340) < 2
