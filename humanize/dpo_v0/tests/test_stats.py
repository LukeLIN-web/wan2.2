from __future__ import annotations

import warnings

import numpy as np
import pytest

from eval import stats


def test_axes_default_constants():
    assert stats.AXES_DEFAULT == ["SA", "PTV", "persistence", "inertia", "momentum"]
    assert stats.N_AXES_DEFAULT == 5
    assert stats.N_RESAMPLES_DEFAULT == 10_000
    assert stats.ALPHA_DEFAULT == 0.05
    assert stats.RNG_SEED_DEFAULT == 0


def test_bootstrap_ci_simple_distribution():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean, lo, hi = stats.bootstrap_ci(arr, n_resamples=2000, seed=0)
    assert mean == pytest.approx(3.0)
    assert lo < mean < hi
    assert lo >= 1.0
    assert hi <= 5.0
    mean2, lo2, hi2 = stats.bootstrap_ci(arr, n_resamples=2000, seed=0)
    assert (mean, lo, hi) == (mean2, lo2, hi2)


def test_bootstrap_ci_empty_and_single():
    mean, lo, hi = stats.bootstrap_ci(np.array([]))
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)
    mean, lo, hi = stats.bootstrap_ci(np.array([2.5]))
    assert mean == 2.5
    assert np.isnan(lo) and np.isnan(hi)


def test_sign_test_with_ties():
    arr = np.array([1.0, 1.0, -1.0, 0.0, 0.0])
    n_pos, n_neg, n_tie, p = stats.sign_test(arr)
    assert (n_pos, n_neg, n_tie) == (2, 1, 2)
    assert p == pytest.approx(1.0)


def test_sign_test_all_positive():
    arr = np.array([0.5, 1.0, 0.2, 0.7, 0.3])
    n_pos, n_neg, n_tie, p = stats.sign_test(arr)
    assert (n_pos, n_neg, n_tie) == (5, 0, 0)
    assert p < 0.1


def test_sign_test_all_zeros():
    arr = np.array([0.0, 0.0, 0.0])
    n_pos, n_neg, n_tie, p = stats.sign_test(arr)
    assert (n_pos, n_neg, n_tie) == (0, 0, 3)
    assert p == 1.0


def test_paired_sign_test_intersection_only():
    a = {"p1": 0.5, "p2": -0.3, "p3": 0.2, "only_a": 99.0}
    b = {"p1": 0.1, "p2": -0.5, "p3": 0.4, "only_b": -99.0}
    n_pos, n_neg, n_tie, _ = stats.paired_sign_test(a, b)
    assert n_pos + n_neg + n_tie == 3


def test_paired_sign_test_empty_intersection_raises():
    with pytest.raises(ValueError):
        stats.paired_sign_test({"x": 1.0}, {"y": 2.0})


def test_within_noise_boundary():
    assert stats.within_noise(0.05, -0.10, 0.10) is True
    assert stats.within_noise(0.10, -0.10, 0.10) is False
    assert stats.within_noise(0.15, -0.10, 0.10) is False
    assert stats.within_noise(-0.05, -0.10, 0.10) is True
    assert stats.within_noise(float("nan"), 0.0, 1.0) is False


def test_compute_delta_skips_missing_pids_and_warns():
    trained = {
        "pid1": {a: 3.0 for a in stats.AXES_DEFAULT},
        "pid2": {a: 2.0 for a in stats.AXES_DEFAULT},
        "only_trained": {a: 4.0 for a in stats.AXES_DEFAULT},
    }
    baseline = {
        "pid1": {a: 1.0 for a in stats.AXES_DEFAULT},
        "pid2": {a: 1.5 for a in stats.AXES_DEFAULT},
        "only_baseline": {a: 0.0 for a in stats.AXES_DEFAULT},
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = stats.compute_delta(trained, baseline)
    assert any("skipped" in str(w.message) for w in caught)
    for axis in stats.AXES_DEFAULT:
        assert out[axis].shape == (2,)
        assert sorted(out[axis].tolist()) == [0.5, 2.0]


def test_per_class_delta_buckets_correctly():
    pid_A = "2455740c4d45"
    pid_B = "1b1c06c5ff1c"
    pid_C = "8b8d6d0a9919"
    deltas_by_axis = {
        "SA": {pid_A: 0.5, pid_B: -0.2, pid_C: 0.0},
        "PTV": {pid_A: 0.1, pid_B: 0.3, pid_C: -0.1},
    }
    pcd = stats.per_class_delta(deltas_by_axis)
    assert set(pcd.keys()) == {"A", "B", "C"}
    assert pcd["A"]["SA"].tolist() == [0.5]
    assert pcd["B"]["SA"].tolist() == [-0.2]
    assert pcd["C"]["PTV"].tolist() == [-0.1]


def test_per_class_delta_unknown_pid_dropped():
    pcd = stats.per_class_delta({"SA": {"unknown_pid": 1.0}})
    assert pcd == {}


def test_bonferroni_alpha():
    assert stats.bonferroni_alpha(5, 7, alpha=0.05) == pytest.approx(0.05 / 35)
    assert stats.bonferroni_alpha(5, 1) == pytest.approx(0.01)
    with pytest.raises(ValueError):
        stats.bonferroni_alpha(0, 1)


def test_prompt_class_covers_42():
    assert len(stats.PROMPT_CLASS) == 42
    assert set(stats.PROMPT_CLASS.values()) == {"A", "B", "C", "D", "E", "F", "G"}


def test_render_headline_table_smoke():
    deltas = {ax: np.array([0.1, 0.2, 0.3]) for ax in stats.AXES_DEFAULT}
    md = stats.render_headline_table(deltas, n=3)
    assert "| metric |" in md
    assert "axes-avg" in md
    assert "rolling-read-only" in md  # n=3 < 42


def test_render_per_axis_table_smoke():
    deltas = {ax: np.array([0.1, -0.1, 0.0]) for ax in stats.AXES_DEFAULT}
    md = stats.render_per_axis_table(deltas)
    for ax in stats.AXES_DEFAULT:
        assert f"| {ax} |" in md


def test_render_per_class_table_smoke():
    pid_A = "2455740c4d45"
    pcd = stats.per_class_delta({"SA": {pid_A: 0.5}, "PTV": {pid_A: 0.3}})
    md = stats.render_per_class_table(pcd)
    assert "| A " in md
    assert "| B " in md  # missing classes still shown with n=0
