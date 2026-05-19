"""
tests/test_ablation_utils.py
Tests for bootstrap_ci (pure NumPy — no model, no SAE).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sae_gemma.ablations import bootstrap_ci


# ── Deterministic seed guarantees (bootstrap uses seed=0 internally) ─────────


def test_all_correct_accuracy_1():
    """All-correct array: mean accuracy == 1.0 and CI == [1.0, 1.0]."""
    correct = np.ones(100, dtype=bool)
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=200)
    assert mean == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_all_wrong_accuracy_0():
    """All-wrong array: mean accuracy == 0.0 and CI == [0.0, 0.0]."""
    correct = np.zeros(100, dtype=bool)
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=200)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_mixed_mean_close_to_true_proportion():
    """For a 70%-correct array the returned mean should be close to 0.7."""
    rng = np.random.default_rng(123)
    correct = rng.random(500) < 0.7
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=500)
    assert abs(mean - 0.7) < 0.05, f"mean={mean:.3f} too far from 0.7"


def test_ci_contains_mean():
    """CI bounds must satisfy lo <= mean <= hi for any input."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        n = rng.integers(50, 300)
        p = rng.uniform(0.1, 0.9)
        correct = rng.random(n) < p
        mean, lo, hi = bootstrap_ci(correct, n_bootstrap=300)
        assert lo <= mean + 1e-9, f"lo={lo} > mean={mean}"
        assert hi >= mean - 1e-9, f"hi={hi} < mean={mean}"


def test_ci_lo_leq_hi():
    """The lower CI bound is always <= upper CI bound."""
    correct = np.array([True, False, True, True, False])
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=500)
    assert lo <= hi


def test_return_types_are_python_float():
    """bootstrap_ci should return plain Python floats, not np.float64."""
    correct = np.array([True, True, False])
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=100)
    assert isinstance(mean, float)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_ci_width_shrinks_with_more_data():
    """Wider CI for small n, narrower CI for large n (same true proportion ~0.5)."""
    rng = np.random.default_rng(42)
    p = 0.5
    small = rng.random(30) < p
    large = rng.random(3000) < p

    _, lo_s, hi_s = bootstrap_ci(small, n_bootstrap=500)
    _, lo_l, hi_l = bootstrap_ci(large, n_bootstrap=500)

    width_small = hi_s - lo_s
    width_large = hi_l - lo_l
    assert width_small > width_large, (
        f"Expected small-n CI ({width_small:.3f}) wider than large-n CI ({width_large:.3f})"
    )


def test_n_bootstrap_affects_reproducibility():
    """bootstrap_ci is reproducible: same input produces same output on repeated calls."""
    correct = np.array([True, False, True, True, False, True])
    result1 = bootstrap_ci(correct, n_bootstrap=300)
    result2 = bootstrap_ci(correct, n_bootstrap=300)
    assert result1 == result2, "bootstrap_ci is not reproducible with seed=0"


def test_single_element_all_correct():
    """Edge case: single True element."""
    correct = np.array([True])
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=100)
    assert mean == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_single_element_all_wrong():
    """Edge case: single False element."""
    correct = np.array([False])
    mean, lo, hi = bootstrap_ci(correct, n_bootstrap=100)
    assert mean == pytest.approx(0.0)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.0)


def test_alpha_affects_ci_width():
    """Smaller alpha (wider interval) should produce a wider or equal CI than larger alpha."""
    rng = np.random.default_rng(99)
    correct = rng.random(200) < 0.6
    _, lo_wide, hi_wide = bootstrap_ci(correct, n_bootstrap=500, alpha=0.01)  # 99% CI
    _, lo_narrow, hi_narrow = bootstrap_ci(correct, n_bootstrap=500, alpha=0.10)  # 90% CI
    assert (hi_wide - lo_wide) >= (hi_narrow - lo_narrow) - 1e-9
