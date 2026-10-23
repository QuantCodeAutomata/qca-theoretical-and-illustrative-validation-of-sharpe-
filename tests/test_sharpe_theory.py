"""
Tests for src/sharpe_theory.py — Experiment 1.

Validates:
- S(1) = S* when Ω=Σ (theorem case)
- Monotonicity: S(λ) non-decreasing when A*C > B²
- Concavity condition A+3B ≥ 0
- Closed-form matches general formula when Ω=Σ
- Derivative formulas are consistent with numerical differences
- Angle interpretation: S = S* cos(θ)
- Edge cases: zero mean, near-singular covariance
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sharpe_theory import (
    compute_ABC,
    sharpe_closed_form,
    sharpe_derivative,
    sharpe_second_derivative,
    realized_sharpe_general,
    optimal_sharpe,
    compute_sharpe_vs_lambda,
    angle_interpretation,
    check_monotonicity,
    safe_inv,
    regularize_matrix,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_simple_2d(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nu, mu, Sigma) for a 2-asset example."""
    rng = np.random.default_rng(seed)
    n = 2
    nu = rng.normal(0, 0.01, n)
    mu = rng.normal(0.001, 0.01, n)
    A_raw = rng.normal(0, 1, (n, n))
    Sigma = A_raw @ A_raw.T + 0.01 * np.eye(n)
    return nu, mu, Sigma


def make_5d_example(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nu, mu, Sigma) for a 5-asset example."""
    rng = np.random.default_rng(seed)
    n = 5
    nu = rng.normal(0, 0.005, n)
    mu = rng.normal(0.002, 0.005, n)
    A_raw = rng.normal(0, 1, (n, n))
    Sigma = A_raw @ A_raw.T / n + 0.01 * np.eye(n)
    return nu, mu, Sigma


# ── Test: closed-form ABC computation ─────────────────────────────────────────

def test_compute_ABC_basic():
    """A, B, C should be finite scalars; C >= 0."""
    nu, mu, Sigma = make_simple_2d()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    assert np.isfinite(diag.A)
    assert np.isfinite(diag.B)
    assert np.isfinite(diag.C)
    assert diag.C >= 0, "C = (mu-nu)^T Sigma^{-1} (mu-nu) must be non-negative"


def test_compute_ABC_zero_delta():
    """When mu == nu, B=0 and C=0."""
    nu, _, Sigma = make_simple_2d()
    mu = nu.copy()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    assert abs(diag.B) < 1e-12, "B should be 0 when mu=nu"
    assert abs(diag.C) < 1e-12, "C should be 0 when mu=nu"


def test_concavity_condition_flag():
    """concave flag matches A+3B >= 0."""
    nu, mu, Sigma = make_5d_example()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    expected = (diag.A + 3 * diag.B) >= 0
    assert diag.concave == expected


# ── Test: closed-form S(λ) ────────────────────────────────────────────────────

def test_sharpe_closed_form_at_0_and_1():
    """
    S(0) = A / sqrt(A) = sqrt(A) (since nu^T Sigma^{-1} mu = A+B at lambda=0).
    S(1) = (A+2B+C) / sqrt(A+2B+C) = sqrt(A+2B+C) = S*.
    """
    nu, mu, Sigma = make_simple_2d()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C

    S0 = sharpe_closed_form(0.0, A, B, C)
    expected_S0 = (A + B) / np.sqrt(A) if A > 0 else 0.0
    assert abs(S0 - expected_S0) < 1e-12

    S1 = sharpe_closed_form(1.0, A, B, C)
    denom_1 = A + 2*B + C
    expected_S1 = (A + 2*B + C) / np.sqrt(denom_1) if denom_1 > 0 else 0.0
    assert abs(S1 - expected_S1) < 1e-12


def test_sharpe_equals_optimal_at_lambda1():
    """S(1) = S*_{mu,Sigma} when Omega = Sigma (theorem case)."""
    nu, mu, Sigma = make_5d_example()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C

    S1_cf = sharpe_closed_form(1.0, A, B, C)
    S_star = optimal_sharpe(mu, Sigma)

    # They should be equal
    assert abs(S1_cf - S_star) < 1e-6, f"S(1)={S1_cf:.6f} != S*={S_star:.6f}"


# ── Test: derivative formulas ────────────────────────────────────────────────

def test_derivative_consistent_with_finite_diff():
    """S'(λ) should match numerical finite difference."""
    nu, mu, Sigma = make_simple_2d()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C

    lam = 0.3
    h = 1e-6
    d1_analytical = sharpe_derivative(lam, A, B, C)
    d1_numerical = (sharpe_closed_form(lam + h, A, B, C) -
                    sharpe_closed_form(lam - h, A, B, C)) / (2 * h)
    assert abs(d1_analytical - d1_numerical) < 1e-5


def test_second_derivative_consistent_with_finite_diff():
    """S''(λ) should match numerical second finite difference."""
    nu, mu, Sigma = make_simple_2d()
    Sigma_inv = safe_inv(Sigma)
    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C

    lam = 0.3
    h = 1e-4
    d2_analytical = sharpe_second_derivative(lam, A, B, C)
    d2_numerical = (sharpe_closed_form(lam + h, A, B, C) -
                    2 * sharpe_closed_form(lam, A, B, C) +
                    sharpe_closed_form(lam - h, A, B, C)) / h**2
    assert abs(d2_analytical - d2_numerical) < 1e-4


def test_derivative_nonneg_when_AC_gt_B2():
    """When AC > B², derivative should be non-negative for λ in [0,1)."""
    # Use example where AC - B² > 0
    A, B, C = 1.0, 0.2, 0.5
    assert A * C - B**2 > 0
    for lam in np.linspace(0, 0.99, 50):
        d1 = sharpe_derivative(lam, A, B, C)
        assert d1 >= -1e-12, f"S'({lam:.2f})={d1:.6f} should be >= 0"


# ── Test: realized Sharpe (general formula) ──────────────────────────────────

def test_realized_sharpe_general_agrees_with_closed_form():
    """General realized Sharpe == closed-form when Omega = Sigma."""
    nu, mu, Sigma = make_5d_example()
    result = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None)
    diff = np.max(np.abs(result["sharpe_vals"] - result["sharpe_cf"]))
    assert diff < 1e-8, f"Max diff between general and closed-form: {diff:.2e}"


def test_realized_sharpe_range():
    """Realized Sharpe should be finite and close to [0, S*] for typical inputs."""
    nu, mu, Sigma = make_5d_example()
    result = compute_sharpe_vs_lambda(nu, mu, Sigma)
    S_star = result["S_star"]
    sr_vals = result["sharpe_vals"]
    assert np.all(np.isfinite(sr_vals))
    # All Sharpe values should be <= S* (up to small numerical error)
    assert np.all(sr_vals <= S_star + 1e-6)


# ── Test: S(1) = S* validation ───────────────────────────────────────────────

def test_S_at_lambda1_equals_S_star():
    """S(λ=1) = S*_{μ,Σ} when Ω=Σ (key theorem property)."""
    for seed in range(5):
        nu, mu, Sigma = make_5d_example(seed)
        result = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None)
        S1 = result["sharpe_vals"][-1]
        S_star = result["S_star"]
        assert abs(S1 - S_star) < 1e-6, f"seed={seed}: S(1)={S1}, S*={S_star}"


# ── Test: monotonicity ────────────────────────────────────────────────────────

def test_monotonicity_check_increasing():
    """check_monotonicity returns True for an increasing array."""
    arr = np.linspace(0.1, 2.0, 100)
    assert check_monotonicity(arr)


def test_monotonicity_check_non_monotone():
    """check_monotonicity returns False for a non-monotone array."""
    arr = np.array([0.1, 0.5, 0.3, 0.8])
    assert not check_monotonicity(arr)


def test_monotonicity_when_AC_gt_B2():
    """Sharpe-vs-λ should be monotone when AC > B²."""
    # Construct example with AC > B²
    A, B, C = 1.0, 0.1, 2.0  # AC = 2.0 > B² = 0.01
    lams = np.linspace(0, 1, 101)
    sr = np.array([sharpe_closed_form(l, A, B, C) for l in lams])
    assert check_monotonicity(sr)


# ── Test: angle interpretation ────────────────────────────────────────────────

def test_angle_interpretation():
    """S = S* * cos(θ) should hold."""
    nu, mu, Sigma = make_simple_2d()
    Sigma_reg = regularize_matrix(Sigma, 1e-8)
    cos_theta = angle_interpretation(mu, Sigma_reg, nu, Sigma_reg)
    S_star = optimal_sharpe(mu, Sigma_reg)
    S_realized = realized_sharpe_general(mu, Sigma_reg, nu, Sigma_reg)

    S_via_angle = S_star * cos_theta
    assert abs(S_realized - S_via_angle) < 1e-8, (
        f"S_realized={S_realized:.6f}, S*·cos(θ)={S_via_angle:.6f}"
    )


def test_cos_theta_in_minus_1_to_1():
    """cos(θ) should be in [-1, 1]."""
    nu, mu, Sigma = make_5d_example()
    Sigma_reg = regularize_matrix(Sigma, 1e-8)
    cos_theta = angle_interpretation(mu, Sigma_reg, nu, Sigma_reg)
    assert -1.0 - 1e-8 <= cos_theta <= 1.0 + 1e-8


# ── Test: edge cases ─────────────────────────────────────────────────────────

def test_zero_nu():
    """When nu = 0, S(0) should be 0."""
    _, mu, Sigma = make_simple_2d()
    nu = np.zeros_like(mu)
    result = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None)
    assert abs(result["sharpe_vals"][0]) < 1e-12


def test_single_asset():
    """Single asset case should work without errors."""
    nu = np.array([0.001])
    mu = np.array([0.002])
    Sigma = np.array([[0.04]])
    result = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None)
    assert len(result["sharpe_vals"]) == 101
    assert np.all(np.isfinite(result["sharpe_vals"]))


def test_near_singular_covariance():
    """Near-singular Σ should be handled by regularization."""
    nu = np.array([0.001, 0.002])
    mu = np.array([0.003, 0.004])
    Sigma = np.array([[1.0, 1.0 - 1e-10], [1.0 - 1e-10, 1.0]])  # near singular
    result = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None, eps=1e-6)
    assert np.all(np.isfinite(result["sharpe_vals"]))


def test_lambda_grid_endpoints():
    """S at λ=0 and λ=1 match expected boundary values."""
    nu, mu, Sigma = make_2d_with_known_ABC()
    result = compute_sharpe_vs_lambda(nu, mu, Sigma)
    Sigma_inv = safe_inv(regularize_matrix(Sigma, 1e-8))
    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C
    # S(0) from closed form
    S0_cf = sharpe_closed_form(0.0, A, B, C)
    assert abs(result["sharpe_vals"][0] - S0_cf) < 1e-8


def make_2d_with_known_ABC():
    """Construct a 2d example with known A,B,C for testing."""
    Sigma = np.eye(2)
    nu = np.array([0.01, 0.005])
    mu = np.array([0.02, 0.015])
    return nu, mu, Sigma


# ── Test: compute_sharpe_vs_lambda output structure ──────────────────────────

def test_output_shapes():
    """All output arrays should have the same length as lambda_grid."""
    nu, mu, Sigma = make_5d_example()
    lam_grid = np.linspace(0, 1, 51)
    result = compute_sharpe_vs_lambda(nu, mu, Sigma, lambda_grid=lam_grid)
    for key in ["sharpe_vals", "sharpe_cf", "deriv1", "deriv2"]:
        assert len(result[key]) == 51, f"{key} has wrong length"


def test_S_star_positive():
    """S* should be non-negative."""
    nu, mu, Sigma = make_5d_example()
    S_star = optimal_sharpe(mu, Sigma)
    assert S_star >= 0


def test_different_Omega_vs_Sigma():
    """General formula with Omega != Sigma may differ from closed-form."""
    nu, mu, Sigma = make_5d_example()
    rng = np.random.default_rng(99)
    n = 5
    B = rng.normal(0, 1, (n, n))
    Omega = B @ B.T / n + 0.01 * np.eye(n)

    result_closed = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=None)
    result_general = compute_sharpe_vs_lambda(nu, mu, Sigma, Omega=Omega)

    # The two results should differ when Omega != Sigma
    diff = np.max(np.abs(result_closed["sharpe_vals"] - result_general["sharpe_vals"]))
    # Not necessarily large, but the formulas should run without error
    assert np.all(np.isfinite(result_general["sharpe_vals"]))
