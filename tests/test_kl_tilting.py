"""
Tests for src/kl_tilting.py — KL-divergence based exponential tilting.

Validates:
- Tilted weights are positive and normalized
- Moment matching: E_Q[r] ≈ μ_λ to tight tolerance
- λ=0 recovers historical mean
- λ=1 reaches view mean (within tolerance)
- Tilted covariance is positive semi-definite
- Dual objective is convex (numerical check)
- Edge cases: λ=0 gives uniform weights, extreme values
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.kl_tilting import (
    exponential_tilt,
    compute_tilted_covariance,
    compute_all_lambda_tilt,
    validate_tilt_result,
    dual_objective,
    dual_objective_val,
    log_sum_exp,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_small_dataset(T: int = 100, n: int = 3, seed: int = 42) -> np.ndarray:
    """Return T×n daily return matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, (T, n))


def make_dataset_with_known_means(T: int = 200, n: int = 4, seed: int = 0):
    """Return (returns, mu_hist, mu_view) with controlled means."""
    rng = np.random.default_rng(seed)
    R = rng.normal(0.001, 0.01, (T, n))
    mu_hist = R.mean(axis=0)
    # Make mu_view slightly different
    mu_view = mu_hist + rng.normal(0.002, 0.001, n)
    return R, mu_hist, mu_view


# ── Test: log_sum_exp numerical stability ─────────────────────────────────────

def test_log_sum_exp_basic():
    """log_sum_exp should match naive computation for small values."""
    x = np.array([1.0, 2.0, 3.0])
    result = log_sum_exp(x)
    expected = np.log(np.sum(np.exp(x)))
    assert abs(result - expected) < 1e-12


def test_log_sum_exp_large_values():
    """log_sum_exp should handle large values without overflow."""
    x = np.array([1000.0, 1001.0, 999.0])
    result = log_sum_exp(x)
    assert np.isfinite(result)
    # Should be approximately 1001 + log(1 + exp(-1) + exp(-2))
    expected = 1001 + np.log(1 + np.exp(-1) + np.exp(-2))
    assert abs(result - expected) < 1e-8


# ── Test: exponential_tilt ────────────────────────────────────────────────────

def test_tilt_lambda0_uniform_weights():
    """At λ=0, target mean = historical mean → tilted weights ≈ uniform."""
    R, mu_hist, mu_view = make_dataset_with_known_means()
    result = exponential_tilt(R, mu_hist, tol=1e-10)
    q = result["q_weights"]
    T = len(R)
    # Weights should be approximately 1/T when target = historical mean
    assert np.allclose(q, 1.0 / T, atol=1e-6), (
        f"max deviation from uniform: {np.max(np.abs(q - 1/T)):.2e}"
    )


def test_tilt_weights_positive():
    """Tilted weights must all be strictly positive."""
    R, mu_hist, mu_view = make_dataset_with_known_means()
    lam = 0.5
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam, tol=1e-9)
    assert np.all(result["q_weights"] > 0), "All weights must be positive"


def test_tilt_weights_normalized():
    """Tilted weights must sum to 1."""
    R, mu_hist, mu_view = make_dataset_with_known_means()
    for lam in [0.2, 0.5, 0.8, 1.0]:
        mu_lam = (1 - lam) * mu_hist + lam * mu_view
        result = exponential_tilt(R, mu_lam, tol=1e-9)
        assert abs(result["q_weights"].sum() - 1.0) < 1e-10, (
            f"λ={lam}: weights sum to {result['q_weights'].sum()}"
        )


def test_tilt_moment_matching():
    """E_Q[r] ≈ μ_λ to tight tolerance."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=500, n=5)
    for lam in [0.1, 0.3, 0.5, 0.7, 1.0]:
        mu_lam = (1 - lam) * mu_hist + lam * mu_view
        result = exponential_tilt(R, mu_lam, tol=1e-10)
        q = result["q_weights"]
        mu_realized = (q[:, None] * R).sum(axis=0)
        err = np.max(np.abs(mu_realized - mu_lam))
        assert err < 1e-7, f"λ={lam}: moment matching error {err:.2e}"


def test_tilt_lambda1_reaches_view():
    """At λ=1, tilted mean should equal μ_view."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=300, n=3)
    result = exponential_tilt(R, mu_view, tol=1e-10)
    q = result["q_weights"]
    mu_realized = (q[:, None] * R).sum(axis=0)
    err = np.max(np.abs(mu_realized - mu_view))
    assert err < 1e-7, f"λ=1: error from μ_view = {err:.2e}"


# ── Test: compute_tilted_covariance ──────────────────────────────────────────

def test_tilted_covariance_is_psd():
    """Tilted covariance should be positive semi-definite."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=200, n=4)
    lam = 0.5
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam)
    q = result["q_weights"]
    Sigma = compute_tilted_covariance(R, q, mu_lam)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert np.all(eigvals >= -1e-10), f"Tilted cov not PSD: min eigval={eigvals.min():.4e}"


def test_tilted_covariance_shape():
    """Tilted covariance should have shape (n, n)."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=100, n=5)
    lam = 0.3
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam)
    q = result["q_weights"]
    Sigma = compute_tilted_covariance(R, q, mu_lam)
    assert Sigma.shape == (5, 5)


def test_tilted_covariance_symmetric():
    """Tilted covariance should be symmetric."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=150, n=4)
    lam = 0.6
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam)
    q = result["q_weights"]
    Sigma = compute_tilted_covariance(R, q, mu_lam)
    assert np.allclose(Sigma, Sigma.T, atol=1e-12)


# ── Test: compute_all_lambda_tilt ─────────────────────────────────────────────

def test_compute_all_lambda_tilt_length():
    """Should return one result per λ value."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=200, n=3)
    lams = [0.0, 0.5, 1.0]
    results = compute_all_lambda_tilt(R, mu_hist, mu_view, lams)
    assert len(results) == 3


def test_compute_all_lambda_tilt_correct_mu():
    """Each result's μ_λ should match (1-λ)μ_hist + λμ_view."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=200, n=3)
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = compute_all_lambda_tilt(R, mu_hist, mu_view, lams)
    for r in results:
        lam = r["lambda"]
        expected_mu = (1 - lam) * mu_hist + lam * mu_view
        assert np.allclose(r["mu_lambda"], expected_mu, atol=1e-12)


def test_compute_all_lambda_tilt_moment_matching():
    """All λ values should satisfy moment matching."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=300, n=3)
    lams = [0.0, 0.3, 0.6, 1.0]
    results = compute_all_lambda_tilt(R, mu_hist, mu_view, lams, tol=1e-9)
    for r in results:
        val = validate_tilt_result(r["q_weights"], R, r["mu_lambda"])
        assert val["moment_matched"], (
            f"λ={r['lambda']}: moment error = {val['moment_error']:.2e}"
        )


# ── Test: validate_tilt_result ────────────────────────────────────────────────

def test_validate_tilt_result_uniform():
    """Uniform weights over historical mean should pass all checks."""
    R = make_small_dataset(T=100, n=3)
    mu_hist = R.mean(axis=0)
    T = len(R)
    q = np.ones(T) / T
    val = validate_tilt_result(q, R, mu_hist, tol=1e-6)
    assert val["positive"]
    assert val["normalized"]
    assert val["moment_matched"]


def test_validate_negative_weights_fails():
    """Negative weights should fail the positivity check."""
    T = 10
    q = np.ones(T) / T
    q[0] = -0.01
    q[1] += 0.01  # renormalize
    R = np.random.normal(0, 1, (T, 2))
    mu = R.mean(axis=0)
    val = validate_tilt_result(q, R, mu)
    assert not val["positive"]


def test_validate_unnormalized_fails():
    """Unnormalized weights should fail the normalization check."""
    T = 20
    q = np.ones(T) / T * 1.5  # wrong sum
    R = np.random.normal(0, 1, (T, 3))
    mu = R.mean(axis=0)
    val = validate_tilt_result(q, R, mu)
    assert not val["normalized"]


# ── Test: dual_objective ──────────────────────────────────────────────────────

def test_dual_objective_finite():
    """Dual objective should return finite value."""
    R = make_small_dataset(T=50, n=2)
    mu_hist = R.mean(axis=0)
    mu_view = mu_hist + 0.001
    mu_target = 0.5 * mu_hist + 0.5 * mu_view
    eta = np.zeros(2)
    # dual_objective returns (value, gradient); use dual_objective_val for scalar
    val = dual_objective_val(eta, R, mu_target)
    assert np.isfinite(val)


def test_dual_objective_returns_tuple():
    """dual_objective should return (scalar, gradient array)."""
    R = make_small_dataset(T=50, n=2)
    mu_hist = R.mean(axis=0)
    mu_target = mu_hist + 0.001
    eta = np.zeros(2)
    result = dual_objective(eta, R, mu_target)
    assert isinstance(result, tuple) and len(result) == 2
    val, grad = result
    assert np.isfinite(val)
    assert grad.shape == (2,)


def test_dual_objective_convex_numerically():
    """
    Dual objective psi(eta) should be convex.
    Check: psi((a+b)/2) <= (psi(a) + psi(b)) / 2
    """
    R = make_small_dataset(T=100, n=2)
    mu_hist = R.mean(axis=0)
    mu_view = mu_hist + 0.005
    mu_target = 0.5 * mu_hist + 0.5 * mu_view

    rng = np.random.default_rng(7)
    for _ in range(10):
        eta_a = rng.normal(0, 0.1, 2)
        eta_b = rng.normal(0, 0.1, 2)
        eta_mid = (eta_a + eta_b) / 2
        psi_a = dual_objective_val(eta_a, R, mu_target)
        psi_b = dual_objective_val(eta_b, R, mu_target)
        psi_mid = dual_objective_val(eta_mid, R, mu_target)
        assert psi_mid <= (psi_a + psi_b) / 2 + 1e-10, (
            f"Convexity violated: psi(mid)={psi_mid:.6f} > avg={((psi_a+psi_b)/2):.6f}"
        )


# ── Test: edge cases ─────────────────────────────────────────────────────────

def test_single_observation():
    """Single observation case: weights must be [1.0]."""
    R = np.array([[0.01, 0.02]])
    mu_target = R[0]  # exact mean
    result = exponential_tilt(R, mu_target)
    assert abs(result["q_weights"][0] - 1.0) < 1e-8


def test_large_T_small_n():
    """Should handle T >> n without issues."""
    R = make_small_dataset(T=2000, n=3)
    mu_hist = R.mean(axis=0)
    mu_view = mu_hist + 0.001
    mu_lam = 0.5 * mu_hist + 0.5 * mu_view
    result = exponential_tilt(R, mu_lam)
    assert result["converged"]
    q = result["q_weights"]
    assert np.all(q > 0)
    assert abs(q.sum() - 1.0) < 1e-10


def test_tilt_extreme_lambda():
    """λ very close to 1 should still converge."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=500, n=3)
    lam = 0.99
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam, tol=1e-8, max_iter=200)
    q = result["q_weights"]
    assert np.all(np.isfinite(q))
    assert abs(q.sum() - 1.0) < 1e-8


def test_likelihood_ratio_weights():
    """Likelihood ratio ℓ_λ = T * q_{t,λ} should sum to T."""
    R, mu_hist, mu_view = make_dataset_with_known_means(T=200, n=4)
    lam = 0.4
    mu_lam = (1 - lam) * mu_hist + lam * mu_view
    result = exponential_tilt(R, mu_lam)
    T = len(R)
    ell = T * result["q_weights"]
    assert abs(ell.sum() - T) < 1e-8
