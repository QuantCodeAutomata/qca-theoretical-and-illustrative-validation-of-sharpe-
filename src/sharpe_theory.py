"""
Sharpe Ratio Theory Module — Experiment 1.

Implements the closed-form realized Sharpe ratio under Markowitz optimization
with misspecified mean forecasts, following the paper's theoretical framework.

The core result: given forecast mean ν and true mean μ, the interpolated forecast
    ν_λ = (1-λ)ν + λμ,  λ ∈ [0,1]
yields realized Sharpe (with Ω=Σ):
    S(λ) = (A + B + λ(B+C)) / sqrt(A + 2λB + λ²C)

where:
    A = ν^T Σ^{-1} ν
    B = ν^T Σ^{-1} (μ - ν)
    C = (μ-ν)^T Σ^{-1} (μ-ν)

Sufficient condition for global concavity on [0,1]: A + 3B ≥ 0.

# Custom — Context7 found no library equivalent (paper Eq. theoretical framework)
"""

import numpy as np
from typing import NamedTuple


# ──────────────────────────────────────────────────────────────────────────────
# Matrix helpers
# ──────────────────────────────────────────────────────────────────────────────

def regularize_matrix(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Add small diagonal regularization to ensure invertibility.

    Parameters
    ----------
    M : np.ndarray  Shape (n, n)
    eps : float     Regularization strength.

    Returns
    -------
    np.ndarray  M + eps * I
    """
    return M + eps * np.eye(M.shape[0])


def safe_inv(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute matrix inverse with regularization fallback.

    Parameters
    ----------
    M : np.ndarray  Shape (n, n)
    eps : float     Regularization if M is near-singular.

    Returns
    -------
    np.ndarray  M^{-1}
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.inv(regularize_matrix(M, eps))


# ──────────────────────────────────────────────────────────────────────────────
# Core scalar quantities A, B, C
# ──────────────────────────────────────────────────────────────────────────────

class SharpeDiagnostics(NamedTuple):
    """Container for the scalar quantities driving S(λ)."""
    A: float    # ν^T Σ^{-1} ν
    B: float    # ν^T Σ^{-1} (μ - ν)
    C: float    # (μ-ν)^T Σ^{-1} (μ-ν)
    concave: bool  # A + 3B ≥ 0


def compute_ABC(
    nu: np.ndarray,
    mu: np.ndarray,
    Sigma_inv: np.ndarray,
) -> SharpeDiagnostics:
    """
    Compute the scalar quantities A, B, C used in the closed-form S(λ).

    Parameters
    ----------
    nu : np.ndarray      Shape (n,) — forecast mean vector ν.
    mu : np.ndarray      Shape (n,) — true (realized) mean vector μ.
    Sigma_inv : np.ndarray  Shape (n, n) — inverse of Σ.

    Returns
    -------
    SharpeDiagnostics
    """
    delta = mu - nu
    A = float(nu @ Sigma_inv @ nu)
    B = float(nu @ Sigma_inv @ delta)
    C = float(delta @ Sigma_inv @ delta)
    concave = (A + 3 * B) >= 0
    return SharpeDiagnostics(A=A, B=B, C=C, concave=concave)


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe ratio formulas
# ──────────────────────────────────────────────────────────────────────────────

def realized_sharpe_general(
    mu: np.ndarray,
    Sigma: np.ndarray,
    nu: np.ndarray,
    Omega: np.ndarray,
) -> float:
    """
    Realized Sharpe ratio under misspecified mean and covariance (general case).

        S_{μ,Σ}(ν,Ω) = (μ^T Ω^{-1} ν) / sqrt(ν^T Ω^{-1} Σ Ω^{-1} ν)

    The unconstrained Markowitz portfolio proportional to Ω^{-1} ν is used.

    Parameters
    ----------
    mu : np.ndarray      Shape (n,) — true mean μ.
    Sigma : np.ndarray   Shape (n, n) — true covariance Σ.
    nu : np.ndarray      Shape (n,) — forecast mean ν.
    Omega : np.ndarray   Shape (n, n) — forecast covariance Ω.

    Returns
    -------
    float  Realized Sharpe ratio.
    """
    Omega_inv = safe_inv(Omega)
    numerator = float(mu @ Omega_inv @ nu)
    denom_sq = float(nu @ Omega_inv @ Sigma @ Omega_inv @ nu)
    if denom_sq <= 0:
        return 0.0
    return numerator / np.sqrt(denom_sq)


def optimal_sharpe(mu: np.ndarray, Sigma: np.ndarray) -> float:
    """
    True maximum attainable Sharpe ratio.

        S*_{μ,Σ} = sqrt(μ^T Σ^{-1} μ)

    Parameters
    ----------
    mu : np.ndarray      Shape (n,) — true mean μ.
    Sigma : np.ndarray   Shape (n, n) — true covariance Σ.

    Returns
    -------
    float  Maximum attainable Sharpe.
    """
    Sigma_inv = safe_inv(Sigma)
    val = float(mu @ Sigma_inv @ mu)
    return np.sqrt(max(val, 0.0))


def sharpe_closed_form(lam: float, A: float, B: float, C: float) -> float:
    """
    Closed-form S(λ) = (A + B + λ(B+C)) / sqrt(A + 2λB + λ²C).

    Valid when Ω = Σ (forecast covariance equals true covariance).

    Parameters
    ----------
    lam : float   Interpolation parameter λ ∈ [0,1].
    A, B, C : float  Scalar diagnostics from compute_ABC.

    Returns
    -------
    float  Realized Sharpe at λ.
    """
    numer = A + B + lam * (B + C)
    denom_sq = A + 2 * lam * B + lam**2 * C
    if denom_sq <= 0:
        return 0.0
    return numer / np.sqrt(denom_sq)


def sharpe_derivative(lam: float, A: float, B: float, C: float) -> float:
    """
    First derivative S'(λ) = ((1-λ)(AC - B²)) / (A + 2λB + λ²C)^{3/2}.

    Parameters
    ----------
    lam : float   λ value.
    A, B, C : float  Scalar diagnostics.

    Returns
    -------
    float  S'(λ).
    """
    denom_sq = A + 2 * lam * B + lam**2 * C
    if denom_sq <= 0:
        return 0.0
    numer = (1 - lam) * (A * C - B**2)
    return numer / (denom_sq**1.5)


def sharpe_second_derivative(lam: float, A: float, B: float, C: float) -> float:
    """
    Second derivative S''(λ) = -((AC-B²) * H(λ)) / (A + 2λB + λ²C)^{5/2}
    where H(λ) = A + 3B + (3C - B)λ - 2Cλ².

    Parameters
    ----------
    lam : float   λ value.
    A, B, C : float  Scalar diagnostics.

    Returns
    -------
    float  S''(λ).
    """
    denom_sq = A + 2 * lam * B + lam**2 * C
    if denom_sq <= 0:
        return 0.0
    H = A + 3 * B + (3 * C - B) * lam - 2 * C * lam**2
    numer = -(A * C - B**2) * H
    return numer / (denom_sq**2.5)


# ──────────────────────────────────────────────────────────────────────────────
# Interpolation grid computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_sharpe_vs_lambda(
    nu: np.ndarray,
    mu: np.ndarray,
    Sigma: np.ndarray,
    Omega: np.ndarray | None = None,
    lambda_grid: np.ndarray | None = None,
    eps: float = 1e-8,
) -> dict:
    """
    Compute realized Sharpe S(λ) over a dense grid λ ∈ [0,1].

    If Omega is None or Omega == Sigma, uses the closed-form expression.
    Otherwise uses the general misspecification formula.

    Parameters
    ----------
    nu : np.ndarray      Forecast mean vector ν, shape (n,).
    mu : np.ndarray      True mean vector μ, shape (n,).
    Sigma : np.ndarray   True covariance Σ, shape (n, n).
    Omega : np.ndarray or None
        Forecast covariance Ω. If None, defaults to Σ (simplified theorem case).
    lambda_grid : np.ndarray or None
        Grid of λ values. Defaults to np.linspace(0, 1, 101).
    eps : float          Regularization for matrix inversion.

    Returns
    -------
    dict with keys:
        'lambda_grid'  : np.ndarray
        'sharpe_vals'  : np.ndarray — realized Sharpe at each λ (general)
        'sharpe_cf'    : np.ndarray — closed-form S(λ) (only if Omega == Sigma)
        'deriv1'       : np.ndarray — S'(λ) values
        'deriv2'       : np.ndarray — S''(λ) values
        'S_star'       : float — optimal Sharpe S*_{μ,Σ}
        'diagnostics'  : SharpeDiagnostics
        'use_closed_form' : bool
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.0, 1.0, 101)

    Sigma_reg = regularize_matrix(Sigma, eps)
    Sigma_inv = safe_inv(Sigma_reg)

    use_closed_form = Omega is None
    if Omega is None:
        Omega_use = Sigma_reg
    else:
        Omega_use = regularize_matrix(Omega, eps)

    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C
    S_star = optimal_sharpe(mu, Sigma_reg)

    sharpe_vals = np.zeros(len(lambda_grid))
    sharpe_cf = np.zeros(len(lambda_grid))
    deriv1 = np.zeros(len(lambda_grid))
    deriv2 = np.zeros(len(lambda_grid))

    for i, lam in enumerate(lambda_grid):
        nu_lam = (1 - lam) * nu + lam * mu
        sharpe_vals[i] = realized_sharpe_general(mu, Sigma_reg, nu_lam, Omega_use)
        sharpe_cf[i] = sharpe_closed_form(lam, A, B, C)
        deriv1[i] = sharpe_derivative(lam, A, B, C)
        deriv2[i] = sharpe_second_derivative(lam, A, B, C)

    return {
        "lambda_grid": lambda_grid,
        "sharpe_vals": sharpe_vals,
        "sharpe_cf": sharpe_cf,
        "deriv1": deriv1,
        "deriv2": deriv2,
        "S_star": S_star,
        "diagnostics": diag,
        "use_closed_form": use_closed_form,
    }


def angle_interpretation(
    mu: np.ndarray,
    Sigma: np.ndarray,
    nu: np.ndarray,
    Omega: np.ndarray,
) -> float:
    """
    Compute cos θ_{μ,Σ}(ν,Ω) = (μ^T Ω^{-1} ν) / (S*_{μ,Σ} * sqrt(ν^T Ω^{-1} Σ Ω^{-1} ν)).

    Verifies S_{μ,Σ}(ν,Ω) = S*_{μ,Σ} cos θ_{μ,Σ}(ν,Ω).

    Parameters
    ----------
    mu, Sigma, nu, Omega : as in realized_sharpe_general.

    Returns
    -------
    float  cos θ value.
    """
    Sigma_inv = safe_inv(Sigma)
    Omega_inv = safe_inv(Omega)
    S_star = np.sqrt(max(float(mu @ Sigma_inv @ mu), 0.0))
    denom_nu = np.sqrt(max(float(nu @ Omega_inv @ Sigma @ Omega_inv @ nu), 0.0))
    numerator = float(mu @ Omega_inv @ nu)
    if S_star == 0 or denom_nu == 0:
        return 0.0
    return numerator / (S_star * denom_nu)


def check_monotonicity(sharpe_vals: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if the Sharpe curve is (approximately) monotonically non-decreasing.

    Parameters
    ----------
    sharpe_vals : np.ndarray  Sharpe values at sorted λ grid.
    tol : float               Tolerance for small negative differences.

    Returns
    -------
    bool  True if monotonically non-decreasing.
    """
    diffs = np.diff(sharpe_vals)
    return bool(np.all(diffs >= -tol))
