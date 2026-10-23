"""
KL-Divergence Exponential Tilting Module.

Implements the KL-minimization / exponential tilting problem from the paper:

    find η_λ such that q_{t,λ} = exp(η_λ^T r_t) / Σ_s exp(η_λ^T r_s)
    and Σ_t q_{t,λ} r_t = μ_λ

Equivalent to minimizing D_KL(Q || P) subject to E_Q[r] = μ_λ,
or maximizing the convex dual:
    ψ(η) = log[(1/T) Σ_t exp(η^T r_t)] - η^T μ_λ

Uses log-sum-exp stabilization for numerical stability.

# Custom — Context7 found no library equivalent (paper KL-tilting methodology)
"""

import logging
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def log_sum_exp(log_weights: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp.

    Parameters
    ----------
    log_weights : np.ndarray  Shape (T,).

    Returns
    -------
    float  log(Σ_t exp(log_weights[t])).
    """
    c = log_weights.max()
    return float(c + np.log(np.sum(np.exp(log_weights - c))))


def compute_tilt_weights(
    eta: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized tilted probabilities q_{t} = exp(η^T r_t) / Σ_s exp(η^T r_s).

    Parameters
    ----------
    eta : np.ndarray    Shape (n,) — tilting parameter vector.
    returns : np.ndarray  Shape (T, n) — historical return matrix.

    Returns
    -------
    np.ndarray  Shape (T,) — tilted probability weights, positive, summing to 1.
    """
    log_w = returns @ eta  # (T,)
    log_w -= log_w.max()   # numerical stabilization
    w = np.exp(log_w)
    return w / w.sum()


def dual_objective(
    eta: np.ndarray,
    returns: np.ndarray,
    mu_target: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Convex dual objective and gradient for the tilting problem.

        ψ(η) = log[(1/T) Σ_t exp(η^T r_t)] - η^T μ_target

    Minimizing ψ(η) yields the exponential tilt η* satisfying E_Q[r] = μ_target.

    Parameters
    ----------
    eta : np.ndarray      Shape (n,) — current tilting parameter.
    returns : np.ndarray  Shape (T, n) — historical returns.
    mu_target : np.ndarray  Shape (n,) — target mean μ_λ.

    Returns
    -------
    tuple of (objective_value, gradient)
    """
    T = returns.shape[0]
    log_w = returns @ eta  # (T,)
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    # ψ(η) = log[(1/T) Σ exp(η^T r_t)] - η^T μ
    lse = log_sum_exp(returns @ eta)
    obj = lse - np.log(T) - float(eta @ mu_target)

    # ∇ψ(η) = E_Q[r] - μ_target
    grad = (w @ returns) - mu_target  # (n,)

    return obj, grad


def solve_exponential_tilt(
    returns: np.ndarray,
    mu_target: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 2000,
    eta_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Solve for the exponential tilt η that matches target mean μ_target.

    Minimizes the convex dual ψ(η) = log[(1/T) Σ_t exp(η^T r_t)] - η^T μ_target
    using L-BFGS-B from scipy.

    Parameters
    ----------
    returns : np.ndarray      Shape (T, n) — historical daily return matrix.
    mu_target : np.ndarray    Shape (n,) — target mean vector μ_λ.
    tol : float               Convergence tolerance.
    max_iter : int            Maximum iterations for L-BFGS-B.
    eta_init : np.ndarray or None
        Initial η. Defaults to zeros.

    Returns
    -------
    tuple of (eta_opt, q_weights, converged)
        eta_opt : np.ndarray  Shape (n,) — optimal tilting parameter.
        q_weights : np.ndarray  Shape (T,) — tilted probabilities.
        converged : bool      Whether optimizer converged.
    """
    n = returns.shape[1]
    if eta_init is None:
        eta_init = np.zeros(n)

    result = minimize(
        fun=dual_objective,
        x0=eta_init,
        args=(returns, mu_target),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol**2, "gtol": tol},
    )

    eta_opt = result.x
    q_weights = compute_tilt_weights(eta_opt, returns)
    converged = result.success

    if not converged:
        logger.warning("Exponential tilt did not converge: %s", result.message)

    # Validate moment matching
    achieved_mean = q_weights @ returns
    moment_error = np.max(np.abs(achieved_mean - mu_target))
    logger.debug("Moment matching error: %.2e", moment_error)

    return eta_opt, q_weights, converged


def compute_tilted_covariance(
    returns: np.ndarray,
    q_weights: np.ndarray,
    mu_tilted: np.ndarray,
) -> np.ndarray:
    """
    Compute the tilted covariance Σ_λ = Σ_t q_{t,λ}(r_t - μ_λ)(r_t - μ_λ)^T.

    Parameters
    ----------
    returns : np.ndarray    Shape (T, n) — historical returns.
    q_weights : np.ndarray  Shape (T,) — tilted probabilities summing to 1.
    mu_tilted : np.ndarray  Shape (n,) — tilted mean μ_λ.

    Returns
    -------
    np.ndarray  Shape (n, n) — tilted covariance matrix Σ_λ.
    """
    centered = returns - mu_tilted[np.newaxis, :]  # (T, n)
    # Σ_λ = (centered * q_weights[:, None])^T @ centered
    Sigma_tilt = (centered * q_weights[:, np.newaxis]).T @ centered
    return Sigma_tilt


def compute_all_lambda_tilt(
    returns_hist: np.ndarray,
    mu_hist: np.ndarray,
    mu_view: np.ndarray,
    lambda_values: list[float] | np.ndarray,
    tol: float = 1e-9,
) -> list[dict]:
    """
    Compute exponential tilting results for all λ values.

    For each λ:
        μ_λ = (1-λ) μ_hist + λ μ_view
        Solve for q_{t,λ} via dual optimization
        Compute Σ_λ from tilted weights

    Parameters
    ----------
    returns_hist : np.ndarray   Shape (T, n) — historical returns.
    mu_hist : np.ndarray        Shape (n,) — historical mean.
    mu_view : np.ndarray        Shape (n,) — expert view mean (realized future mean).
    lambda_values : array-like  Values of λ ∈ [0,1].
    tol : float                 Convergence tolerance.

    Returns
    -------
    list of dicts, one per λ, with keys:
        'lambda', 'mu_lambda', 'eta', 'q_weights', 'Sigma_lambda',
        'converged', 'moment_error'
    """
    T, n = returns_hist.shape
    results = []
    eta_prev = None  # warm-start from previous λ

    for lam in lambda_values:
        mu_lam = (1.0 - lam) * mu_hist + lam * mu_view

        # Check feasibility: mu_lam should be in convex hull of returns
        r_min = returns_hist.min(axis=0)
        r_max = returns_hist.max(axis=0)
        in_hull = np.all(mu_lam >= r_min - 1e-10) and np.all(mu_lam <= r_max + 1e-10)
        if not in_hull:
            logger.warning("λ=%.2f: μ_λ may be outside convex hull of returns.", lam)

        eta_opt, q_weights, converged = solve_exponential_tilt(
            returns_hist, mu_lam, tol=tol, eta_init=eta_prev
        )
        eta_prev = eta_opt.copy()

        Sigma_lam = compute_tilted_covariance(returns_hist, q_weights, mu_lam)

        achieved_mean = q_weights @ returns_hist
        moment_error = float(np.max(np.abs(achieved_mean - mu_lam)))

        logger.info(
            "λ=%.2f: converged=%s, moment_err=%.2e, q_sum=%.8f",
            lam, converged, moment_error, q_weights.sum(),
        )

        results.append({
            "lambda": lam,
            "mu_lambda": mu_lam,
            "eta": eta_opt,
            "q_weights": q_weights,
            "Sigma_lambda": Sigma_lam,
            "converged": converged,
            "moment_error": moment_error,
        })

    return results


def exponential_tilt(
    returns: np.ndarray,
    mu_target: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 2000,
    eta_init: np.ndarray | None = None,
) -> dict:
    """
    Convenience wrapper around solve_exponential_tilt returning a dict.

    Parameters
    ----------
    returns : np.ndarray      Shape (T, n).
    mu_target : np.ndarray    Shape (n,) — target mean μ_λ.
    tol, max_iter, eta_init : see solve_exponential_tilt.

    Returns
    -------
    dict with 'eta', 'q_weights', 'converged'.
    """
    eta, q, conv = solve_exponential_tilt(returns, mu_target, tol=tol,
                                          max_iter=max_iter, eta_init=eta_init)
    return {"eta": eta, "q_weights": q, "converged": conv}


def dual_objective_val(
    eta: np.ndarray,
    returns: np.ndarray,
    mu_target: np.ndarray,
) -> float:
    """
    Return only the scalar value of the convex dual objective (no gradient).

    Useful for numerical convexity checks.

    Parameters
    ----------
    eta : np.ndarray      Shape (n,).
    returns : np.ndarray  Shape (T, n).
    mu_target : np.ndarray  Shape (n,).

    Returns
    -------
    float
    """
    val, _ = dual_objective(eta, returns, mu_target)
    return val


def validate_tilt_result(
    q_weights: np.ndarray,
    returns: np.ndarray,
    mu_target: np.ndarray,
    tol: float = 1e-6,
) -> dict:
    """
    Validate exponential tilt result: check positivity, normalization, moment matching.

    Parameters
    ----------
    q_weights : np.ndarray    Shape (T,) — tilted probabilities.
    returns : np.ndarray      Shape (T, n) — historical returns.
    mu_target : np.ndarray    Shape (n,) — target mean.
    tol : float               Tolerance for moment matching check.

    Returns
    -------
    dict with validation results.
    """
    positive = bool(np.all(q_weights > 0))
    normalized = bool(abs(q_weights.sum() - 1.0) < 1e-8)
    achieved_mean = q_weights @ returns
    moment_error = float(np.max(np.abs(achieved_mean - mu_target)))
    moment_matched = moment_error < tol

    return {
        "positive": positive,
        "normalized": normalized,
        "moment_matched": moment_matched,
        "moment_error": moment_error,
        "q_min": float(q_weights.min()),
        "q_max": float(q_weights.max()),
    }
