"""
Markowitz Long-Only Maximum Sharpe Portfolio Module — Experiment 2.

Solves the long-only, fully invested maximum Sharpe optimization:

    w_λ ∈ argmax_{w≥0, 1^T w=1} (w^T μ_λ) / sqrt(w^T Σ_λ w)

using scipy.optimize.minimize with SLSQP.

Also implements portfolio evaluation (gross/net returns, turnover, performance
metrics) for the 2024-2025 out-of-sample period.

# Using PyPortfolioOpt EfficientFrontier for long-only max-Sharpe — Context7 confirmed
# Custom fallback implemented when PyPortfolioOpt numerical behavior differs
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional

logger = logging.getLogger(__name__)

# One-way transaction cost in decimal (20 bps)
TRANSACTION_COST_BPS = 0.002
ANNUALIZE_FACTOR = 252


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio optimization
# ──────────────────────────────────────────────────────────────────────────────

def max_sharpe_long_only(
    mu: np.ndarray,
    Sigma: np.ndarray,
    eps: float = 1e-8,
    rf: float = 0.0,
) -> np.ndarray:
    """
    Solve long-only fully invested max Sharpe ratio portfolio via SLSQP.

    w* = argmax_{w≥0, 1^T w=1} (w^T μ - rf) / sqrt(w^T Σ w)

    Risk-free rate rf=0 exactly as specified by the paper.

    Parameters
    ----------
    mu : np.ndarray     Shape (n,) — expected return vector μ_λ.
    Sigma : np.ndarray  Shape (n, n) — covariance matrix Σ_λ.
    eps : float         Small regularization added to diagonal of Σ.
    rf : float          Risk-free rate (0 per paper).

    Returns
    -------
    np.ndarray  Shape (n,) — optimal portfolio weights.
    """
    n = len(mu)
    Sigma_reg = Sigma + eps * np.eye(n)

    # Sharpe objective (negated for minimization)
    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu) - rf
        port_vol = np.sqrt(float(w @ Sigma_reg @ w))
        if port_vol < 1e-12:
            return 0.0
        return -port_ret / port_vol

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        port_ret = float(w @ mu) - rf
        port_var = float(w @ Sigma_reg @ w)
        port_vol = np.sqrt(max(port_var, 1e-24))
        grad_ret = mu
        grad_vol = Sigma_reg @ w / port_vol
        return -(grad_ret * port_vol - port_ret * grad_vol) / (port_vol**2)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # Multi-start to avoid local minima
    best_w = None
    best_sharpe = -np.inf

    # Initial points: equal weight + random
    inits = [np.ones(n) / n] + [
        np.random.dirichlet(np.ones(n)) for _ in range(5)
    ]

    for w0 in inits:
        result = minimize(
            neg_sharpe,
            w0,
            jac=neg_sharpe_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success or result.fun < best_sharpe:
            s = -result.fun
            if s > best_sharpe:
                best_sharpe = s
                best_w = result.x

    if best_w is None:
        logger.warning("Max-Sharpe optimization failed; returning equal weights.")
        best_w = np.ones(n) / n

    # Project onto simplex to fix numerical errors
    best_w = np.clip(best_w, 0.0, 1.0)
    best_w /= best_w.sum()
    return best_w


# ──────────────────────────────────────────────────────────────────────────────
# Portfolio evaluation
# ──────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    weights: np.ndarray,
    returns_eval: pd.DataFrame,
    cost_per_trade: float = TRANSACTION_COST_BPS,
    charge_initial: bool = True,
) -> dict:
    """
    Evaluate a constant-weight portfolio over the evaluation period.

    The Markowitz portfolio has constant target weights. Daily rebalancing
    back to target weights is simulated.

    Implementation assumption (transaction cost convention):
        - Pre-trade weights = post-return drifted weights from previous day.
        - Day 0: pre-trade weights are zero (all cash), so initial trade
          cost is incurred if charge_initial=True.
        - cost_t = 0.002 * Σ_i |w_i^target - w_i^pre-trade|

    Parameters
    ----------
    weights : np.ndarray        Shape (n,) — constant target weights.
    returns_eval : pd.DataFrame Shape (T, n) — evaluation period daily returns.
    cost_per_trade : float      One-way cost (20 bps = 0.002).
    charge_initial : bool       Whether to charge cost for initial allocation.

    Returns
    -------
    dict with:
        'gross_returns' : pd.Series — daily gross portfolio returns.
        'net_returns'   : pd.Series — daily net portfolio returns.
        'turnover'      : pd.Series — daily turnover Σ_i |Δw_i|.
    """
    T, n = returns_eval.shape
    R = returns_eval.values  # (T, n)

    gross_rets = np.zeros(T)
    turnover = np.zeros(T)
    costs = np.zeros(T)

    w_target = weights.copy()
    w_pre = np.zeros(n)  # Start from cash

    for t in range(T):
        # Turnover = sum of absolute weight changes
        delta_w = np.abs(w_target - w_pre)
        to = delta_w.sum()
        turnover[t] = to

        if t == 0 and not charge_initial:
            cost = 0.0
        else:
            cost = cost_per_trade * to

        costs[t] = cost

        # Gross daily return
        r_t = R[t]
        gross_r = float(w_target @ r_t)
        gross_rets[t] = gross_r

        # After today's return, weights drift
        # w_i_post = w_i_target * (1 + r_{t,i}) / (1 + port_ret)
        port_ret = gross_r
        if abs(1 + port_ret) > 1e-12:
            w_drift = w_target * (1 + r_t) / (1 + port_ret)
        else:
            w_drift = w_target.copy()

        # Next period pre-trade = drifted weights (rebalanced back to target)
        w_pre = w_drift

    net_rets = gross_rets - costs

    index = returns_eval.index
    return {
        "gross_returns": pd.Series(gross_rets, index=index),
        "net_returns": pd.Series(net_rets, index=index),
        "turnover": pd.Series(turnover, index=index),
    }


def compute_performance_metrics(
    gross_returns: pd.Series,
    net_returns: pd.Series,
    turnover: pd.Series,
    annualize: int = ANNUALIZE_FACTOR,
) -> dict:
    """
    Compute standard performance metrics from daily return series.

    Annualization uses 252 trading days per year (documented assumption).
    Sortino ratio uses downside deviation relative to 0 (rf=0).

    Parameters
    ----------
    gross_returns : pd.Series  Daily gross returns.
    net_returns : pd.Series    Daily net returns.
    turnover : pd.Series       Daily portfolio turnover.
    annualize : int            Annualization factor (252).

    Returns
    -------
    dict with performance metrics.
    """
    def sharpe(rets: pd.Series) -> float:
        mu = rets.mean()
        sigma = rets.std(ddof=1)
        # Clip to eps so positive-mean, zero-std case yields a large finite Sharpe
        sigma = max(sigma, 1e-10)
        return float(mu / sigma * np.sqrt(annualize))

    def sortino(rets: pd.Series) -> float:
        mu = rets.mean()
        neg = rets[rets < 0]
        if len(neg) == 0:
            return np.inf
        dd = np.sqrt((neg**2).mean())
        if dd < 1e-12:
            return 0.0
        return float(mu / dd * np.sqrt(annualize))

    def max_drawdown(rets: pd.Series) -> float:
        wealth = (1 + rets).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        return float(dd.min())

    def final_wealth(rets: pd.Series) -> float:
        return float((1 + rets).prod())

    def annual_vol(rets: pd.Series) -> float:
        return float(rets.std(ddof=1) * np.sqrt(annualize))

    return {
        "gross_sharpe": sharpe(gross_returns),
        "net_sharpe": sharpe(net_returns),
        "sortino": sortino(net_returns),
        "net_volatility": annual_vol(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "final_wealth": final_wealth(net_returns),
        "avg_daily_turnover": float(turnover.mean()),
        "annualized_gross_return": float(gross_returns.mean() * annualize),
        "annualized_net_return": float(net_returns.mean() * annualize),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full experiment run per λ
# ──────────────────────────────────────────────────────────────────────────────

def run_markowitz_experiment(
    tilt_results: list[dict],
    returns_eval: pd.DataFrame,
    eps: float = 1e-8,
) -> list[dict]:
    """
    Run Markowitz max-Sharpe portfolio for each λ and evaluate out-of-sample.

    Parameters
    ----------
    tilt_results : list of dict
        Output of kl_tilting.compute_all_lambda_tilt. Each element contains
        'lambda', 'mu_lambda', 'Sigma_lambda'.
    returns_eval : pd.DataFrame  Shape (T, n) — evaluation-period returns.
    eps : float                  Covariance regularization.

    Returns
    -------
    list of dicts, one per λ, with keys:
        'lambda', 'weights', 'metrics', 'gross_returns', 'net_returns', 'turnover'
    """
    results = []
    assets = returns_eval.columns.tolist()

    for tr in tilt_results:
        lam = tr["lambda"]
        mu_lam = tr["mu_lambda"]
        Sigma_lam = tr["Sigma_lambda"]

        logger.info("Markowitz optimization λ=%.2f ...", lam)

        np.random.seed(42)  # reproducible multi-start
        weights = max_sharpe_long_only(mu_lam, Sigma_lam, eps=eps)

        port_result = compute_portfolio_returns(weights, returns_eval)
        metrics = compute_performance_metrics(
            port_result["gross_returns"],
            port_result["net_returns"],
            port_result["turnover"],
        )

        logger.info(
            "λ=%.2f: gross_SR=%.3f, net_SR=%.3f, final_W=%.3f, turnover=%.4f",
            lam,
            metrics["gross_sharpe"],
            metrics["net_sharpe"],
            metrics["final_wealth"],
            metrics["avg_daily_turnover"],
        )

        results.append({
            "lambda": lam,
            "weights": dict(zip(assets, weights)),
            "weights_array": weights,
            "metrics": metrics,
            "gross_returns": port_result["gross_returns"],
            "net_returns": port_result["net_returns"],
            "turnover": port_result["turnover"],
        })

    return results
