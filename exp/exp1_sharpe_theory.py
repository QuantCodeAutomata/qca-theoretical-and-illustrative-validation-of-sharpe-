"""
Experiment 1: Theoretical and Illustrative Validation of Sharpe Ratio Behavior.

Reproduces Figure 1-type results showing how realized Sharpe changes as the
estimated mean vector is interpolated toward the true future mean:
    ν_λ = (1-λ)ν + λμ, λ ∈ [0,1]

Two datasets:
- Dataset 1: 4 NSE stocks (2020-2022 est., 2023 eval.)  → globally concave case
- Dataset 2: 20 NSE stocks (2006-2020 est., 2021-2025 eval.) → inflection case

Implementation assumptions documented:
1. Evaluation window for Dataset 1: 2023 (paper does not fully specify this).
2. Estimation/evaluation for Dataset 2: 2006-2020 / 2021-2025 (assumption).
3. Covariance: Ω=Σ used (simplified theorem case with closed-form formulas).
4. Regularization: Σ + 1e-8 I applied if needed for invertibility.
5. Adjusted closing prices from yfinance (corporate-action assumption).
6. Annualized Sharpe ratio NOT used; raw period Sharpe shown (documented).
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import load_exp1_dataset1, load_exp1_dataset2
from src.sharpe_theory import (
    compute_ABC,
    compute_sharpe_vs_lambda,
    optimal_sharpe,
    angle_interpretation,
    check_monotonicity,
    safe_inv,
    regularize_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dataset(
    returns_est: pd.DataFrame,
    returns_eval: pd.DataFrame,
    label: str,
    eps: float = 1e-8,
) -> dict:
    """
    Run the Sharpe-vs-λ analysis for one dataset.

    Parameters
    ----------
    returns_est : pd.DataFrame   Estimation-window daily returns.
    returns_eval : pd.DataFrame  Evaluation-window daily returns.
    label : str                  Dataset label for logging.
    eps : float                  Covariance regularization.

    Returns
    -------
    dict with all computed quantities.
    """
    # Align columns between estimation and evaluation
    common_cols = list(set(returns_est.columns) & set(returns_eval.columns))
    common_cols.sort()
    R_est = returns_est[common_cols].values
    R_eval = returns_eval[common_cols].values

    logger.info("%s: estimation %d obs, evaluation %d obs, %d assets",
                label, R_est.shape[0], R_eval.shape[0], len(common_cols))

    # From estimation window: forecast mean ν and covariance Ω
    nu = R_est.mean(axis=0)
    Omega = np.cov(R_est, rowvar=False)

    # From evaluation window: realized mean μ and covariance Σ
    mu = R_eval.mean(axis=0)
    Sigma = np.cov(R_eval, rowvar=False)

    # Regularize
    Omega_reg = regularize_matrix(Omega, eps)
    Sigma_reg = regularize_matrix(Sigma, eps)

    # Using Ω=Σ simplification for closed-form theorem (documented assumption)
    Sigma_inv = safe_inv(Sigma_reg)

    diag = compute_ABC(nu, mu, Sigma_inv)
    A, B, C = diag.A, diag.B, diag.C
    concave = diag.concave

    logger.info("%s: A=%.4f, B=%.4f, C=%.4f, A+3B=%.4f (concave=%s)",
                label, A, B, C, A + 3*B, concave)

    # S* (optimal Sharpe)
    S_star = optimal_sharpe(mu, Sigma_reg)

    # Lambda grid
    lambda_grid = np.linspace(0.0, 1.0, 101)

    # Compute Sharpe vs λ with Ω=Σ
    result = compute_sharpe_vs_lambda(
        nu=nu,
        mu=mu,
        Sigma=Sigma_reg,
        Omega=None,  # Ω=Σ
        lambda_grid=lambda_grid,
        eps=eps,
    )

    # Also compute with Ω≠Σ (general misspecification)
    result_general = compute_sharpe_vs_lambda(
        nu=nu,
        mu=mu,
        Sigma=Sigma_reg,
        Omega=Omega_reg,
        lambda_grid=lambda_grid,
        eps=eps,
    )

    # Validate S(1) ≈ S* in the Ω=Σ case
    S_at_1 = result["sharpe_vals"][-1]
    S_star_check = result["S_star"]
    err = abs(S_at_1 - S_star_check)
    logger.info("%s: S(λ=1)=%.6f, S*=%.6f, |error|=%.2e", label, S_at_1, S_star_check, err)

    # Monotonicity check
    is_mono = check_monotonicity(result["sharpe_vals"])
    logger.info("%s: monotone=%s", label, is_mono)

    # Angle interpretation at λ=0 (ν_0 = ν)
    cos_theta = angle_interpretation(mu, Sigma_reg, nu, Sigma_reg)
    S_realized_0 = result["sharpe_vals"][0]
    S_check_via_angle = S_star * cos_theta
    logger.info(
        "%s: S(0)=%.4f, S*·cos(θ)=%.4f (cos_θ=%.4f)",
        label, S_realized_0, S_check_via_angle, cos_theta,
    )

    return {
        "label": label,
        "common_cols": common_cols,
        "nu": nu,
        "mu": mu,
        "Omega": Omega_reg,
        "Sigma": Sigma_reg,
        "A": A, "B": B, "C": C,
        "concave_condition": concave,
        "S_star": S_star,
        "lambda_grid": lambda_grid,
        "sharpe_vals": result["sharpe_vals"],
        "sharpe_cf": result["sharpe_cf"],
        "sharpe_general": result_general["sharpe_vals"],
        "deriv1": result["deriv1"],
        "deriv2": result["deriv2"],
        "is_monotone": is_mono,
        "cos_theta": cos_theta,
        "n_est": R_est.shape[0],
        "n_eval": R_eval.shape[0],
    }


def plot_sharpe_curves(
    ds1: dict,
    ds2: dict,
    save_path: str,
) -> None:
    """
    Produce Figure 1-type plots: Sharpe vs λ for both datasets.

    Left: Dataset 1 (globally concave)
    Right: Dataset 2 (inflection/non-concave case)
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ds in zip(axes, [ds1, ds2]):
        lam = ds["lambda_grid"]
        sr = ds["sharpe_vals"]
        sr_cf = ds["sharpe_cf"]
        sr_gen = ds["sharpe_general"]
        S_star = ds["S_star"]
        A, B, C = ds["A"], ds["B"], ds["C"]
        label = ds["label"]
        concave = ds["concave_condition"]

        ax.plot(lam, sr, "b-", lw=2.5, label=r"Realized $S(\lambda)$ [$\Omega=\Sigma$]")
        ax.plot(lam, sr_cf, "b--", lw=1.5, alpha=0.7, label="Closed-form")
        ax.plot(lam, sr_gen, "r-", lw=2, alpha=0.8, label=r"Realized $S(\lambda)$ [$\Omega\neq\Sigma$]")
        ax.axhline(S_star, color="green", ls=":", lw=1.5, label=r"$S^*_{\mu,\Sigma}$")

        # Mark S(0) and S(1)
        ax.scatter([0, 1], [sr[0], sr[-1]], color="blue", zorder=5)

        condition_str = f"$A+3B = {A+3*B:.3f}$ {'≥0 ✓' if concave else '<0 (inflection)'}"
        ax.set_title(
            f"{label}\n"
            f"$A={A:.3f},\\ B={B:.3f},\\ C={C:.3f}$\n"
            f"{condition_str}",
            fontsize=10,
        )
        ax.set_xlabel(r"$\lambda$", fontsize=13)
        ax.set_ylabel("Realized Sharpe Ratio", fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved to %s", save_path)


def plot_derivatives(ds: dict, save_path: str) -> None:
    """Plot first and second derivatives of S(λ) for a dataset."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    lam = ds["lambda_grid"]
    d1 = ds["deriv1"]
    d2 = ds["deriv2"]
    label = ds["label"]

    axes[0].plot(lam, d1, "b-", lw=2)
    axes[0].axhline(0, color="k", ls="--", lw=0.8)
    axes[0].set_title(f"{label}: $S'(\\lambda)$")
    axes[0].set_xlabel(r"$\lambda$")
    axes[0].set_ylabel(r"$S'(\lambda)$")

    axes[1].plot(lam, d2, "r-", lw=2)
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_title(f"{label}: $S''(\\lambda)$")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel(r"$S''(\lambda)$")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_results(ds1: dict, ds2: dict) -> dict:
    """Save numerical results and return summary dict."""
    summary = {}

    for ds in [ds1, ds2]:
        label = ds["label"]
        key = label.lower().replace(" ", "_")
        summary[key] = {
            "assets": ds["common_cols"],
            "n_estimation_obs": ds["n_est"],
            "n_evaluation_obs": ds["n_eval"],
            "A": round(ds["A"], 6),
            "B": round(ds["B"], 6),
            "C": round(ds["C"], 6),
            "A_plus_3B": round(ds["A"] + 3 * ds["B"], 6),
            "concavity_condition_holds": bool(ds["concave_condition"]),
            "S_star": round(ds["S_star"], 6),
            "S_at_lambda_0": round(float(ds["sharpe_vals"][0]), 6),
            "S_at_lambda_0.5": round(float(ds["sharpe_vals"][50]), 6),
            "S_at_lambda_1": round(float(ds["sharpe_vals"][-1]), 6),
            "is_monotone": bool(ds["is_monotone"]),
            "cos_theta_at_lambda_0": round(ds["cos_theta"], 6),
        }

    return summary


def main() -> None:
    """Run Experiment 1 end-to-end."""
    logger.info("=" * 60)
    logger.info("Experiment 1: Sharpe Theory Validation")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading Dataset 1 (4 stocks)...")
    r_est1, r_eval1 = load_exp1_dataset1(cache_dir="data/cache")

    logger.info("Loading Dataset 2 (20 stocks)...")
    r_est2, r_eval2 = load_exp1_dataset2(cache_dir="data/cache")

    # Run analysis
    logger.info("\n--- Dataset 1 Analysis ---")
    ds1 = run_dataset(r_est1, r_eval1, label="Dataset 1 (4 stocks, concave)")

    logger.info("\n--- Dataset 2 Analysis ---")
    ds2 = run_dataset(r_est2, r_eval2, label="Dataset 2 (20 stocks, inflection)")

    # Determine which is concave / has inflection based on data
    # (Dataset 1 expected to be concave; Dataset 2 may show inflection)
    logger.info("\n--- Results Summary ---")
    logger.info("Dataset 1 - A+3B=%.4f, concave=%s, monotone=%s",
                ds1["A"] + 3*ds1["B"], ds1["concave_condition"], ds1["is_monotone"])
    logger.info("Dataset 2 - A+3B=%.4f, concave=%s, monotone=%s",
                ds2["A"] + 3*ds2["B"], ds2["concave_condition"], ds2["is_monotone"])

    # Plots
    plot_sharpe_curves(
        ds1, ds2,
        save_path=os.path.join(RESULTS_DIR, "exp1_sharpe_vs_lambda.png"),
    )
    plot_derivatives(
        ds1,
        save_path=os.path.join(RESULTS_DIR, "exp1_derivatives_ds1.png"),
    )
    plot_derivatives(
        ds2,
        save_path=os.path.join(RESULTS_DIR, "exp1_derivatives_ds2.png"),
    )

    # Save numerical results
    summary = save_results(ds1, ds2)
    results_path = os.path.join(RESULTS_DIR, "exp1_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Final checks
    logger.info("\n--- Verification ---")
    logger.info("S(0)→S(1) improvement Dataset 1: %.4f → %.4f",
                ds1["sharpe_vals"][0], ds1["sharpe_vals"][-1])
    logger.info("S(0)→S(1) improvement Dataset 2: %.4f → %.4f",
                ds2["sharpe_vals"][0], ds2["sharpe_vals"][-1])

    return summary


if __name__ == "__main__":
    main()
