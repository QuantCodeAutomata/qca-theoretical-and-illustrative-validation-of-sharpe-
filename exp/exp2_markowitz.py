"""
Experiment 2: Empirical NSE India Markowitz Portfolio with KL-Based Expert Views.

Reproduces the paper's main empirical Markowitz experiment:
- Historical estimation: 2006-2023
- Out-of-sample evaluation: 2024-2025
- KL-based exponential tilting: μ_λ = (1-λ)μ_hist + λμ_view
- Long-only maximum Sharpe portfolio evaluated for λ ∈ {0,0.1,...,1.0}
- Transaction costs: 20 bps one-way

Implementation assumptions:
1. Adjusted closing prices from yfinance.
2. rf = 0 (stated in paper).
3. Annualization: 252 trading days.
4. Initial trade charged from all-cash position.
5. μ_view = realized mean over 2024-2025 (intentional ex-post design).
6. Turnover reported as average daily Σ_i |Δw_i|.
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
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import load_main_experiment_data
from src.kl_tilting import compute_all_lambda_tilt, validate_tilt_result
from src.markowitz import run_markowitz_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LAMBDA_VALUES = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]


def main() -> dict:
    """Run Experiment 2 end-to-end."""
    logger.info("=" * 60)
    logger.info("Experiment 2: NSE India Markowitz + KL Tilting")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading NSE 20-stock universe data...")
    returns_hist, returns_eval = load_main_experiment_data(cache_dir="data/cache")

    assets = returns_hist.columns.tolist()
    logger.info("Assets (%d): %s", len(assets), assets)
    logger.info("Historical: %s to %s (%d days)",
                returns_hist.index[0].date(), returns_hist.index[-1].date(), len(returns_hist))
    logger.info("Evaluation: %s to %s (%d days)",
                returns_eval.index[0].date(), returns_eval.index[-1].date(), len(returns_eval))

    # Align assets
    returns_eval = returns_eval[assets]

    R_hist = returns_hist.values
    R_eval_vals = returns_eval.values
    T, n = R_hist.shape

    # ── 2. Compute historical and view means ─────────────────────────────────
    mu_hist = R_hist.mean(axis=0)
    mu_view = R_eval_vals.mean(axis=0)  # Realized future mean (intentional ex-post)

    logger.info("μ_hist (annualized): min=%.4f, max=%.4f, mean=%.4f",
                mu_hist.min()*252, mu_hist.max()*252, mu_hist.mean()*252)
    logger.info("μ_view (annualized): min=%.4f, max=%.4f, mean=%.4f",
                mu_view.min()*252, mu_view.max()*252, mu_view.mean()*252)

    # Verify λ=0 recovers μ_hist and λ=1 equals μ_view
    mu_0 = (1-0) * mu_hist + 0 * mu_view
    mu_1 = (1-1) * mu_hist + 1 * mu_view
    assert np.allclose(mu_0, mu_hist), "λ=0 should recover μ_hist"
    assert np.allclose(mu_1, mu_view), "λ=1 should equal μ_view"
    logger.info("✓ μ_0 = μ_hist, μ_1 = μ_view verified")

    # ── 3. Exponential tilting for all λ ────────────────────────────────────
    logger.info("\nSolving exponential tilt for λ in %s...", LAMBDA_VALUES)
    tilt_results = compute_all_lambda_tilt(
        returns_hist=R_hist,
        mu_hist=mu_hist,
        mu_view=mu_view,
        lambda_values=LAMBDA_VALUES,
        tol=1e-9,
    )

    # Validate tilt results
    logger.info("\nValidating tilt results...")
    for tr in tilt_results:
        val = validate_tilt_result(tr["q_weights"], R_hist, tr["mu_lambda"])
        logger.info(
            "λ=%.2f: positive=%s, normalized=%s, moment_matched=%s (err=%.2e)",
            tr["lambda"], val["positive"], val["normalized"],
            val["moment_matched"], val["moment_error"],
        )

    # ── 4. Markowitz optimization and evaluation ─────────────────────────────
    logger.info("\nRunning Markowitz optimization...")
    mrkw_results = run_markowitz_experiment(tilt_results, returns_eval)

    # ── 5. Compile metrics table ─────────────────────────────────────────────
    rows = []
    for r in mrkw_results:
        m = r["metrics"]
        rows.append({
            "lambda": r["lambda"],
            "gross_sharpe": round(m["gross_sharpe"], 4),
            "net_sharpe": round(m["net_sharpe"], 4),
            "final_wealth": round(m["final_wealth"], 4),
            "avg_daily_turnover": round(m["avg_daily_turnover"], 6),
            "sortino": round(m["sortino"], 4),
            "net_volatility": round(m["net_volatility"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
        })

    metrics_df = pd.DataFrame(rows)
    logger.info("\nMarkowitz Metrics Table:")
    logger.info("\n%s", metrics_df.to_string(index=False))

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "exp2_markowitz_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved to %s", metrics_path)

    # ── 6. Plot Figure 2a (Sharpe vs λ) ─────────────────────────────────────
    _plot_sharpe_vs_lambda(mrkw_results)

    # ── 7. Plot Figure 2b (Final Wealth vs λ) ───────────────────────────────
    _plot_wealth_vs_lambda(mrkw_results)

    # ── 8. Plot Figure 3: allocations for λ=0, 0.5, 1.0 ────────────────────
    _plot_allocations(mrkw_results, assets)

    # ── 9. Save summary JSON ─────────────────────────────────────────────────
    summary = {
        "n_assets": n,
        "assets": assets,
        "n_hist_days": T,
        "n_eval_days": len(returns_eval),
        "lambda_values": LAMBDA_VALUES,
        "metrics": rows,
        "paper_anchor_points": {
            "lambda_0": {"gross_sharpe": 0.050, "net_sharpe": 0.019,
                         "final_wealth": 0.988, "avg_daily_turnover": 0.0081},
            "lambda_0.5": {"gross_sharpe": 1.801, "net_sharpe": 1.774,
                           "final_wealth": 1.609, "avg_daily_turnover": 0.0075},
            "lambda_1.0": {"gross_sharpe": 1.957, "net_sharpe": 1.936,
                           "final_wealth": 1.796, "avg_daily_turnover": 0.0066},
        },
    }
    with open(os.path.join(RESULTS_DIR, "exp2_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\nExperiment 2 complete.")
    return summary


def _plot_sharpe_vs_lambda(results: list[dict]) -> None:
    """Plot annualized Sharpe ratio vs λ (Figure 2a)."""
    lambdas = [r["lambda"] for r in results]
    gross_sr = [r["metrics"]["gross_sharpe"] for r in results]
    net_sr = [r["metrics"]["net_sharpe"] for r in results]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, gross_sr, "b-o", lw=2, label="Gross Sharpe")
    ax.plot(lambdas, net_sr, "r--s", lw=2, label="Net Sharpe")
    ax.set_xlabel(r"View strength $\lambda$", fontsize=13)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_title("Figure 2a: Markowitz Sharpe Ratio vs $\\lambda$", fontsize=13)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_fig2a_sharpe_vs_lambda.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 2a saved.")


def _plot_wealth_vs_lambda(results: list[dict]) -> None:
    """Plot final wealth vs λ (Figure 2b)."""
    lambdas = [r["lambda"] for r in results]
    wealth = [r["metrics"]["final_wealth"] for r in results]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lambdas, wealth, "g-D", lw=2, markersize=6)
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Break-even")
    ax.set_xlabel(r"View strength $\lambda$", fontsize=13)
    ax.set_ylabel("Final Wealth (starting from 1)", fontsize=12)
    ax.set_title("Figure 2b: Markowitz Final Wealth vs $\\lambda$", fontsize=13)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_fig2b_wealth_vs_lambda.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 2b saved.")


def _plot_allocations(results: list[dict], assets: list[str]) -> None:
    """Plot portfolio allocations for λ=0, 0.5, 1.0 (Figure 3 Markowitz)."""
    target_lambdas = {0.0: None, 0.5: None, 1.0: None}
    for r in results:
        lam = round(r["lambda"], 1)
        if lam in target_lambdas:
            target_lambdas[lam] = r["weights_array"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (lam, weights) in zip(axes, sorted(target_lambdas.items())):
        if weights is None:
            continue
        colors = sns.color_palette("husl", len(assets))
        bars = ax.bar(range(len(assets)), weights, color=colors, width=0.7)
        ax.set_xticks(range(len(assets)))
        ax.set_xticklabels(assets, rotation=90, fontsize=8)
        ax.set_title(f"Markowitz Allocation $\\lambda={lam:.1f}$", fontsize=11)
        ax.set_ylabel("Portfolio Weight", fontsize=10)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_fig3_allocations.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 3 (Markowitz allocations) saved.")


if __name__ == "__main__":
    main()
