"""
Experiment 3: NSE India LSTM Portfolio with KL-Weighted Sharpe Objective.

Reproduces the paper's end-to-end machine learning experiment:
- Training: LSTM with KL-tilted Sharpe objective on 2006-2023 data
- Evaluation: 2024-2025 out-of-sample period
- 5 random seeds per λ for CPU run (paper: 100 seeds)
- λ ∈ {0, 0.1, ..., 1.0}

Implementation assumptions documented:
1. Lookback L = 20 days (not specified in paper).
2. Optimizer: Adam (lr=1e-3 warm-up, cosine annealing down to 1e-5).
3. Full-batch training, 300 iterations (50 warm-up + 250 cosine annealing).
4. Gradient clipping: max_norm=1.0.
5. No dropout or L2 weight decay (not specified).
6. Initial transaction cost charged from all-cash position.
7. Tilt weights mapped to sequences by next-day return index.
8. Annualization: 252 trading days.
9. Adjusted closing prices from yfinance.
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
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetch_data import load_main_experiment_data
from src.kl_tilting import compute_all_lambda_tilt, validate_tilt_result
from src.lstm_portfolio import run_lstm_for_lambda, LOOKBACK, N_SEEDS

# CPU-only constraint: paper specifies 100 seeds; reduced to 5 for practical runtime.
# (15.4 s/seed × 11 λ × 100 seeds ≈ 4.7 h on CPU; × 5 seeds ≈ 14 min)
N_SEEDS = 5  # noqa: F811 (intentional override of imported constant)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

LAMBDA_VALUES = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> dict:
    """Run Experiment 3 end-to-end."""
    logger.info("=" * 60)
    logger.info("Experiment 3: NSE India LSTM Portfolio + KL Tilting")
    logger.info("Using device: %s", DEVICE)
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading NSE 20-stock universe data...")
    returns_hist, returns_eval = load_main_experiment_data(cache_dir="data/cache")

    assets = returns_hist.columns.tolist()
    n_assets = len(assets)
    logger.info("Assets (%d): %s", n_assets, assets)

    # Align assets in evaluation
    returns_eval = returns_eval[assets]

    R_hist = returns_hist.values
    R_eval_vals = returns_eval.values
    T, n = R_hist.shape

    # ── 2. Compute historical and view means ─────────────────────────────────
    mu_hist = R_hist.mean(axis=0)
    mu_view = R_eval_vals.mean(axis=0)  # Realized future mean (ex-post design)

    logger.info("Historical period: %s to %s (%d days)",
                returns_hist.index[0].date(), returns_hist.index[-1].date(), T)
    logger.info("Evaluation period: %s to %s (%d days)",
                returns_eval.index[0].date(), returns_eval.index[-1].date(), len(returns_eval))

    # ── 3. Exponential tilting ───────────────────────────────────────────────
    logger.info("\nSolving exponential tilt for all λ values...")
    tilt_results_list = compute_all_lambda_tilt(
        returns_hist=R_hist,
        mu_hist=mu_hist,
        mu_view=mu_view,
        lambda_values=LAMBDA_VALUES,
        tol=1e-9,
    )
    tilt_dict = {tr["lambda"]: tr for tr in tilt_results_list}

    # Validate
    for tr in tilt_results_list:
        val = validate_tilt_result(tr["q_weights"], R_hist, tr["mu_lambda"])
        if not val["moment_matched"]:
            logger.warning("λ=%.2f: moment matching failed (err=%.2e)", tr["lambda"], val["moment_error"])

    # ── 4. LSTM training and evaluation for each λ ───────────────────────────
    all_lstm_results = []

    for lam in LAMBDA_VALUES:
        logger.info("\n" + "="*50)
        logger.info("λ = %.2f", lam)
        logger.info("="*50)

        tilt_data = tilt_dict[lam]

        result = run_lstm_for_lambda(
            lam=lam,
            tilt_data=tilt_data,
            returns_train=R_hist,
            returns_eval=returns_eval,
            n_seeds=N_SEEDS,
            lookback=LOOKBACK,
            device=DEVICE,
            verbose=True,
        )
        all_lstm_results.append(result)

    # ── 5. Compile metrics table ─────────────────────────────────────────────
    rows = []
    for r in all_lstm_results:
        if "error" in r:
            continue
        m = r["mean_metrics"]
        rows.append({
            "lambda": r["lambda"],
            "gross_sharpe": round(m["gross_sharpe"], 4),
            "net_sharpe": round(m["net_sharpe"], 4),
            "final_wealth": round(m["final_wealth"], 4),
            "avg_daily_turnover": round(m["avg_daily_turnover"], 6),
            "sortino": round(m["sortino"], 4),
            "net_volatility": round(m["net_volatility"], 4),
            "max_drawdown": round(m["max_drawdown"], 4),
            "n_valid_seeds": r["n_valid_seeds"],
        })

    metrics_df = pd.DataFrame(rows)
    logger.info("\nLSTM Metrics Table (averaged over seeds):")
    logger.info("\n%s", metrics_df.to_string(index=False))

    metrics_path = os.path.join(RESULTS_DIR, "exp3_lstm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved to %s", metrics_path)

    # ── 6. Plot Figure 2a (Sharpe vs λ) ─────────────────────────────────────
    _plot_sharpe_vs_lambda(all_lstm_results)

    # ── 7. Plot Figure 2b (Final Wealth vs λ) ───────────────────────────────
    _plot_wealth_vs_lambda(all_lstm_results)

    # ── 8. Plot Figure 3: mean allocations for λ=0, 0.5, 1.0 ───────────────
    _plot_mean_allocations(all_lstm_results, assets)

    # ── 9. Save summary JSON ─────────────────────────────────────────────────
    summary = {
        "n_assets": n_assets,
        "assets": assets,
        "n_hist_days": T,
        "n_eval_days": len(returns_eval),
        "lookback": LOOKBACK,
        "n_seeds_run": N_SEEDS,
        "n_seeds_paper": 100,
        "device": DEVICE,
        "lambda_values": LAMBDA_VALUES,
        "metrics": rows,
        "implementation_assumptions": {
            "lookback_L": LOOKBACK,
            "optimizer": "Adam",
            "lr_warmup": 1e-3,
            "lr_min": 1e-5,
            "n_warmup_iter": 50,
            "n_cosine_iter": 250,
            "grad_clip": 1.0,
            "hidden_size": 16,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "rf": 0.0,
            "annualization": 252,
        },
        "paper_anchor_points": {
            "lambda_0": {"gross_sharpe": 0.037, "net_sharpe": -0.021,
                         "final_wealth": 0.978, "avg_daily_turnover": 0.0153},
            "lambda_0.5": {"gross_sharpe": 1.733, "net_sharpe": 1.691,
                           "final_wealth": 1.565, "avg_daily_turnover": 0.0118},
            "lambda_1.0": {"gross_sharpe": 1.936, "net_sharpe": 1.912,
                           "final_wealth": 1.769, "avg_daily_turnover": 0.0074},
        },
    }
    with open(os.path.join(RESULTS_DIR, "exp3_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("\nExperiment 3 complete.")
    return summary


def _plot_sharpe_vs_lambda(results: list[dict]) -> None:
    """Plot mean Sharpe vs λ with dispersion bands (Figure 2a LSTM)."""
    valid = [r for r in results if "mean_metrics" in r]
    lambdas = [r["lambda"] for r in valid]
    gross_sr = [r["mean_metrics"]["gross_sharpe"] for r in valid]
    net_sr = [r["mean_metrics"]["net_sharpe"] for r in valid]
    gross_std = [r["std_metrics"]["gross_sharpe_std"] for r in valid]
    net_std = [r["std_metrics"]["net_sharpe_std"] for r in valid]

    lams = np.array(lambdas)
    g_sr = np.array(gross_sr)
    n_sr = np.array(net_sr)
    g_std = np.array(gross_std)
    n_std = np.array(net_std)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lams, g_sr, "b-o", lw=2, label="Gross Sharpe (mean)")
    ax.fill_between(lams, g_sr - g_std, g_sr + g_std, alpha=0.2, color="blue")
    ax.plot(lams, n_sr, "r--s", lw=2, label="Net Sharpe (mean)")
    ax.fill_between(lams, n_sr - n_std, n_sr + n_std, alpha=0.2, color="red")
    ax.set_xlabel(r"View strength $\lambda$", fontsize=13)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_title(f"Figure 2a: LSTM Sharpe Ratio vs $\\lambda$ ({N_SEEDS} seeds; paper: 100)", fontsize=12)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_fig2a_sharpe_vs_lambda.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("LSTM Figure 2a saved.")


def _plot_wealth_vs_lambda(results: list[dict]) -> None:
    """Plot mean final wealth vs λ (Figure 2b LSTM)."""
    valid = [r for r in results if "mean_metrics" in r]
    lambdas = [r["lambda"] for r in valid]
    wealth = [r["mean_metrics"]["final_wealth"] for r in valid]
    wealth_std = [r["std_metrics"]["final_wealth_std"] for r in valid]

    lams = np.array(lambdas)
    w_arr = np.array(wealth)
    w_std = np.array(wealth_std)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lams, w_arr, "g-D", lw=2, markersize=6, label="Mean Final Wealth")
    ax.fill_between(lams, w_arr - w_std, w_arr + w_std, alpha=0.2, color="green")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="Break-even")
    ax.set_xlabel(r"View strength $\lambda$", fontsize=13)
    ax.set_ylabel("Final Wealth (starting from 1)", fontsize=12)
    ax.set_title(f"Figure 2b: LSTM Final Wealth vs $\\lambda$ ({N_SEEDS} seeds; paper: 100)", fontsize=12)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_fig2b_wealth_vs_lambda.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("LSTM Figure 2b saved.")


def _plot_mean_allocations(results: list[dict], assets: list[str]) -> None:
    """Plot mean daily allocations for λ=0, 0.5, 1.0 (Figure 3 LSTM)."""
    target_lambdas = {0.0: None, 0.5: None, 1.0: None}
    for r in results:
        lam = round(r["lambda"], 1)
        if lam in target_lambdas and "mean_weights" in r:
            target_lambdas[lam] = r["mean_weights"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (lam, weights) in zip(axes, sorted(target_lambdas.items())):
        if weights is None:
            ax.set_title(f"λ={lam:.1f}: no data")
            continue
        # Mean allocation across time
        mean_alloc = weights.mean(axis=0)
        colors = sns.color_palette("husl", len(assets))
        ax.bar(range(len(assets)), mean_alloc, color=colors, width=0.7)
        ax.set_xticks(range(len(assets)))
        ax.set_xticklabels(assets, rotation=90, fontsize=8)
        ax.set_title(f"LSTM Mean Allocation $\\lambda={lam:.1f}$", fontsize=11)
        ax.set_ylabel("Mean Portfolio Weight", fontsize=10)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_fig3_allocations.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Figure 3 (LSTM allocations) saved.")


if __name__ == "__main__":
    main()
