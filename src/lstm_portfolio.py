"""
LSTM Portfolio Learning Module — Experiment 3.

Implements end-to-end portfolio optimization with an LSTM that outputs
long-only fully invested daily weights. Trained using the view-adjusted
weighted Sharpe objective:

    S_{φ,λ} = m_λ(φ) / sqrt(v_λ(φ))

where:
    m_λ(φ) = E_P[ℓ_λ(r_t) R_t(φ)]     (weighted expected return)
    v_λ(φ) = E_P[ℓ_λ(r_t)(R_t(φ) - m_λ(φ))²]  (weighted variance)
    ℓ_λ(r_t) = T q_{t,λ}               (likelihood ratio weights)

Architecture: single-layer LSTM with 16 hidden units + softmax output.
Training: full-batch, 300 iterations (50 warm-up + 250 cosine annealing).
Lookback window L = 20 days (implementation assumption; not specified in paper).

Implementation assumptions:
- Optimizer: Adam with lr=1e-3 for warm-up phase.
- Cosine annealing: CosineAnnealingLR with T_max=250, eta_min=1e-5.
- Gradient clipping: max_norm=1.0 to stabilize training.
- Dropout: none (not specified in paper).
- L2 weight decay: 0.0 (not specified in paper).
- Lookback L: 20 days (not specified in paper).
- Initial transaction cost: charged (from all-cash position).
- Annualization: 252 trading days.

# Custom — Context7 found no library equivalent (paper KL-tilted Sharpe LSTM objective)
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

logger = logging.getLogger(__name__)

# Lookback window (implementation assumption)
LOOKBACK = 20
HIDDEN_SIZE = 16
N_SEEDS = 100
N_ITER_TOTAL = 300
N_WARMUP = 50
N_COSINE = 250
LR_WARMUP = 1e-3
LR_MIN = 1e-5
GRAD_CLIP = 1.0
TRANSACTION_COST_BPS = 0.002
ANNUALIZE_FACTOR = 252


# ──────────────────────────────────────────────────────────────────────────────
# LSTM model definition
# ──────────────────────────────────────────────────────────────────────────────

class LSTMPortfolio(nn.Module):
    """
    Single-layer LSTM portfolio network.

    Input:  sequence of L daily return vectors of shape (L, n_assets)
    Output: portfolio weights of shape (n_assets,) — long-only via softmax.
    """

    def __init__(self, n_assets: int, hidden_size: int = HIDDEN_SIZE):
        """
        Parameters
        ----------
        n_assets : int     Number of assets (input and output dimension).
        hidden_size : int  LSTM hidden units (16 per paper).
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_assets,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  Shape (L, n_assets) — lookback window.

        Returns
        -------
        torch.Tensor  Shape (n_assets,) — portfolio weights via softmax.
        """
        # Add batch dimension: (1, L, n_assets)
        out, _ = self.lstm(x.unsqueeze(0))
        # Take last timestep: (1, hidden_size)
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(0)  # (n_assets,)
        return torch.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training data preparation
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(
    returns_train: np.ndarray,
    lookback: int = LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rolling lookback windows for LSTM training.

    For each training index i in [lookback, T-1]:
        Input X[i] = returns[i-lookback : i]   shape (lookback, n)
        Target Y[i] = returns[i]               shape (n,) — next-day return

    Parameters
    ----------
    returns_train : np.ndarray  Shape (T, n) — training returns.
    lookback : int              Number of past days L.

    Returns
    -------
    tuple (X, Y) where:
        X : np.ndarray  Shape (T-lookback, lookback, n)
        Y : np.ndarray  Shape (T-lookback, n)
    """
    T, n = returns_train.shape
    X_list, Y_list = [], []
    for i in range(lookback, T):
        X_list.append(returns_train[i - lookback: i])
        Y_list.append(returns_train[i])
    return np.array(X_list), np.array(Y_list)


# ──────────────────────────────────────────────────────────────────────────────
# KL-tilted Sharpe objective
# ──────────────────────────────────────────────────────────────────────────────

def tilted_sharpe_loss(
    portfolio_returns: torch.Tensor,
    tilt_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute negative KL-tilted Sharpe ratio (for minimization).

    S_{φ,λ} = m_λ / sqrt(v_λ)

    where:
        m_λ = Σ_t ℓ_λ(r_t) R_t(φ)            (weighted mean)
        v_λ = Σ_t ℓ_λ(r_t) (R_t(φ) - m_λ)²  (weighted variance)
        ℓ_λ(r_t) = T q_{t,λ}                 (likelihood ratio)

    Note: tilt_weights here are already the likelihood ratios ℓ_λ = T q_{t,λ}.

    Parameters
    ----------
    portfolio_returns : torch.Tensor  Shape (N_seq,) — daily portfolio returns.
    tilt_weights : torch.Tensor       Shape (N_seq,) — likelihood ratio weights.

    Returns
    -------
    torch.Tensor  Scalar — negative Sharpe ratio.
    """
    # Normalize tilt weights to sum to 1 for expectation computation
    w = tilt_weights / tilt_weights.sum()

    m = (w * portfolio_returns).sum()
    v = (w * (portfolio_returns - m) ** 2).sum()

    eps = torch.tensor(1e-8)
    sharpe = m / torch.sqrt(v + eps)
    return -sharpe  # negate for minimization


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_lstm_portfolio(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    tilt_weights_seq: np.ndarray,
    n_assets: int,
    seed: int = 0,
    n_iter: int = N_ITER_TOTAL,
    n_warmup: int = N_WARMUP,
    lr_warmup: float = LR_WARMUP,
    lr_min: float = LR_MIN,
    grad_clip: float = GRAD_CLIP,
    device: str = "cpu",
) -> tuple[LSTMPortfolio, list[float]]:
    """
    Train a single LSTM portfolio model with the KL-tilted Sharpe objective.

    Training schedule (per paper):
        - 50 warm-up iterations with constant lr=lr_warmup
        - 250 iterations with cosine annealing from lr_warmup to lr_min

    Full-batch training: all sequences used in each iteration.

    Parameters
    ----------
    X_train : np.ndarray     Shape (N_seq, L, n) — input sequences.
    Y_train : np.ndarray     Shape (N_seq, n) — next-day returns.
    tilt_weights_seq : np.ndarray  Shape (N_seq,) — ℓ_λ for each sequence.
    n_assets : int           Number of assets.
    seed : int               Random seed for initialization.
    n_iter : int             Total training iterations (300).
    n_warmup : int           Warm-up iterations (50).
    lr_warmup : float        Learning rate during warm-up.
    lr_min : float           Minimum LR for cosine annealing.
    grad_clip : float        Gradient clipping max norm.
    device : str             'cpu' or 'cuda'.

    Returns
    -------
    tuple of (trained_model, loss_history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMPortfolio(n_assets=n_assets, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = Adam(model.parameters(), lr=lr_warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=(n_iter - n_warmup), eta_min=lr_min)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    W_t = torch.tensor(tilt_weights_seq, dtype=torch.float32, device=device)

    loss_history = []

    for it in range(n_iter):
        model.train()
        optimizer.zero_grad()

        # Forward pass: compute portfolio weights for each sequence
        # Full batch: iterate over sequences
        port_rets = []
        for i in range(len(X_t)):
            w_i = model(X_t[i])          # (n_assets,)
            r_i = float((w_i * Y_t[i]).sum())
            port_rets.append((w_i * Y_t[i]).sum())

        port_rets_t = torch.stack(port_rets)  # (N_seq,)

        loss = tilted_sharpe_loss(port_rets_t, W_t)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Only apply cosine annealing after warm-up
        if it >= n_warmup:
            scheduler.step()

        loss_history.append(float(loss.item()))

    return model, loss_history


def train_lstm_portfolio_batched(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    tilt_weights_seq: np.ndarray,
    n_assets: int,
    seed: int = 0,
    n_iter: int = N_ITER_TOTAL,
    n_warmup: int = N_WARMUP,
    lr_warmup: float = LR_WARMUP,
    lr_min: float = LR_MIN,
    grad_clip: float = GRAD_CLIP,
    device: str = "cpu",
) -> tuple[LSTMPortfolio, list[float]]:
    """
    Batched (vectorized) version of LSTM training for efficiency.

    Processes all sequences in parallel as a batch through the LSTM.

    Parameters
    ----------
    (Same as train_lstm_portfolio)

    Returns
    -------
    tuple of (trained_model, loss_history)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMPortfolio(n_assets=n_assets, hidden_size=HIDDEN_SIZE).to(device)
    optimizer = Adam(model.parameters(), lr=lr_warmup)
    scheduler = CosineAnnealingLR(optimizer, T_max=(n_iter - n_warmup), eta_min=lr_min)

    # Batched LSTM forward — modify model to handle batch
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)   # (N, L, n)
    Y_t = torch.tensor(Y_train, dtype=torch.float32, device=device)   # (N, n)
    W_t = torch.tensor(tilt_weights_seq, dtype=torch.float32, device=device)  # (N,)

    loss_history = []

    for it in range(n_iter):
        model.train()
        optimizer.zero_grad()

        # Batched forward pass through LSTM
        # X_t: (N, L, n) — batch_first=True
        lstm_out, _ = model.lstm(X_t)          # (N, L, hidden)
        last_hidden = lstm_out[:, -1, :]       # (N, hidden)
        logits = model.fc(last_hidden)         # (N, n_assets)
        weights = torch.softmax(logits, dim=-1)  # (N, n_assets)

        # Portfolio returns: R_t = w_t^T r_{t+1}
        port_rets = (weights * Y_t).sum(dim=1)  # (N,)

        loss = tilted_sharpe_loss(port_rets, W_t)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if it >= n_warmup:
            scheduler.step()

        loss_history.append(float(loss.item()))

    return model, loss_history


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_lstm_portfolio(
    model: LSTMPortfolio,
    returns_train: np.ndarray,
    returns_eval: pd.DataFrame,
    lookback: int = LOOKBACK,
    cost_per_trade: float = TRANSACTION_COST_BPS,
    device: str = "cpu",
) -> dict:
    """
    Generate daily portfolio weights over the evaluation period and compute returns.

    On each evaluation day t, the input is the previous L days of returns.
    The lookback window uses training data for the first L days.

    Implementation assumption (initial cost): charged from all-cash position on day 0.

    Parameters
    ----------
    model : LSTMPortfolio       Trained model.
    returns_train : np.ndarray  Shape (T_train, n) — training returns (for lookback).
    returns_eval : pd.DataFrame Shape (T_eval, n) — evaluation returns.
    lookback : int              Lookback window L.
    cost_per_trade : float      One-way transaction cost.
    device : str                Computation device.

    Returns
    -------
    dict with:
        'weights'       : np.ndarray  Shape (T_eval, n) — daily weights.
        'gross_returns' : pd.Series   Daily gross portfolio returns.
        'net_returns'   : pd.Series   Daily net portfolio returns.
        'turnover'      : pd.Series   Daily turnover.
    """
    model.eval()
    n = returns_train.shape[1]
    T_eval = len(returns_eval)

    R_eval = returns_eval.values  # (T_eval, n)

    # History for lookback: combine end of training + evaluation
    # Use only past data at each evaluation step
    all_returns = np.vstack([returns_train, R_eval])  # (T_train + T_eval, n)
    eval_start_idx = len(returns_train)  # index of first eval day in all_returns

    weights_all = np.zeros((T_eval, n))

    with torch.no_grad():
        for t in range(T_eval):
            # Past L returns available at time t (before observing day t return)
            idx = eval_start_idx + t
            window = all_returns[idx - lookback: idx]  # (L, n)
            x = torch.tensor(window, dtype=torch.float32, device=device)
            w = model(x).cpu().numpy()
            weights_all[t] = w

    # Compute returns with transaction costs
    gross_rets = np.zeros(T_eval)
    turnover = np.zeros(T_eval)
    costs = np.zeros(T_eval)

    w_pre = np.zeros(n)  # start from cash

    for t in range(T_eval):
        w_target = weights_all[t]
        delta_w = np.abs(w_target - w_pre)
        to = delta_w.sum()
        turnover[t] = to
        cost = cost_per_trade * to
        costs[t] = cost

        r_t = R_eval[t]
        gross_r = float(w_target @ r_t)
        gross_rets[t] = gross_r

        # Drift weights
        port_ret = gross_r
        if abs(1 + port_ret) > 1e-12:
            w_drift = w_target * (1 + r_t) / (1 + port_ret)
        else:
            w_drift = w_target.copy()
        w_pre = w_drift

    net_rets = gross_rets - costs
    index = returns_eval.index

    return {
        "weights": weights_all,
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
    Compute performance metrics for LSTM portfolio.

    Parameters
    ----------
    gross_returns : pd.Series
    net_returns : pd.Series
    turnover : pd.Series
    annualize : int  Annualization factor (252).

    Returns
    -------
    dict with performance metrics.
    """
    def sharpe(rets: pd.Series) -> float:
        mu = rets.mean()
        sigma = rets.std(ddof=1)
        return float(mu / sigma * np.sqrt(annualize)) if sigma > 1e-12 else 0.0

    def sortino(rets: pd.Series) -> float:
        mu = rets.mean()
        neg = rets[rets < 0]
        if len(neg) == 0:
            return np.inf
        dd = np.sqrt((neg**2).mean())
        return float(mu / dd * np.sqrt(annualize)) if dd > 1e-12 else 0.0

    def max_drawdown(rets: pd.Series) -> float:
        wealth = (1 + rets).cumprod()
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        return float(dd.min())

    def final_wealth(rets: pd.Series) -> float:
        return float((1 + rets).prod())

    return {
        "gross_sharpe": sharpe(gross_returns),
        "net_sharpe": sharpe(net_returns),
        "sortino": sortino(net_returns),
        "net_volatility": float(net_returns.std(ddof=1) * np.sqrt(annualize)),
        "max_drawdown": max_drawdown(net_returns),
        "final_wealth": final_wealth(net_returns),
        "avg_daily_turnover": float(turnover.mean()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed experiment runner
# ──────────────────────────────────────────────────────────────────────────────

def run_lstm_for_lambda(
    lam: float,
    tilt_data: dict,
    returns_train: np.ndarray,
    returns_eval: pd.DataFrame,
    n_seeds: int = N_SEEDS,
    lookback: int = LOOKBACK,
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """
    Train N_seeds LSTM models for a given λ and aggregate results.

    Parameters
    ----------
    lam : float         Interpolation parameter λ.
    tilt_data : dict    Tilting results for this λ (from kl_tilting).
    returns_train : np.ndarray  Shape (T, n) — training returns.
    returns_eval : pd.DataFrame  Shape (T_eval, n) — evaluation returns.
    n_seeds : int       Number of random seeds.
    lookback : int      Lookback window L.
    device : str        PyTorch device.
    verbose : bool      Print per-seed progress.

    Returns
    -------
    dict with aggregated results and per-seed details.
    """
    n_assets = returns_train.shape[1]
    q_weights = tilt_data["q_weights"]  # (T,)

    # Build training sequences
    X_train, Y_train = build_sequences(returns_train, lookback=lookback)
    N_seq = len(X_train)

    # Map tilt weights to sequences: sequence i corresponds to training step i+lookback
    # The portfolio return R_i = r_{i+lookback}^T w_{i+lookback}
    # Assign tilt weight of observation i+lookback (next-day return)
    T = returns_train.shape[0]
    seq_indices = np.arange(lookback, T)  # indices of next-day returns in training data
    q_seq = q_weights[seq_indices]        # tilt weights for sequence-aligned observations
    ell_seq = T * q_seq                   # likelihood ratio ℓ_λ = T * q_{t,λ}

    per_seed_metrics = []
    per_seed_weights = []

    for seed in range(n_seeds):
        if verbose and seed % 10 == 0:
            logger.info("  λ=%.2f, seed=%d/%d", lam, seed, n_seeds)

        try:
            model, _ = train_lstm_portfolio_batched(
                X_train, Y_train, ell_seq, n_assets,
                seed=seed, device=device,
            )
            eval_result = evaluate_lstm_portfolio(
                model, returns_train, returns_eval, lookback=lookback, device=device,
            )
            metrics = compute_performance_metrics(
                eval_result["gross_returns"],
                eval_result["net_returns"],
                eval_result["turnover"],
            )
            per_seed_metrics.append(metrics)
            per_seed_weights.append(eval_result["weights"])
        except Exception as e:
            logger.warning("λ=%.2f seed=%d failed: %s", lam, seed, e)
            continue

    if not per_seed_metrics:
        logger.error("All seeds failed for λ=%.2f", lam)
        return {"lambda": lam, "error": "all seeds failed"}

    # Aggregate: mean across seeds
    def agg_metric(key: str) -> float:
        vals = [m[key] for m in per_seed_metrics]
        return float(np.mean(vals))

    def std_metric(key: str) -> float:
        vals = [m[key] for m in per_seed_metrics]
        return float(np.std(vals))

    metric_keys = list(per_seed_metrics[0].keys())
    mean_metrics = {k: agg_metric(k) for k in metric_keys}
    std_metrics = {f"{k}_std": std_metric(k) for k in metric_keys}

    # Mean daily weights across seeds
    mean_weights = np.mean(per_seed_weights, axis=0)  # (T_eval, n)

    logger.info(
        "λ=%.2f: mean_gross_SR=%.3f, mean_net_SR=%.3f, "
        "mean_final_W=%.3f, mean_turnover=%.4f (n_seeds=%d)",
        lam,
        mean_metrics["gross_sharpe"],
        mean_metrics["net_sharpe"],
        mean_metrics["final_wealth"],
        mean_metrics["avg_daily_turnover"],
        len(per_seed_metrics),
    )

    return {
        "lambda": lam,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "mean_weights": mean_weights,
        "per_seed_metrics": per_seed_metrics,
        "n_valid_seeds": len(per_seed_metrics),
    }
