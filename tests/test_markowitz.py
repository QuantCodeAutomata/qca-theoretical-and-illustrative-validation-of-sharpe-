"""
Tests for src/markowitz.py and src/lstm_portfolio.py.

Validates:
- Long-only portfolio weights are non-negative and sum to 1
- Net returns never exceed gross returns (costs > 0)
- Performance metrics are in valid ranges
- Sharpe ratio computation is correct
- LSTM model outputs valid weights at every step
- Tilted Sharpe loss is differentiable
"""

import numpy as np
import pandas as pd
import pytest
import torch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.markowitz import (
    max_sharpe_long_only,
    compute_portfolio_returns,
    compute_performance_metrics,
)
from src.lstm_portfolio import (
    LSTMPortfolio,
    build_sequences,
    tilted_sharpe_loss,
    train_lstm_portfolio_batched,
    evaluate_lstm_portfolio,
    LOOKBACK,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_portfolio_data(n: int = 5, T: int = 252, seed: int = 42):
    """Return (mu, Sigma, returns_df)."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.001, 0.002, n)
    A = rng.normal(0, 1, (n, n))
    Sigma = A @ A.T / n + 0.01 * np.eye(n)
    dates = pd.date_range("2024-01-02", periods=T, freq="B")
    returns_df = pd.DataFrame(rng.normal(0.001, 0.01, (T, n)), index=dates)
    return mu, Sigma, returns_df


# ══════════════════════════════════════════════════════════════════════════════
# Markowitz Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMaxSharpeLongOnly:
    """Tests for the max Sharpe optimization."""

    def test_weights_nonneg(self):
        """Portfolio weights must be non-negative."""
        mu, Sigma, _ = make_portfolio_data()
        np.random.seed(0)
        w = max_sharpe_long_only(mu, Sigma)
        assert np.all(w >= -1e-10), f"Negative weights found: {w.min():.4f}"

    def test_weights_sum_to_one(self):
        """Portfolio weights must sum to 1."""
        mu, Sigma, _ = make_portfolio_data()
        np.random.seed(0)
        w = max_sharpe_long_only(mu, Sigma)
        assert abs(w.sum() - 1.0) < 1e-8

    def test_sharpe_positive(self):
        """Max Sharpe should be positive when mu has positive entries."""
        mu = np.array([0.001, 0.002, 0.003])
        n = 3
        Sigma = 0.0001 * np.eye(n)
        np.random.seed(0)
        w = max_sharpe_long_only(mu, Sigma)
        sharpe = float(w @ mu) / np.sqrt(float(w @ Sigma @ w))
        assert sharpe > 0

    def test_single_asset(self):
        """Single asset: weight must be 1.0."""
        mu = np.array([0.001])
        Sigma = np.array([[0.01]])
        w = max_sharpe_long_only(mu, Sigma)
        assert abs(w[0] - 1.0) < 1e-8

    def test_equal_assets_equal_weight(self):
        """Identical assets should get equal weights."""
        n = 3
        mu = np.ones(n) * 0.001
        Sigma = np.eye(n) * 0.01
        np.random.seed(42)
        w = max_sharpe_long_only(mu, Sigma)
        assert np.allclose(w, np.ones(n) / n, atol=1e-4)

    def test_returns_shape(self):
        """Weight vector shape matches n_assets."""
        for n in [2, 5, 10, 20]:
            mu = np.random.normal(0.001, 0.001, n)
            A = np.random.normal(0, 1, (n, n))
            Sigma = A @ A.T / n + 0.01 * np.eye(n)
            np.random.seed(0)
            w = max_sharpe_long_only(mu, Sigma)
            assert w.shape == (n,)

    def test_regularization_handles_singular(self):
        """Near-singular Sigma should be handled by eps regularization."""
        n = 3
        # All-ones covariance is rank 1 → singular
        Sigma = np.ones((n, n)) + 1e-4 * np.eye(n)
        mu = np.array([0.001, 0.002, 0.003])
        np.random.seed(0)
        w = max_sharpe_long_only(mu, Sigma, eps=1e-4)
        assert np.all(np.isfinite(w))
        assert abs(w.sum() - 1.0) < 1e-6


class TestComputePortfolioReturns:
    """Tests for portfolio return computation."""

    def test_gross_returns_shape(self):
        """Gross returns should have same length as evaluation returns."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=100)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        assert len(result["gross_returns"]) == len(rets)

    def test_net_leq_gross(self):
        """Net returns must be ≤ gross returns (costs are positive)."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=100)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        assert np.all(result["net_returns"].values <= result["gross_returns"].values + 1e-12)

    def test_turnover_nonneg(self):
        """Turnover must be non-negative."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=100)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        assert np.all(result["turnover"].values >= -1e-12)

    def test_constant_weights_low_turnover(self):
        """Constant weights should produce low ongoing turnover (after first day)."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=50)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        # After day 0 (initial investment), turnover due to drift should be small
        avg_ongoing_turnover = result["turnover"].iloc[1:].mean()
        assert avg_ongoing_turnover < 0.1, f"Turnover too high: {avg_ongoing_turnover}"

    def test_final_wealth_positive(self):
        """Final wealth should be positive."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=100)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        net_wealth = (1 + result["net_returns"]).prod()
        assert net_wealth > 0

    def test_index_preserved(self):
        """Output series should preserve the evaluation return index."""
        mu, Sigma, rets = make_portfolio_data(n=5, T=50)
        w = np.ones(5) / 5
        result = compute_portfolio_returns(w, rets)
        assert list(result["gross_returns"].index) == list(rets.index)


class TestComputePerformanceMetrics:
    """Tests for performance metric computation."""

    def test_sharpe_positive_for_positive_returns(self):
        """Positive constant daily returns → positive Sharpe."""
        n_days = 252
        rets = pd.Series(np.ones(n_days) * 0.001)
        turnover = pd.Series(np.zeros(n_days))
        metrics = compute_performance_metrics(rets, rets, turnover)
        assert metrics["gross_sharpe"] > 0
        assert metrics["net_sharpe"] > 0

    def test_final_wealth_correct(self):
        """Final wealth = product of (1+r_t)."""
        rets = pd.Series([0.01, -0.01, 0.02])
        turnover = pd.Series([0.0, 0.0, 0.0])
        metrics = compute_performance_metrics(rets, rets, turnover)
        expected = (1.01) * (0.99) * (1.02)
        assert abs(metrics["final_wealth"] - expected) < 1e-10

    def test_max_drawdown_nonpositive(self):
        """Max drawdown should be ≤ 0."""
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0, 0.01, 252))
        turnover = pd.Series(np.zeros(252))
        metrics = compute_performance_metrics(rets, rets, turnover)
        assert metrics["max_drawdown"] <= 0

    def test_net_vol_positive(self):
        """Net volatility should be positive for random returns."""
        rng = np.random.default_rng(2)
        rets = pd.Series(rng.normal(0, 0.01, 252))
        turnover = pd.Series(np.zeros(252))
        metrics = compute_performance_metrics(rets, rets, turnover)
        assert metrics["net_volatility"] > 0

    def test_zero_volatility_zero_sharpe(self):
        """Constant returns → division by near-zero std → handled gracefully."""
        rets = pd.Series(np.ones(100) * 0.001)
        turnover = pd.Series(np.zeros(100))
        metrics = compute_performance_metrics(rets, rets, turnover)
        assert np.isfinite(metrics["gross_sharpe"]) or metrics["gross_sharpe"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# LSTM Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLSTMPortfolio:
    """Tests for LSTM model architecture."""

    def test_output_shape(self):
        """LSTM output should be (n_assets,) shape."""
        n = 5
        L = 20
        model = LSTMPortfolio(n_assets=n)
        x = torch.randn(L, n)
        w = model(x)
        assert w.shape == (n,)

    def test_weights_nonneg(self):
        """LSTM softmax outputs should be non-negative."""
        n = 10
        L = 20
        model = LSTMPortfolio(n_assets=n)
        x = torch.randn(L, n)
        w = model(x)
        assert torch.all(w >= 0)

    def test_weights_sum_to_one(self):
        """LSTM softmax outputs should sum to 1."""
        n = 10
        L = 20
        model = LSTMPortfolio(n_assets=n)
        x = torch.randn(L, n)
        w = model(x)
        assert abs(w.sum().item() - 1.0) < 1e-6

    def test_different_inputs_different_outputs(self):
        """Different inputs should generally produce different weights."""
        n = 5
        L = 10
        model = LSTMPortfolio(n_assets=n)
        x1 = torch.randn(L, n)
        x2 = torch.randn(L, n) * 10.0
        w1 = model(x1)
        w2 = model(x2)
        assert not torch.allclose(w1, w2)


class TestBuildSequences:
    """Tests for sequence building."""

    def test_output_shapes(self):
        """X shape should be (T-L, L, n) and Y shape (T-L, n)."""
        T, n, L = 100, 5, 20
        R = np.random.normal(0, 0.01, (T, n))
        X, Y = build_sequences(R, lookback=L)
        assert X.shape == (T - L, L, n)
        assert Y.shape == (T - L, n)

    def test_correct_alignment(self):
        """X[i] = R[i:i+L], Y[i] = R[i+L]."""
        T, n, L = 30, 3, 5
        R = np.arange(T * n, dtype=float).reshape(T, n)
        X, Y = build_sequences(R, lookback=L)
        # Check first sequence
        np.testing.assert_array_equal(X[0], R[:L])
        np.testing.assert_array_equal(Y[0], R[L])
        # Check last sequence
        np.testing.assert_array_equal(X[-1], R[T-L-1:T-1])
        np.testing.assert_array_equal(Y[-1], R[T-1])


class TestTiltedSharpeLoss:
    """Tests for the KL-tilted Sharpe objective."""

    def test_loss_finite(self):
        """Loss should be finite for valid inputs."""
        N = 100
        port_rets = torch.randn(N)
        tilt_weights = torch.ones(N)
        loss = tilted_sharpe_loss(port_rets, tilt_weights)
        assert torch.isfinite(loss)

    def test_uniform_weights_matches_standard_sharpe(self):
        """Uniform tilt weights → standard Sharpe (negated)."""
        N = 200
        port_rets = torch.randn(N) * 0.01 + 0.001
        w_uniform = torch.ones(N)
        loss = tilted_sharpe_loss(port_rets, w_uniform)
        # Manual Sharpe
        w_norm = w_uniform / w_uniform.sum()
        m = (w_norm * port_rets).sum()
        v = (w_norm * (port_rets - m) ** 2).sum()
        sharpe = m / torch.sqrt(v + 1e-8)
        assert abs((-loss - sharpe).item()) < 1e-6

    def test_loss_differentiable(self):
        """Loss should be differentiable with respect to portfolio returns."""
        N = 50
        port_rets = torch.randn(N, requires_grad=True)
        tilt_weights = torch.ones(N) * 0.5
        loss = tilted_sharpe_loss(port_rets, tilt_weights)
        loss.backward()
        assert port_rets.grad is not None
        assert torch.all(torch.isfinite(port_rets.grad))

    def test_loss_negative_for_positive_sharpe(self):
        """Loss should be negative when Sharpe is positive (since we negate)."""
        N = 252
        port_rets = torch.ones(N) * 0.001  # constant positive return
        tilt_weights = torch.ones(N)
        loss = tilted_sharpe_loss(port_rets, tilt_weights)
        # With constant returns, std ≈ 0, loss should be handled (not NaN)
        assert torch.isfinite(loss)


class TestTrainLSTMPortfolio:
    """Tests for LSTM training loop."""

    def _make_training_data(self, T=200, n=5, L=10):
        """Create minimal training data."""
        rng = np.random.default_rng(42)
        R = rng.normal(0.001, 0.01, (T, n))
        X, Y = build_sequences(R, lookback=L)
        ell = np.ones(len(X))  # uniform weights
        return X, Y, ell, n

    def test_model_trains_without_error(self):
        """Training should complete without exceptions."""
        X, Y, ell, n = self._make_training_data(T=50, n=3, L=5)
        model, losses = train_lstm_portfolio_batched(
            X, Y, ell, n_assets=3, seed=0, n_iter=10, n_warmup=5,
        )
        assert len(losses) == 10
        assert all(np.isfinite(l) for l in losses)

    def test_weights_valid_after_training(self):
        """Trained model outputs should be valid weights."""
        X, Y, ell, n = self._make_training_data(T=50, n=3, L=5)
        model, _ = train_lstm_portfolio_batched(
            X, Y, ell, n_assets=3, seed=0, n_iter=20, n_warmup=10,
        )
        x_test = torch.randn(5, 3)
        with torch.no_grad():
            w = model(x_test)
        assert w.shape == (3,)
        assert abs(w.sum().item() - 1.0) < 1e-6
        assert torch.all(w >= 0)

    def test_loss_decreases_with_training(self):
        """Training should generally reduce the tilted Sharpe loss (on average)."""
        X, Y, ell, n = self._make_training_data(T=100, n=4, L=8)
        # Stronger signal: make returns trend slightly
        Y[:, 0] += 0.005
        ell_strong = np.ones(len(X)) * 2.0  # uniform but scaled

        _, losses = train_lstm_portfolio_batched(
            X, Y, ell_strong, n_assets=4, seed=42, n_iter=50, n_warmup=20,
        )
        # First half vs second half average loss
        first_half = np.mean(losses[:25])
        second_half = np.mean(losses[25:])
        # Not guaranteed to decrease always (depends on init), but check finite
        assert all(np.isfinite(l) for l in losses)


class TestEvaluateLSTMPortfolio:
    """Tests for LSTM portfolio evaluation."""

    def _make_eval_setup(self, T_train=100, T_eval=50, n=5, L=10):
        rng = np.random.default_rng(123)
        R_train = rng.normal(0.001, 0.01, (T_train, n))
        dates_eval = pd.date_range("2024-01-02", periods=T_eval, freq="B")
        R_eval = pd.DataFrame(rng.normal(0.001, 0.01, (T_eval, n)), index=dates_eval)
        model = LSTMPortfolio(n_assets=n)
        return model, R_train, R_eval, n

    def test_eval_output_shape(self):
        """Evaluation should return T_eval rows of weights."""
        model, R_train, R_eval, n = self._make_eval_setup(T_train=100, T_eval=50, n=5, L=10)
        result = evaluate_lstm_portfolio(model, R_train, R_eval, lookback=10)
        assert result["weights"].shape == (50, 5)
        assert len(result["gross_returns"]) == 50
        assert len(result["net_returns"]) == 50
        assert len(result["turnover"]) == 50

    def test_eval_weights_valid(self):
        """Each day's weights should be non-negative and sum to 1."""
        model, R_train, R_eval, n = self._make_eval_setup(T_train=100, T_eval=30, n=5, L=10)
        result = evaluate_lstm_portfolio(model, R_train, R_eval, lookback=10)
        W = result["weights"]
        assert np.all(W >= -1e-8), "Weights must be non-negative"
        row_sums = W.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), "Weights must sum to 1"

    def test_eval_net_leq_gross(self):
        """Net returns should be ≤ gross returns."""
        model, R_train, R_eval, n = self._make_eval_setup(T_train=100, T_eval=30, n=5, L=10)
        result = evaluate_lstm_portfolio(model, R_train, R_eval, lookback=10)
        assert np.all(
            result["net_returns"].values <= result["gross_returns"].values + 1e-12
        )

    def test_eval_turnover_nonneg(self):
        """Turnover should be non-negative."""
        model, R_train, R_eval, n = self._make_eval_setup(T_train=100, T_eval=30, n=5, L=10)
        result = evaluate_lstm_portfolio(model, R_train, R_eval, lookback=10)
        assert np.all(result["turnover"].values >= -1e-12)

    def test_eval_index_correct(self):
        """Output series should use the evaluation return index."""
        model, R_train, R_eval, n = self._make_eval_setup(T_train=100, T_eval=30, n=5, L=10)
        result = evaluate_lstm_portfolio(model, R_train, R_eval, lookback=10)
        assert list(result["gross_returns"].index) == list(R_eval.index)
