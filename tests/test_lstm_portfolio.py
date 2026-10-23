"""
Tests for src/lstm_portfolio.py.

Validates:
- LSTMPortfolio forward pass: output shape and long-only simplex constraint
- build_sequences: correct shapes and alignment
- tilted_sharpe_loss: scalar output and gradient flow
- train_lstm_portfolio_batched: training runs without error, returns loss list
- evaluate_lstm_portfolio: correct return shapes, net <= gross on average
- compute_performance_metrics: Sharpe, Sortino, max-drawdown properties
- run_lstm_for_lambda: aggregation over 2 seeds on tiny synthetic data
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.lstm_portfolio import (
    LSTMPortfolio,
    build_sequences,
    tilted_sharpe_loss,
    train_lstm_portfolio_batched,
    evaluate_lstm_portfolio,
    compute_performance_metrics,
    run_lstm_for_lambda,
    LOOKBACK,
    HIDDEN_SIZE,
)

# ── Shared constants ──────────────────────────────────────────────────────────

N_ASSETS = 4
N_DAYS_TRAIN = 80   # enough for LOOKBACK_TEST + sequences
N_DAYS_EVAL = 25
LOOKBACK_TEST = 5   # small window for fast tests


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def returns_train(rng: np.random.Generator) -> np.ndarray:
    return (rng.standard_normal((N_DAYS_TRAIN, N_ASSETS)) * 0.01).astype(np.float64)


@pytest.fixture()
def returns_eval_df(rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=N_DAYS_EVAL, freq="B")
    return pd.DataFrame(
        rng.standard_normal((N_DAYS_EVAL, N_ASSETS)) * 0.01,
        index=idx,
        columns=[f"A{i}" for i in range(N_ASSETS)],
    )


@pytest.fixture()
def model() -> LSTMPortfolio:
    torch.manual_seed(0)
    return LSTMPortfolio(n_assets=N_ASSETS, hidden_size=4)


@pytest.fixture()
def uniform_tilt_data(returns_train: np.ndarray) -> dict:
    T = len(returns_train)
    return {
        "lambda": 0.0,
        "q_weights": np.ones(T) / T,
        "mu_lambda": returns_train.mean(axis=0),
        "eta": np.zeros(N_ASSETS),
    }


# ── LSTMPortfolio ─────────────────────────────────────────────────────────────

class TestLSTMPortfolio:
    def test_output_shape(self, model: LSTMPortfolio) -> None:
        x = torch.randn(LOOKBACK_TEST, N_ASSETS)
        w = model(x)
        assert w.shape == (N_ASSETS,), "Output must be a 1-D weight vector"

    def test_weights_sum_to_one(self, model: LSTMPortfolio) -> None:
        x = torch.randn(LOOKBACK_TEST, N_ASSETS)
        w = model(x)
        assert abs(w.sum().item() - 1.0) < 1e-5, "Weights must sum to 1.0"

    def test_weights_non_negative(self, model: LSTMPortfolio) -> None:
        """Softmax output must be non-negative (long-only constraint)."""
        x = torch.randn(LOOKBACK_TEST, N_ASSETS)
        w = model(x)
        assert (w >= 0).all(), "All weights must be ≥ 0 (softmax output)"

    def test_different_seeds_differ(self) -> None:
        x = torch.randn(LOOKBACK_TEST, N_ASSETS)
        torch.manual_seed(0)
        m0 = LSTMPortfolio(n_assets=N_ASSETS, hidden_size=4)
        torch.manual_seed(99)
        m99 = LSTMPortfolio(n_assets=N_ASSETS, hidden_size=4)
        assert not torch.allclose(m0(x), m99(x)), "Different seeds should give different weights"

    def test_single_asset_weight_is_one(self) -> None:
        """Single-asset portfolio must always allocate 100%."""
        torch.manual_seed(1)
        m = LSTMPortfolio(n_assets=1, hidden_size=4)
        x = torch.randn(LOOKBACK_TEST, 1)
        w = m(x)
        assert abs(w.item() - 1.0) < 1e-5, "Single-asset weight must be 1.0"

    def test_module_exported_constants(self) -> None:
        assert isinstance(LOOKBACK, int) and LOOKBACK > 0
        assert isinstance(HIDDEN_SIZE, int) and HIDDEN_SIZE > 0


# ── build_sequences ───────────────────────────────────────────────────────────

class TestBuildSequences:
    def test_shapes(self, returns_train: np.ndarray) -> None:
        X, Y = build_sequences(returns_train, lookback=LOOKBACK_TEST)
        expected_n = N_DAYS_TRAIN - LOOKBACK_TEST
        assert X.shape == (expected_n, LOOKBACK_TEST, N_ASSETS)
        assert Y.shape == (expected_n, N_ASSETS)

    def test_alignment(self, returns_train: np.ndarray) -> None:
        X, Y = build_sequences(returns_train, lookback=LOOKBACK_TEST)
        # The first label should be returns_train[LOOKBACK_TEST]
        np.testing.assert_array_equal(Y[0], returns_train[LOOKBACK_TEST])
        # The first input window should be returns_train[0:LOOKBACK_TEST]
        np.testing.assert_array_equal(X[0], returns_train[:LOOKBACK_TEST])

    def test_lookback_one(self, returns_train: np.ndarray) -> None:
        X, Y = build_sequences(returns_train, lookback=1)
        assert X.shape == (N_DAYS_TRAIN - 1, 1, N_ASSETS)
        assert Y.shape == (N_DAYS_TRAIN - 1, N_ASSETS)

    def test_lookback_T_minus_one(self, returns_train: np.ndarray) -> None:
        L = N_DAYS_TRAIN - 1
        X, Y = build_sequences(returns_train, lookback=L)
        assert X.shape == (1, L, N_ASSETS)
        assert Y.shape == (1, N_ASSETS)

    def test_dtype_is_float64(self, returns_train: np.ndarray) -> None:
        X, Y = build_sequences(returns_train, lookback=LOOKBACK_TEST)
        assert X.dtype == np.float64
        assert Y.dtype == np.float64


# ── tilted_sharpe_loss ────────────────────────────────────────────────────────

class TestTiltedSharpeLoss:
    def test_returns_scalar_tensor(self) -> None:
        r = torch.randn(50)
        w = torch.ones(50)
        loss = tilted_sharpe_loss(r, w)
        assert loss.dim() == 0, "tilted_sharpe_loss must return a scalar"

    def test_negative_for_uniform_positive_returns(self) -> None:
        """Sharpe is positive ⟹ loss (negative Sharpe) is negative."""
        r = torch.full((50,), 0.01)
        w = torch.ones(50)
        loss = tilted_sharpe_loss(r, w)
        assert loss.item() < 0, "Expected negative loss for constant positive returns"

    def test_positive_for_uniform_negative_returns(self) -> None:
        """Negative Sharpe ⟹ loss is positive."""
        r = torch.full((50,), -0.01)
        w = torch.ones(50)
        loss = tilted_sharpe_loss(r, w)
        assert loss.item() > 0, "Expected positive loss for constant negative returns"

    def test_gradient_flows(self) -> None:
        r = torch.randn(30, requires_grad=True)
        w = torch.ones(30)
        loss = tilted_sharpe_loss(r, w)
        loss.backward()
        assert r.grad is not None, "Gradient must propagate through tilted_sharpe_loss"
        assert r.grad.shape == (30,)

    def test_finite_for_random_inputs(self) -> None:
        rng = np.random.default_rng(7)
        r = torch.tensor(rng.standard_normal(100) * 0.01, dtype=torch.float32)
        w = torch.ones(100)
        loss = tilted_sharpe_loss(r, w)
        assert np.isfinite(loss.item()), "Loss must be finite for typical returns"

    def test_equal_weighted_vs_non_uniform(self) -> None:
        """Non-uniform weights change the loss value."""
        rng = np.random.default_rng(99)
        r = torch.tensor(rng.standard_normal(60) * 0.01 + 0.001, dtype=torch.float32)
        w_uniform = torch.ones(60)
        w_skewed = torch.cat([torch.full((30,), 0.1), torch.full((30,), 1.9)])
        loss_u = tilted_sharpe_loss(r, w_uniform).item()
        loss_s = tilted_sharpe_loss(r, w_skewed).item()
        assert loss_u != loss_s, "Different tilt weights must give different losses"


# ── train_lstm_portfolio_batched ──────────────────────────────────────────────

class TestTrainLSTMBatched:
    def _build_inputs(self, returns_train: np.ndarray):
        X, Y = build_sequences(returns_train, lookback=LOOKBACK_TEST)
        ell = np.ones(len(X), dtype=np.float64)
        return X, Y, ell

    def test_runs_without_error(self, returns_train: np.ndarray) -> None:
        X, Y, ell = self._build_inputs(returns_train)
        model, losses = train_lstm_portfolio_batched(
            X, Y, ell, N_ASSETS, seed=0, n_iter=5, n_warmup=2,
            lr_warmup=1e-3, lr_min=1e-5, device="cpu",
        )
        assert isinstance(losses, list) and len(losses) == 5
        assert all(np.isfinite(l) for l in losses), "All losses must be finite"

    def test_returns_lstm_portfolio(self, returns_train: np.ndarray) -> None:
        X, Y, ell = self._build_inputs(returns_train)
        model, _ = train_lstm_portfolio_batched(
            X, Y, ell, N_ASSETS, seed=1, n_iter=3, n_warmup=1, device="cpu",
        )
        assert isinstance(model, LSTMPortfolio)

    def test_different_seeds_yield_different_models(self, returns_train: np.ndarray) -> None:
        X, Y, ell = self._build_inputs(returns_train)
        m0, _ = train_lstm_portfolio_batched(X, Y, ell, N_ASSETS, seed=0, n_iter=3, n_warmup=1, device="cpu")
        m1, _ = train_lstm_portfolio_batched(X, Y, ell, N_ASSETS, seed=1, n_iter=3, n_warmup=1, device="cpu")
        x = torch.randn(LOOKBACK_TEST, N_ASSETS)
        assert not torch.allclose(m0(x), m1(x)), "Different seeds must produce different models"

    def test_loss_length_matches_n_iter(self, returns_train: np.ndarray) -> None:
        X, Y, ell = self._build_inputs(returns_train)
        n_iter = 8
        _, losses = train_lstm_portfolio_batched(
            X, Y, ell, N_ASSETS, seed=0, n_iter=n_iter, n_warmup=2, device="cpu",
        )
        assert len(losses) == n_iter


# ── evaluate_lstm_portfolio ───────────────────────────────────────────────────

class TestEvaluateLSTM:
    def test_output_shapes(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame, model: LSTMPortfolio
    ) -> None:
        result = evaluate_lstm_portfolio(
            model, returns_train, returns_eval_df, lookback=LOOKBACK_TEST
        )
        assert result["weights"].shape == (N_DAYS_EVAL, N_ASSETS)
        assert len(result["gross_returns"]) == N_DAYS_EVAL
        assert len(result["net_returns"]) == N_DAYS_EVAL
        assert len(result["turnover"]) == N_DAYS_EVAL

    def test_weights_long_only_and_sum_to_one(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame, model: LSTMPortfolio
    ) -> None:
        result = evaluate_lstm_portfolio(
            model, returns_train, returns_eval_df, lookback=LOOKBACK_TEST
        )
        w = result["weights"]
        assert (w >= -1e-7).all(), "All weights must be non-negative"
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)

    def test_net_at_most_gross_on_average(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame, model: LSTMPortfolio
    ) -> None:
        result = evaluate_lstm_portfolio(
            model, returns_train, returns_eval_df, lookback=LOOKBACK_TEST
        )
        assert result["net_returns"].mean() <= result["gross_returns"].mean() + 1e-10

    def test_turnover_non_negative(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame, model: LSTMPortfolio
    ) -> None:
        result = evaluate_lstm_portfolio(
            model, returns_train, returns_eval_df, lookback=LOOKBACK_TEST
        )
        assert (result["turnover"] >= -1e-10).all(), "Turnover must be ≥ 0"

    def test_returns_are_pandas_series(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame, model: LSTMPortfolio
    ) -> None:
        result = evaluate_lstm_portfolio(
            model, returns_train, returns_eval_df, lookback=LOOKBACK_TEST
        )
        assert isinstance(result["gross_returns"], pd.Series)
        assert isinstance(result["net_returns"], pd.Series)
        assert isinstance(result["turnover"], pd.Series)


# ── compute_performance_metrics ───────────────────────────────────────────────

class TestComputePerformanceMetrics:
    def _series(self, vals: list) -> pd.Series:
        idx = pd.date_range("2024-01-02", periods=len(vals), freq="B")
        return pd.Series(vals, index=idx)

    def test_required_keys_present(self) -> None:
        rets = self._series([0.001] * 50)
        to = self._series([0.01] * 50)
        m = compute_performance_metrics(rets, rets, to)
        for key in ["gross_sharpe", "net_sharpe", "sortino", "net_volatility",
                    "max_drawdown", "final_wealth", "avg_daily_turnover"]:
            assert key in m, f"Key '{key}' missing from metrics dict"

    def test_positive_sharpe_for_positive_returns(self) -> None:
        # Use normally distributed returns with positive mean so variance > 0
        rng = np.random.default_rng(10)
        vals = list(rng.standard_normal(252) * 0.005 + 0.002)  # mean > 0, std > 0
        rets = self._series(vals)
        to = self._series([0.01] * 252)
        m = compute_performance_metrics(rets, rets, to)
        assert m["gross_sharpe"] > 0
        assert m["net_sharpe"] > 0

    def test_zero_returns_sharpe_is_zero(self) -> None:
        rets = self._series([0.0] * 100)
        to = self._series([0.0] * 100)
        m = compute_performance_metrics(rets, rets, to)
        assert m["gross_sharpe"] == 0.0

    def test_max_drawdown_is_non_positive(self) -> None:
        rng = np.random.default_rng(1)
        rets = self._series(list(rng.standard_normal(200) * 0.01))
        to = self._series([0.01] * 200)
        m = compute_performance_metrics(rets, rets, to)
        assert m["max_drawdown"] <= 0.0, "Max drawdown must be ≤ 0"

    def test_final_wealth_grows_for_positive_returns(self) -> None:
        rets = self._series([0.001] * 100)
        to = self._series([0.01] * 100)
        m = compute_performance_metrics(rets, rets, to)
        assert m["final_wealth"] > 1.0

    def test_final_wealth_shrinks_for_negative_returns(self) -> None:
        rets = self._series([-0.001] * 100)
        to = self._series([0.01] * 100)
        m = compute_performance_metrics(rets, rets, to)
        assert m["final_wealth"] < 1.0

    def test_volatility_is_annualized_std(self) -> None:
        """Net volatility = σ * sqrt(252)."""
        rets = self._series(list(np.random.default_rng(2).standard_normal(100) * 0.01))
        to = self._series([0.01] * 100)
        m = compute_performance_metrics(rets, rets, to)
        expected_vol = rets.std(ddof=1) * np.sqrt(252)
        assert abs(m["net_volatility"] - expected_vol) < 1e-10


# ── run_lstm_for_lambda (integration, tiny data) ──────────────────────────────

class TestRunLSTMForLambda:
    def test_returns_expected_keys(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0,
            tilt_data=uniform_tilt_data,
            returns_train=returns_train,
            returns_eval=returns_eval_df,
            n_seeds=2,
            lookback=LOOKBACK_TEST,
            device="cpu",
            verbose=False,
        )
        for key in ["lambda", "mean_metrics", "std_metrics", "mean_weights",
                    "per_seed_metrics", "n_valid_seeds"]:
            assert key in result, f"Key '{key}' missing from result"

    def test_n_valid_seeds_equals_n_seeds(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0, tilt_data=uniform_tilt_data,
            returns_train=returns_train, returns_eval=returns_eval_df,
            n_seeds=2, lookback=LOOKBACK_TEST, device="cpu", verbose=False,
        )
        assert result["n_valid_seeds"] == 2

    def test_mean_weights_shape(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0, tilt_data=uniform_tilt_data,
            returns_train=returns_train, returns_eval=returns_eval_df,
            n_seeds=2, lookback=LOOKBACK_TEST, device="cpu", verbose=False,
        )
        assert result["mean_weights"].shape == (N_DAYS_EVAL, N_ASSETS)

    def test_mean_metrics_keys(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0, tilt_data=uniform_tilt_data,
            returns_train=returns_train, returns_eval=returns_eval_df,
            n_seeds=2, lookback=LOOKBACK_TEST, device="cpu", verbose=False,
        )
        for key in ["gross_sharpe", "net_sharpe", "sortino", "final_wealth",
                    "max_drawdown", "avg_daily_turnover"]:
            assert key in result["mean_metrics"], f"'{key}' missing from mean_metrics"

    def test_per_seed_metrics_length(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0, tilt_data=uniform_tilt_data,
            returns_train=returns_train, returns_eval=returns_eval_df,
            n_seeds=2, lookback=LOOKBACK_TEST, device="cpu", verbose=False,
        )
        assert len(result["per_seed_metrics"]) == 2

    def test_mean_weights_long_only_sum_to_one(
        self, returns_train: np.ndarray, returns_eval_df: pd.DataFrame,
        uniform_tilt_data: dict
    ) -> None:
        result = run_lstm_for_lambda(
            lam=0.0, tilt_data=uniform_tilt_data,
            returns_train=returns_train, returns_eval=returns_eval_df,
            n_seeds=2, lookback=LOOKBACK_TEST, device="cpu", verbose=False,
        )
        w = result["mean_weights"]
        assert (w >= -1e-6).all(), "Mean weights must be ≥ 0"
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-4)
