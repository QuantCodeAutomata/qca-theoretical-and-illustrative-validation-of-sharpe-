# Sharpe Ratio Behavior Under Directionally Correct Mean Views

Quantitative finance experiments reproducing the paper's theoretical and empirical
analysis of realized Sharpe ratio under Markowitz optimization with misspecified
mean forecasts, KL-based exponential tilting of expert views, and an LSTM
end-to-end portfolio learning framework.

## Repository Structure

```
.
├── data/          – Data fetching utilities (NSE India via yfinance)
├── src/           – Core implementations
│   ├── sharpe_theory.py   – Closed-form S(λ) theory (Experiment 1)
│   ├── kl_tilting.py      – KL-divergence exponential tilting (Experiments 2 & 3)
│   ├── markowitz.py       – Long-only max-Sharpe Markowitz (Experiment 2)
│   └── lstm_portfolio.py  – LSTM portfolio with tilted Sharpe objective (Experiment 3)
├── exp/           – Experiment runner scripts
│   ├── exp1_sharpe_theory.py
│   ├── exp2_markowitz.py
│   └── exp3_lstm.py
├── tests/         – pytest unit & integration tests
├── results/       – Saved metrics, plots, and RESULTS.md
└── .gitignore
```

## Experiments

### Experiment 1 – Sharpe Theory Validation
Validates the closed-form S(λ) formula and the global-concavity condition (A+3B≥0)
for two NSE datasets. Uses the interpolation path ν_λ=(1-λ)ν+λμ on [0,1].

### Experiment 2 – Markowitz with KL-Tilted Views
KL-minimization exponential tilting to match expert view means; long-only max-Sharpe
Markowitz portfolios evaluated over 2024–2025 with 20 bps transaction costs.

### Experiment 3 – LSTM Portfolio Learning
Single-layer LSTM (16 hidden units, softmax output) trained with a view-adjusted
weighted Sharpe objective. 100 random seeds per λ; evaluated over 2024–2025.

## Setup

```bash
pip install -r requirements.txt   # or install dependencies individually
```

Required packages: numpy, pandas, scipy, matplotlib, seaborn, torch, yfinance.

## Running Experiments

```bash
# Experiment 1 (theory; fastest)
python exp/exp1_sharpe_theory.py

# Experiment 2 (Markowitz)
python exp/exp2_markowitz.py

# Experiment 3 (LSTM; slowest — 100 seeds × 11 λ values × 300 iterations)
python exp/exp3_lstm.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Implementation Assumptions

| Assumption | Choice |
|---|---|
| Price type | Adjusted closing prices (yfinance default) |
| Dataset 1 eval window | 2023 (not specified in paper) |
| Dataset 2 split | 2006–2020 estimation, 2021–2025 evaluation |
| Covariance in Exp 1 | Ω=Σ (simplified theorem case) |
| Risk-free rate | 0 (as stated in paper) |
| Annualization | ×252 mean, ×√252 vol |
| LSTM lookback L | 20 days |
| LSTM optimizer | Adam, lr=1e-3 warm-up + cosine annealing |
| Initial trade cost | Charged (from all-cash position) |
| Corporate actions | Adjusted prices assumed |

## Results

See [results/RESULTS.md](results/RESULTS.md) for detailed metrics and figures.
