"""
Data fetching module for NSE India daily stock prices.

NOTE: The Massive REST API (based on Polygon.io) covers US equity markets.
NSE India stocks are NOT available via the Massive API. As an implementation
assumption, we use yfinance with the '.NS' (NSE) suffix to download Indian
equity data. This is documented as a deviation from the Massive-only policy
due to market coverage limitations.

Implementation assumption (price adjustment): yfinance returns adjusted closing
prices by default ('auto_adjust=True'). We use adjusted prices to account for
splits and dividends, consistent with the paper's 'closing prices' reference.
This choice is documented per the methodology note.

Implementation assumption (missing data): No imputation is applied. Stocks
with missing data are dropped if they lack complete coverage.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Fixed 20-stock NSE India universe (from paper)
NSE_UNIVERSE_20 = [
    "RELIANCE", "TCS", "HDFCBANK", "AXISBANK", "INFY",
    "HINDUNILVR", "BHARTIARTL", "ITC", "SBIN", "LT",
    "HCLTECH", "KOTAKBANK", "ICICIBANK", "ASIANPAINT", "NTPC",
    "ADANIENT", "MARUTI", "SUNPHARMA", "ULTRACEMCO", "TITAN",
]

# Smaller 4-stock universe for Exp 1 Dataset 1
NSE_UNIVERSE_4 = ["BHARTIARTL", "KOTAKBANK", "SBIN", "MARUTI"]


def nse_ticker(symbol: str) -> str:
    """Append NSE suffix to a stock symbol."""
    return f"{symbol}.NS"


def fetch_nse_prices(
    symbols: list[str],
    start: str,
    end: str,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily adjusted closing prices for NSE India stocks.

    Parameters
    ----------
    symbols : list of str
        NSE stock symbols without suffix (e.g., 'RELIANCE').
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    cache_dir : str, optional
        Directory to cache downloaded data as Parquet. If None, no caching.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices, indexed by date,
        columns are original symbol names (without .NS suffix).
        Dates are timezone-naive.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tag = f"{'_'.join(sorted(symbols))}_{start}_{end}".replace("-", "")
        cache_path = os.path.join(cache_dir, f"prices_{tag[:80]}.parquet")
        if os.path.exists(cache_path):
            logger.info("Loading cached prices from %s", cache_path)
            df = pd.read_parquet(cache_path)
            return df

    tickers = [nse_ticker(s) for s in symbols]
    logger.info("Downloading %d NSE stocks via yfinance (%s to %s)...", len(tickers), start, end)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # Remove timezone info
    prices.index = prices.index.tz_localize(None) if prices.index.tz is not None else prices.index

    # Rename columns from SYMBOL.NS to SYMBOL
    prices.columns = [c.replace(".NS", "") for c in prices.columns]

    # Drop rows where all prices are NaN
    prices = prices.dropna(how="all")

    logger.info("Downloaded price matrix: %d rows × %d columns", *prices.shape)

    if cache_path is not None:
        prices.to_parquet(cache_path)
        logger.info("Cached prices to %s", cache_path)

    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily simple returns r_t = P_t / P_{t-1} - 1.

    Prices are first filtered to rows where ALL stocks have a valid price
    (complete-panel rule applied at the price level), then pct_change is taken.
    This drops spurious non-trading dates that yfinance includes for only a
    subset of stocks (e.g. a date with price for 1/20 stocks).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of closing prices.

    Returns
    -------
    pd.DataFrame
        Daily simple return DataFrame (first return row dropped automatically).
    """
    # Drop price rows where ANY stock is missing (enforces complete-panel at price level)
    prices_clean = prices.dropna(how="any")
    n_dropped = len(prices) - len(prices_clean)
    if n_dropped > 0:
        logger.info("Dropped %d price rows with any NaN before computing returns.", n_dropped)
    returns = prices_clean.pct_change(fill_method=None).dropna(how="all")
    return returns


def get_complete_panel(
    returns: pd.DataFrame,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """
    Enforce complete price availability: drop any asset with any missing return.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily returns DataFrame.
    drop_incomplete : bool
        If True, drop columns with any NaN. No imputation is applied.

    Returns
    -------
    pd.DataFrame
        Returns with only fully-observed assets.
    """
    if drop_incomplete:
        n_before = returns.shape[1]
        returns = returns.dropna(axis=1, how="any")
        n_after = returns.shape[1]
        if n_before != n_after:
            logger.info("Dropped %d assets due to missing data (complete-panel rule).", n_before - n_after)
    return returns


def load_exp1_dataset1(cache_dir: str = "data/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Dataset 1 for Experiment 1: 4-stock NSE universe.

    Estimation window: 2020-01-01 to 2022-12-31
    Evaluation window: 2023-01-01 to 2023-12-31

    Implementation assumption: Evaluation window is 2023, as the paper does not
    specify the exact out-of-sample period for the illustrative figure.

    Returns
    -------
    tuple of (returns_est, returns_eval)
    """
    prices = fetch_nse_prices(
        NSE_UNIVERSE_4,
        start="2020-01-01",
        end="2023-12-31",
        cache_dir=cache_dir,
    )
    returns = compute_daily_returns(prices)
    returns = get_complete_panel(returns)

    est_end = "2022-12-31"
    eval_start = "2023-01-01"

    returns_est = returns.loc[:est_end]
    returns_eval = returns.loc[eval_start:]
    return returns_est, returns_eval


def load_exp1_dataset2(cache_dir: str = "data/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Dataset 2 for Experiment 1: 20-stock NSE universe.

    Estimation window: 2006-2020
    Evaluation window: 2021-2025

    Implementation assumption: Evaluation window is 2021-2025 for illustrative figure.

    Returns
    -------
    tuple of (returns_est, returns_eval)
    """
    prices = fetch_nse_prices(
        NSE_UNIVERSE_20,
        start="2006-01-01",
        end="2025-12-31",
        cache_dir=cache_dir,
    )
    returns = compute_daily_returns(prices)
    returns = get_complete_panel(returns)

    est_end = "2020-12-31"
    eval_start = "2021-01-01"

    returns_est = returns.loc[:est_end]
    returns_eval = returns.loc[eval_start:]
    return returns_est, returns_eval


def load_main_experiment_data(cache_dir: str = "data/cache") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the full 20-stock NSE dataset for Experiments 2 and 3.

    Historical training period: 2006-01-02 to 2023-12-31
    Out-of-sample evaluation period: 2024-01-01 to 2025-12-31

    Returns
    -------
    tuple of (returns_hist, returns_eval)
    """
    prices = fetch_nse_prices(
        NSE_UNIVERSE_20,
        start="2006-01-01",
        end="2025-12-31",
        cache_dir=cache_dir,
    )
    returns = compute_daily_returns(prices)
    returns = get_complete_panel(returns)

    hist_end = "2023-12-31"
    eval_start = "2024-01-01"

    returns_hist = returns.loc[:hist_end]
    returns_eval = returns.loc[eval_start:]

    logger.info(
        "Historical period: %s to %s (%d days, %d assets)",
        returns_hist.index[0].date(),
        returns_hist.index[-1].date(),
        len(returns_hist),
        returns_hist.shape[1],
    )
    logger.info(
        "Evaluation period: %s to %s (%d days)",
        returns_eval.index[0].date(),
        returns_eval.index[-1].date(),
        len(returns_eval),
    )
    return returns_hist, returns_eval


if __name__ == "__main__":
    print("Fetching Exp1 Dataset1 (4 stocks)...")
    r_est, r_eval = load_exp1_dataset1()
    print(f"  Est: {r_est.shape}, Eval: {r_eval.shape}")

    print("Fetching main experiment data (20 stocks)...")
    r_hist, r_ev = load_main_experiment_data()
    print(f"  Hist: {r_hist.shape}, Eval: {r_ev.shape}")
