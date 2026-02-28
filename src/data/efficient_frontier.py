"""
Efficient Frontier Computation from Real Historical Data

This module computes Markowitz mean-variance efficient frontier portfolios
using real historical data from Yahoo Finance.

Pre-training Stage B: Creates 15 portfolios spanning the efficient frontier
from conservative (bonds-heavy) to aggressive (stocks-heavy).

Usage:
    from src.data.efficient_frontier import EfficientFrontierCalculator

    calculator = EfficientFrontierCalculator(num_portfolios=15)
    result = calculator.compute()

    # Access results
    weights = result['weights']  # (15, 3) array: [bonds, us_stocks, intl_stocks]
    returns = result['mean_returns']  # (15,) array of expected annual returns
    volatilities = result['volatilities']  # (15,) array of annual volatilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EfficientFrontierCalculator:
    """
    Computes Markowitz efficient frontier from real historical data.

    Downloads monthly returns for:
    - AGG: US Aggregate Bond ETF (bonds)
    - SPY: S&P 500 ETF (US stocks)
    - EFA: International Developed Markets ETF (international stocks)

    Runs quadratic optimization to find minimum variance portfolios
    for each target return level on the efficient frontier.
    """

    # Asset ETF symbols
    BOND_ETF = 'AGG'
    US_STOCK_ETF = 'SPY'
    INTL_STOCK_ETF = 'EFA'

    # Default date range (AGG starts 2003)
    DEFAULT_START_DATE = '2003-01-01'
    DEFAULT_END_DATE = None  # None means current date

    # Cache location
    CACHE_PATH = Path("data/processed/portfolio_parameters/efficient_frontier_real.json")

    def __init__(
        self,
        num_portfolios: int = 15,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize the efficient frontier calculator.

        Args:
            num_portfolios: Number of portfolios on the efficient frontier (default: 15)
            start_date: Start date for historical data (default: 2003-01-01)
            end_date: End date for historical data (default: current date)
            use_cache: Whether to use cached results if available (default: True)
        """
        self.num_portfolios = num_portfolios
        self.start_date = start_date or self.DEFAULT_START_DATE
        self.end_date = end_date  # None means use current date
        self.use_cache = use_cache

        # Results storage
        self._weights: Optional[np.ndarray] = None
        self._mean_returns: Optional[np.ndarray] = None
        self._volatilities: Optional[np.ndarray] = None
        self._asset_means: Optional[np.ndarray] = None
        self._asset_stds: Optional[np.ndarray] = None
        self._cov_matrix: Optional[np.ndarray] = None

    def compute(self, force_recompute: bool = False) -> Dict[str, np.ndarray]:
        """
        Compute or load the efficient frontier.

        Args:
            force_recompute: If True, ignore cache and recompute

        Returns:
            Dict containing:
                - 'weights': (num_portfolios, 3) array with [bonds, us_stocks, intl_stocks] weights
                - 'mean_returns': (num_portfolios,) array of expected annual returns
                - 'volatilities': (num_portfolios,) array of annual volatilities
                - 'asset_means': (3,) array of asset mean returns [bonds, us, intl]
                - 'asset_stds': (3,) array of asset volatilities
                - 'cov_matrix': (3, 3) covariance matrix
        """
        # Try cache first
        if self.use_cache and not force_recompute:
            cached = self._load_from_cache()
            if cached is not None:
                return cached

        # Compute from real data
        logger.info("Computing efficient frontier from REAL historical data...")

        try:
            result = self._compute_from_yfinance()

            # Cache the results
            if self.use_cache:
                self._save_to_cache(result)

            return result

        except Exception as e:
            logger.warning(f"Failed to compute from real data ({e}), using fallback")
            return self._get_fallback_frontier()

    def _load_from_cache(self) -> Optional[Dict[str, np.ndarray]]:
        """Load efficient frontier from cache if available and date range matches."""
        if not self.CACHE_PATH.exists():
            return None

        try:
            with open(self.CACHE_PATH, 'r') as f:
                cached = json.load(f)

            # Validate date range matches
            cached_start = cached.get('start_date')
            cached_end = cached.get('end_date')

            if cached_start != self.start_date or cached_end != self.end_date:
                logger.info(f"Cache date range mismatch: cached=[{cached_start}, {cached_end}], "
                           f"requested=[{self.start_date}, {self.end_date}]. Recomputing...")
                return None

            logger.info(f"Loaded efficient frontier from cache: {self.CACHE_PATH} "
                       f"(date range: {cached_start} to {cached_end})")

            return {
                'weights': np.array(cached['weights']),
                'mean_returns': np.array(cached['mean_returns']),
                'volatilities': np.array(cached['volatilities']),
                'asset_means': np.array(cached['asset_means']),
                'asset_stds': np.array(cached['asset_stds']),
                'cov_matrix': np.array(cached['cov_matrix'])
            }
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, result: Dict[str, np.ndarray]) -> None:
        """Save efficient frontier to cache with date range metadata."""
        try:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(self.CACHE_PATH, 'w') as f:
                json.dump({
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'weights': result['weights'].tolist(),
                    'mean_returns': result['mean_returns'].tolist(),
                    'volatilities': result['volatilities'].tolist(),
                    'asset_means': result['asset_means'].tolist(),
                    'asset_stds': result['asset_stds'].tolist(),
                    'cov_matrix': result['cov_matrix'].tolist()
                }, f, indent=2)

            logger.info(f"Cached efficient frontier to {self.CACHE_PATH} "
                       f"(date range: {self.start_date} to {self.end_date})")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _compute_from_yfinance(self) -> Dict[str, np.ndarray]:
        """
        Download data from Yahoo Finance and compute efficient frontier.

        Returns:
            Dict with weights, returns, volatilities, and asset statistics
        """
        import pandas as pd
        import yfinance as yf
        from scipy.optimize import minimize

        # Use end_date if provided, otherwise current date
        if self.end_date:
            end_date = self.end_date
        else:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        logger.info(f"  Downloading asset returns from {self.start_date} to {end_date}...")

        # Download monthly data for each asset class
        spy = yf.download(self.US_STOCK_ETF, start=self.start_date, end=end_date,
                         interval='1mo', progress=False)
        agg = yf.download(self.BOND_ETF, start=self.start_date, end=end_date,
                         interval='1mo', progress=False)
        efa = yf.download(self.INTL_STOCK_ETF, start=self.start_date, end=end_date,
                         interval='1mo', progress=False)

        # Compute monthly returns
        def get_returns(df):
            if 'Adj Close' in df.columns:
                prices = df['Adj Close']
            else:
                prices = df['Close']
            return prices.pct_change().dropna()

        spy_returns = get_returns(spy)
        agg_returns = get_returns(agg)
        efa_returns = get_returns(efa)

        # Align to common dates
        common_idx = spy_returns.index.intersection(agg_returns.index).intersection(efa_returns.index)
        spy_returns = spy_returns.loc[common_idx]
        agg_returns = agg_returns.loc[common_idx]
        efa_returns = efa_returns.loc[common_idx]

        n_months = len(common_idx)
        logger.info(f"  Common data points: {n_months} months")

        # Stack into matrix: [bonds, us_stocks, intl_stocks]
        returns_matrix = np.column_stack([
            agg_returns.values,  # Bonds
            spy_returns.values,  # US Stocks
            efa_returns.values   # International Stocks
        ])

        # Calculate annualized mean returns and covariance
        monthly_means = np.mean(returns_matrix, axis=0)
        monthly_cov = np.cov(returns_matrix.T)

        annual_means = monthly_means * 12  # Annualize returns
        annual_cov = monthly_cov * 12       # Annualize covariance
        annual_stds = np.sqrt(np.diag(annual_cov))

        logger.info(f"  Asset statistics (annualized):")
        logger.info(f"    Bonds ({self.BOND_ETF}): mean={annual_means[0]:.2%}, std={annual_stds[0]:.2%}")
        logger.info(f"    US Stocks ({self.US_STOCK_ETF}): mean={annual_means[1]:.2%}, std={annual_stds[1]:.2%}")
        logger.info(f"    Intl Stocks ({self.INTL_STOCK_ETF}): mean={annual_means[2]:.2%}, std={annual_stds[2]:.2%}")

        # Run Markowitz optimization for efficient frontier
        weights, portfolio_returns, portfolio_volatilities = self._optimize_frontier(
            annual_means, annual_cov
        )

        logger.info(f"  Efficient frontier computed:")
        logger.info(f"    Portfolio 0 (Conservative): return={portfolio_returns[0]:.2%}, vol={portfolio_volatilities[0]:.2%}")
        logger.info(f"      Weights: Bonds={weights[0,0]:.1%}, US={weights[0,1]:.1%}, Intl={weights[0,2]:.1%}")
        logger.info(f"    Portfolio {self.num_portfolios-1} (Aggressive): return={portfolio_returns[-1]:.2%}, vol={portfolio_volatilities[-1]:.2%}")
        logger.info(f"      Weights: Bonds={weights[-1,0]:.1%}, US={weights[-1,1]:.1%}, Intl={weights[-1,2]:.1%}")

        # Store internally
        self._weights = weights
        self._mean_returns = portfolio_returns
        self._volatilities = portfolio_volatilities
        self._asset_means = annual_means
        self._asset_stds = annual_stds
        self._cov_matrix = annual_cov

        return {
            'weights': weights,
            'mean_returns': portfolio_returns,
            'volatilities': portfolio_volatilities,
            'asset_means': annual_means,
            'asset_stds': annual_stds,
            'cov_matrix': annual_cov
        }

    def _optimize_frontier(
        self,
        annual_means: np.ndarray,
        annual_cov: np.ndarray
    ) -> tuple:
        """
        Run Markowitz optimization for each target return level.

        Args:
            annual_means: (3,) array of annualized mean returns
            annual_cov: (3, 3) annualized covariance matrix

        Returns:
            Tuple of (weights, returns, volatilities)
        """
        from scipy.optimize import minimize

        n_assets = len(annual_means)
        weights = np.zeros((self.num_portfolios, n_assets))
        portfolio_returns = np.zeros(self.num_portfolios)
        portfolio_volatilities = np.zeros(self.num_portfolios)

        # Target return range (from minimum-variance to maximum return)
        min_return = np.min(annual_means)
        max_return = np.max(annual_means)
        target_returns = np.linspace(min_return, max_return, self.num_portfolios)

        logger.info(f"  Computing {self.num_portfolios} portfolios on efficient frontier...")

        for i, target_return in enumerate(target_returns):
            # Quadratic programming: minimize variance subject to return and weight constraints
            def portfolio_variance(w):
                return w @ annual_cov @ w

            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda w, tr=target_return: w @ annual_means - tr}  # Target return
            ]
            bounds = [(0, 1) for _ in range(n_assets)]  # Long-only
            w0 = np.ones(n_assets) / n_assets  # Equal weight initial guess

            result = minimize(
                portfolio_variance,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-12}
            )

            if result.success:
                w = np.maximum(result.x, 0)  # Ensure non-negative
                w = w / np.sum(w)  # Normalize
                weights[i] = w
                portfolio_returns[i] = w @ annual_means
                portfolio_volatilities[i] = np.sqrt(w @ annual_cov @ w)
            else:
                # Fallback: linear interpolation between min and max return assets
                t = i / (self.num_portfolios - 1)
                weights[i] = [0.85 * (1 - t) + 0.05 * t,
                              0.10 * (1 - t) + 0.50 * t,
                              0.05 * (1 - t) + 0.45 * t]
                portfolio_returns[i] = weights[i] @ annual_means
                portfolio_volatilities[i] = np.sqrt(weights[i] @ annual_cov @ weights[i])

        return weights, portfolio_returns, portfolio_volatilities

    def _get_fallback_frontier(self) -> Dict[str, np.ndarray]:
        """
        Get fallback efficient frontier using hardcoded values.

        Used when yfinance download fails.
        """
        logger.info("Using fallback efficient frontier (hardcoded values)")

        # Based on historical averages
        n = self.num_portfolios

        # Linear interpolation from conservative to aggressive
        weights = np.zeros((n, 3))
        for i in range(n):
            t = i / (n - 1)
            weights[i] = [
                0.85 * (1 - t) + 0.05 * t,  # Bonds: 85% -> 5%
                0.10 * (1 - t) + 0.55 * t,  # US Stocks: 10% -> 55%
                0.05 * (1 - t) + 0.40 * t   # Intl Stocks: 5% -> 40%
            ]

        # Normalize weights
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Approximate asset statistics (historical averages)
        asset_means = np.array([0.035, 0.10, 0.08])  # bonds, us, intl
        asset_stds = np.array([0.045, 0.15, 0.17])

        # Simple correlation matrix
        corr = np.array([
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.7],
            [0.1, 0.7, 1.0]
        ])
        cov_matrix = np.outer(asset_stds, asset_stds) * corr

        # Compute portfolio statistics
        mean_returns = weights @ asset_means
        volatilities = np.array([np.sqrt(w @ cov_matrix @ w) for w in weights])

        return {
            'weights': weights,
            'mean_returns': mean_returns,
            'volatilities': volatilities,
            'asset_means': asset_means,
            'asset_stds': asset_stds,
            'cov_matrix': cov_matrix
        }

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Portfolio weights (num_portfolios, 3)."""
        return self._weights

    @property
    def mean_returns(self) -> Optional[np.ndarray]:
        """Portfolio expected returns (num_portfolios,)."""
        return self._mean_returns

    @property
    def volatilities(self) -> Optional[np.ndarray]:
        """Portfolio volatilities (num_portfolios,)."""
        return self._volatilities


def compute_efficient_frontier(
    num_portfolios: int = 15,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    force_recompute: bool = False
) -> Dict[str, np.ndarray]:
    """
    Convenience function to compute efficient frontier.

    Args:
        num_portfolios: Number of portfolios on the frontier
        start_date: Start date for historical data (default: 2003-01-01)
        end_date: End date for historical data (default: current date)
        use_cache: Whether to use cached results
        force_recompute: If True, ignore cache and recompute

    Returns:
        Dict with weights, returns, volatilities, and asset statistics
    """
    calculator = EfficientFrontierCalculator(
        num_portfolios=num_portfolios,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )
    return calculator.compute(force_recompute=force_recompute)


def get_portfolio_weights(
    num_portfolios: int = 15,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> np.ndarray:
    """
    Get portfolio weights from efficient frontier.

    Args:
        num_portfolios: Number of portfolios
        start_date: Start date for historical data (default: 2003-01-01)
        end_date: End date for historical data (default: current date)

    Returns:
        (num_portfolios, 3) array of weights [bonds, us_stocks, intl_stocks]
    """
    result = compute_efficient_frontier(
        num_portfolios,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
    return result['weights']


# =============================================================================
# PAPER-BASED EFFICIENT FRONTIER (For Reproducibility)
# =============================================================================

def get_paper_efficient_frontier(num_portfolios: int = 15) -> Dict[str, np.ndarray]:
    """
    Get efficient frontier using paper's hardcoded values from Das et al. (2019).

    This uses the exact market data from Table 1 (January 1998 to December 2017):
    - US Bonds: μ=4.93%, σ=4.12%
    - International Stocks: μ=7.70%, σ=19.90%
    - US Stocks: μ=8.86%, σ=19.78%

    The efficient frontier is computed using the Markowitz quadratic formula:
    σ = √(aμ² + bμ + c)

    Args:
        num_portfolios: Number of portfolios on the efficient frontier (default: 15)

    Returns:
        Dict containing:
            - 'mean_returns': (num_portfolios,) array of expected annual returns
            - 'volatilities': (num_portfolios,) array of annual volatilities
            - 'mu_min': Minimum return (0.0526)
            - 'mu_max': Maximum return (0.0886)
    """
    # Market data from Table 1 (Das et al. 2019)
    mu = np.array([0.0493, 0.0770, 0.0886])  # [US Bonds, Intl Stocks, US Stocks]

    # Covariance matrix from Table 1
    Sigma = np.array([
        [0.0017, -0.0017, -0.0021],
        [-0.0017, 0.0396, 0.0309],
        [-0.0021, 0.0309, 0.0392]
    ])

    # Calculate efficient frontier coefficients using Markowitz formulas
    ones = np.ones(3)
    Sigma_inv = np.linalg.inv(Sigma)
    k = mu.T @ Sigma_inv @ ones
    l = mu.T @ Sigma_inv @ mu
    p = ones.T @ Sigma_inv @ ones

    denominator = l * p - k**2
    g = (l * Sigma_inv @ ones - k * Sigma_inv @ mu) / denominator
    h = (p * Sigma_inv @ mu - k * Sigma_inv @ ones) / denominator

    # Quadratic coefficients: σ = √(aμ² + bμ + c)
    eff_frontier_a = h.T @ Sigma @ h
    eff_frontier_b = 2 * g.T @ Sigma @ h
    eff_frontier_c = g.T @ Sigma @ g

    # Portfolio bounds from paper
    mu_min = 0.0526  # Minimum return (portfolio 0)
    mu_max = 0.0886  # Maximum return (portfolio 14)

    # Create equally spaced μ values
    mu_array = np.linspace(mu_min, mu_max, num_portfolios)

    # Calculate corresponding σ using efficient frontier equation
    mu_squared = mu_array ** 2
    variance = eff_frontier_a * mu_squared + eff_frontier_b * mu_array + eff_frontier_c
    sigma_array = np.sqrt(np.maximum(variance, 0.0001))

    return {
        'mean_returns': mu_array,
        'volatilities': sigma_array,
        'mu_min': mu_min,
        'mu_max': mu_max,
        'eff_frontier_a': eff_frontier_a,
        'eff_frontier_b': eff_frontier_b,
        'eff_frontier_c': eff_frontier_c
    }


def get_portfolio_arrays(
    use_real_ef: bool = False,
    num_portfolios: int = 15,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified interface to get portfolio mean returns and volatilities.

    This is the single source of truth for portfolio parameters used by:
    - Dynamic Programming (DP)
    - Reinforcement Learning (RL)
    - Sentiment RL

    Args:
        use_real_ef: If True, compute EF from real Yahoo Finance data.
                     If False, use paper's hardcoded values (default).
        num_portfolios: Number of portfolios on the efficient frontier
        start_date: Start date for real data (only used if use_real_ef=True)
        end_date: End date for real data (only used if use_real_ef=True)

    Returns:
        Tuple of (mean_returns, volatilities):
            - mean_returns: (num_portfolios,) array of expected annual returns
            - volatilities: (num_portfolios,) array of annual volatilities
    """
    if use_real_ef:
        # Load from real Yahoo Finance data
        result = compute_efficient_frontier(
            num_portfolios=num_portfolios,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        return result['mean_returns'], result['volatilities']
    else:
        # Use paper's hardcoded values
        result = get_paper_efficient_frontier(num_portfolios)
        return result['mean_returns'], result['volatilities']
