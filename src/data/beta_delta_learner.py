"""
Beta and Delta Parameter Learning from Historical Data

This module learns the VIX sensitivity parameters (β and δ) that adjust
portfolio returns and volatility based on market sentiment.

Pre-training Stage A: Learns β and δ by regressing real historical returns
and volatility on lagged VIX (previous month's end-of-month VIX).

Professor's formulas:
    μ_adj = μ + β × (θ - VIX) / θ
    σ_adj = σ - δ × (θ - VIX) / θ

Where:
    - θ = 20 (long-term VIX mean)
    - β can be NEGATIVE (mean reversion: high VIX → higher future returns)
    - δ is always POSITIVE (high VIX → higher future volatility)

Usage:
    from src.data.beta_delta_learner import BetaDeltaLearner

    learner = BetaDeltaLearner()
    result = learner.learn()

    # Access results
    portfolio_betas = result['portfolio_betas']  # (15,) array
    portfolio_deltas = result['portfolio_deltas']  # (15,) array
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BetaDeltaLearner:
    """
    Learns VIX sensitivity parameters from historical data.

    Uses monthly data:
    - VIX: Last 5 trading days of previous month (lagged predictor)
    - Returns: Current month's returns from yfinance (SPY, AGG, EFA)
    - Volatility: Forward-looking 3-month rolling volatility

    Regression models:
    - β regression: R_t = α + β × (θ - VIX_{t-1})/θ + ε
    - δ regression: σ_forward = γ + δ × (θ - VIX_{t-1})/θ + η
    """

    # VIX long-term mean
    VIX_THETA = 20.0

    # Asset ETF symbols
    BOND_ETF = 'AGG'
    US_STOCK_ETF = 'SPY'
    INTL_STOCK_ETF = 'EFA'

    # Rolling window for volatility calculation
    ROLLING_WINDOW = 3  # 3-month forward-looking volatility

    # Default date range
    DEFAULT_START_DATE = '1990-01-01'  # VIX data starts 1990
    DEFAULT_END_DATE = None  # None means current date

    # Cache location
    CACHE_PATH = Path("data/processed/portfolio_parameters/beta_delta_params.json")

    def __init__(
        self,
        num_portfolios: int = 15,
        vix_theta: float = 20.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        volatility_method: str = 'rolling_vol'
    ):
        """
        Initialize the beta/delta learner.

        Args:
            num_portfolios: Number of portfolios on efficient frontier (default: 15)
            vix_theta: Long-term VIX mean (default: 20.0)
            start_date: Start date for historical data (default: 1990-01-01)
            end_date: End date for historical data (default: current date)
            use_cache: Whether to use cached results if available (default: True)
            volatility_method: Method for δ regression (default: 'rolling_vol')
                - 'rolling_vol': Regress 3-month forward-looking rolling volatility on VIX
                - 'return_squared': Regress R² (squared returns) on VIX, δ adjusts variance
        """
        self.num_portfolios = num_portfolios
        self.vix_theta = vix_theta
        self.start_date = start_date or self.DEFAULT_START_DATE
        self.end_date = end_date  # None means use current date
        self.use_cache = use_cache
        self.volatility_method = volatility_method

        if volatility_method not in ['rolling_vol', 'return_squared']:
            raise ValueError(f"volatility_method must be 'rolling_vol' or 'return_squared', got '{volatility_method}'")

        # Results storage
        self._asset_betas: Optional[np.ndarray] = None
        self._asset_deltas: Optional[np.ndarray] = None
        self._portfolio_betas: Optional[np.ndarray] = None
        self._portfolio_deltas: Optional[np.ndarray] = None
        self._portfolio_weights: Optional[np.ndarray] = None

    def learn(self, force_recompute: bool = False) -> Dict[str, Any]:
        """
        Learn β and δ parameters from historical data.

        Args:
            force_recompute: If True, ignore cache and recompute

        Returns:
            Dict containing:
                - 'asset_betas': (3,) array [bonds, us_stocks, intl_stocks]
                - 'asset_deltas': (3,) array [bonds, us_stocks, intl_stocks]
                - 'portfolio_betas': (num_portfolios,) array
                - 'portfolio_deltas': (num_portfolios,) array
                - 'portfolio_weights': (num_portfolios, 3) array
                - 'vix_theta': float
        """
        # Try cache first
        if self.use_cache and not force_recompute:
            cached = self._load_from_cache()
            if cached is not None:
                return cached

        # Learn from real data
        logger.info("Learning β and δ from MONTHLY VIX and REAL asset-class returns...")

        try:
            result = self._learn_from_yfinance()

            # Cache the results
            if self.use_cache:
                self._save_to_cache(result)

            return result

        except Exception as e:
            logger.warning(f"Error learning β/δ: {e}, using defaults")
            import traceback
            traceback.print_exc()
            return self._get_default_values()

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """Load β/δ parameters from cache if available and date range matches."""
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

            logger.info(f"Loaded β/δ parameters from cache: {self.CACHE_PATH} "
                       f"(date range: {cached_start} to {cached_end})")

            return {
                'asset_betas': np.array(cached['asset_betas']),
                'asset_deltas': np.array(cached['asset_deltas']),
                'portfolio_betas': np.array(cached['portfolio_betas']),
                'portfolio_deltas': np.array(cached['portfolio_deltas']),
                'portfolio_weights': np.array(cached['portfolio_weights']),
                'vix_theta': cached['vix_theta']
            }
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def _save_to_cache(self, result: Dict[str, Any]) -> None:
        """Save β/δ parameters to cache with date range metadata."""
        try:
            self.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(self.CACHE_PATH, 'w') as f:
                json.dump({
                    'start_date': self.start_date,
                    'end_date': self.end_date,
                    'asset_betas': result['asset_betas'].tolist(),
                    'asset_deltas': result['asset_deltas'].tolist(),
                    'portfolio_betas': result['portfolio_betas'].tolist(),
                    'portfolio_deltas': result['portfolio_deltas'].tolist(),
                    'portfolio_weights': result['portfolio_weights'].tolist(),
                    'vix_theta': result['vix_theta']
                }, f, indent=2)

            logger.info(f"Cached β/δ parameters to {self.CACHE_PATH} "
                       f"(date range: {self.start_date} to {self.end_date})")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _learn_from_yfinance(self) -> Dict[str, Any]:
        """
        Learn β and δ from real yfinance data.

        Returns:
            Dict with learned parameters
        """
        from sklearn.linear_model import LinearRegression
        from src.data.sentiment_provider import SentimentProvider
        from src.data.efficient_frontier import get_portfolio_weights

        # =====================================================================
        # STEP 1: Load VIX data and compute LAST 5 DAYS of each month
        # =====================================================================
        sentiment_provider = SentimentProvider()

        # Determine end date for VIX data
        if self.end_date:
            vix_end_date = self.end_date
        else:
            vix_end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        success = sentiment_provider.initialize_historical(
            start_date=self.start_date,
            end_date=vix_end_date
        )

        if not success or sentiment_provider.vix_data is None:
            logger.warning("Failed to load historical VIX data, using defaults")
            return self._get_default_values()

        vix_data = sentiment_provider.vix_data

        # Convert to DataFrame with datetime index
        if isinstance(vix_data.index, pd.DatetimeIndex):
            vix_df = pd.DataFrame({'vix': vix_data['vix_close'].values}, index=vix_data.index)
        else:
            vix_df = pd.DataFrame({
                'date': pd.to_datetime(vix_data['date']),
                'vix': vix_data['vix_close'].values
            })
            vix_df.set_index('date', inplace=True)

        # Compute LAST 5 DAYS average for each month (end-of-month VIX)
        vix_df['year_month'] = vix_df.index.to_period('M')

        def last_5_days_avg(group):
            return group.tail(5)['vix'].mean()

        monthly_vix_eom = vix_df.groupby('year_month').apply(last_5_days_avg)

        # SHIFT FORWARD: Use previous month's end-of-month VIX as beginning of current month
        monthly_vix_lagged = monthly_vix_eom.shift(1).dropna()

        logger.info(f"  Monthly VIX (last 5 days avg, lagged): {len(monthly_vix_lagged)} observations")
        logger.info(f"  Date range: {monthly_vix_lagged.index.min()} to {monthly_vix_lagged.index.max()}")

        # =====================================================================
        # STEP 2: Download REAL monthly returns from Yahoo Finance
        # =====================================================================
        logger.info("  Downloading REAL monthly returns from Yahoo Finance...")

        try:
            import yfinance as yf

            # Use configured date range
            yf_start_date = self.start_date
            if self.end_date:
                yf_end_date = self.end_date
            else:
                yf_end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

            logger.info(f"    Downloading {self.US_STOCK_ETF} (S&P 500)...")
            spy = yf.download(self.US_STOCK_ETF, start=yf_start_date, end=yf_end_date,
                            interval='1mo', progress=False)

            logger.info(f"    Downloading {self.BOND_ETF} (US Aggregate Bond)...")
            agg = yf.download(self.BOND_ETF, start=yf_start_date, end=yf_end_date,
                            interval='1mo', progress=False)

            logger.info(f"    Downloading {self.INTL_STOCK_ETF} (International Developed)...")
            efa = yf.download(self.INTL_STOCK_ETF, start=yf_start_date, end=yf_end_date,
                            interval='1mo', progress=False)

            # Compute monthly returns from adjusted close prices
            def compute_monthly_returns(df):
                if 'Adj Close' in df.columns:
                    prices = df['Adj Close']
                else:
                    prices = df['Close']
                returns = prices.pct_change().dropna()
                returns.index = pd.to_datetime(returns.index).to_period('M')
                return returns

            us_stock_returns = compute_monthly_returns(spy)
            bond_returns = compute_monthly_returns(agg)
            intl_stock_returns = compute_monthly_returns(efa)

            logger.info(f"    SPY returns: {len(us_stock_returns)} months")
            logger.info(f"    AGG returns: {len(bond_returns)} months")
            logger.info(f"    EFA returns: {len(intl_stock_returns)} months")

            # Find common periods across VIX and all asset returns
            common_periods = (
                set(monthly_vix_lagged.index) &
                set(us_stock_returns.index) &
                set(bond_returns.index) &
                set(intl_stock_returns.index)
            )
            common_periods = sorted(list(common_periods))

            if len(common_periods) < 60:  # Need at least 5 years of data
                logger.warning(f"Insufficient overlapping data ({len(common_periods)} months)")
                return self._learn_synthetic_fallback(monthly_vix_lagged)

            logger.info(f"  Common periods: {len(common_periods)} months ({common_periods[0]}-{common_periods[-1]})")

            # Align all data to common periods using .loc for robust indexing
            # Use .ravel() to ensure 1D arrays (yfinance can return 2D with MultiIndex columns)
            monthly_vix_aligned = monthly_vix_lagged.loc[common_periods].values.ravel()
            monthly_returns = {
                'us_stocks': us_stock_returns.loc[common_periods].values.ravel(),
                'bonds': bond_returns.loc[common_periods].values.ravel(),
                'intl_stocks': intl_stock_returns.loc[common_periods].values.ravel()
            }

            logger.info(f"  REAL data aligned: {len(monthly_vix_aligned)} months")
            for asset, returns in monthly_returns.items():
                ann_mean = returns.mean() * 12
                ann_vol = returns.std() * np.sqrt(12)
                logger.info(f"    {asset}: mean={ann_mean:.2%}/year, vol={ann_vol:.2%}/year")

        except Exception as e:
            logger.warning(f"Failed to download from yfinance ({e}), falling back to synthetic data")
            return self._learn_synthetic_fallback(monthly_vix_lagged)

        # =====================================================================
        # STEP 3: Learn β and δ for EACH ASSET CLASS
        # =====================================================================
        # VIX is ALREADY LAGGED (previous month's end → current month's beginning)
        vix_normalized = (self.vix_theta - monthly_vix_aligned) / self.vix_theta
        X = vix_normalized.reshape(-1, 1)

        asset_betas = {}
        asset_deltas = {}

        if self.volatility_method == 'rolling_vol':
            logger.info(f"\n  β/δ Regression Results (VIX → Returns, {self.ROLLING_WINDOW}-month rolling vol for δ):")
        else:
            logger.info(f"\n  β/δ Regression Results (VIX → Returns, R² for δ as variance adjustment):")

        for asset_name, returns in monthly_returns.items():
            # ─────────────────────────────────────────────────────────────
            # β regression: R_t = α + β × (θ - VIX_{t-1})/θ + ε
            # ─────────────────────────────────────────────────────────────
            reg_beta = LinearRegression().fit(X, returns)
            beta_monthly = reg_beta.coef_[0]
            r2_beta = reg_beta.score(X, returns)

            # ─────────────────────────────────────────────────────────────
            # δ regression: Two methods available
            # ─────────────────────────────────────────────────────────────
            if self.volatility_method == 'rolling_vol':
                # Method 1: FORWARD-LOOKING ROLLING VOLATILITY
                # σ_forward(t) = std(R_t, R_{t+1}, R_{t+2})
                returns_series = pd.Series(returns)
                forward_vol = returns_series[::-1].rolling(window=self.ROLLING_WINDOW).std()[::-1]

                valid_idx = ~forward_vol.isna()
                X_valid = X[valid_idx]
                vol_valid = forward_vol[valid_idx].values

                reg_delta = LinearRegression().fit(X_valid, vol_valid)
                delta_monthly = reg_delta.coef_[0]
                r2_delta = reg_delta.score(X_valid, vol_valid)

                # Convert monthly to annual (volatility scales with sqrt(12))
                delta_annual = delta_monthly * np.sqrt(12)

            else:  # return_squared
                # Method 2: RETURN SQUARED as variance proxy
                # R²_t = γ + δ × (θ - VIX_{t-1})/θ + η
                # δ adjusts VARIANCE, so annualize with ×12
                returns_squared = returns ** 2

                reg_delta = LinearRegression().fit(X, returns_squared)
                delta_monthly = reg_delta.coef_[0]
                r2_delta = reg_delta.score(X, returns_squared)

                # Convert monthly to annual (variance scales with ×12)
                delta_annual = delta_monthly * 12

            # Convert monthly beta to annual
            beta_annual = beta_monthly * 12

            asset_betas[asset_name] = beta_annual
            asset_deltas[asset_name] = abs(delta_annual)  # δ is always positive

            logger.info(f"    {asset_name}: β_annual={beta_annual:+.4f} (R²={r2_beta:.3f}), "
                       f"δ_annual={abs(delta_annual):.4f} (R²={r2_delta:.3f})")

        # =====================================================================
        # STEP 4: Compute PORTFOLIO-SPECIFIC β and δ
        # =====================================================================
        portfolio_weights = get_portfolio_weights(self.num_portfolios)

        # Asset order: [bonds, us_stocks, intl_stocks]
        asset_beta_array = np.array([
            asset_betas['bonds'],
            asset_betas['us_stocks'],
            asset_betas['intl_stocks']
        ])
        asset_delta_array = np.array([
            asset_deltas['bonds'],
            asset_deltas['us_stocks'],
            asset_deltas['intl_stocks']
        ])

        # Portfolio β = weighted sum of asset βs
        portfolio_betas = portfolio_weights @ asset_beta_array
        portfolio_deltas = portfolio_weights @ asset_delta_array

        # Clip to reasonable bounds (allow negative beta for mean reversion)
        portfolio_betas = np.clip(portfolio_betas, -0.15, 0.15)
        portfolio_deltas = np.clip(portfolio_deltas, 0.005, 0.20)

        logger.info(f"\nPortfolio-specific β and δ:")
        for i in [0, 7, 14]:
            logger.info(f"  Portfolio {i}: β={portfolio_betas[i]:.4f}, δ={portfolio_deltas[i]:.4f}, "
                       f"weights=[{portfolio_weights[i,0]:.2f}, {portfolio_weights[i,1]:.2f}, {portfolio_weights[i,2]:.2f}]")

        # Store internally
        self._asset_betas = asset_beta_array
        self._asset_deltas = asset_delta_array
        self._portfolio_betas = portfolio_betas
        self._portfolio_deltas = portfolio_deltas
        self._portfolio_weights = portfolio_weights

        return {
            'asset_betas': asset_beta_array,
            'asset_deltas': asset_delta_array,
            'portfolio_betas': portfolio_betas,
            'portfolio_deltas': portfolio_deltas,
            'portfolio_weights': portfolio_weights,
            'vix_theta': self.vix_theta
        }

    def _learn_synthetic_fallback(self, monthly_vix_lagged) -> Dict[str, Any]:
        """
        Fallback: Learn β/δ using synthetic returns when yfinance fails.

        Uses the original approach of generating monthly returns from annual statistics,
        but with the CORRECTED VIX (last 5 days of previous month).

        Args:
            monthly_vix_lagged: Series of lagged monthly VIX values

        Returns:
            Dict with learned β/δ parameters
        """
        from sklearn.linear_model import LinearRegression
        from src.data.efficient_frontier import get_portfolio_weights

        logger.info("  Using synthetic fallback for monthly returns...")

        project_root = Path(__file__).parent.parent.parent

        # Load annual data
        sp500_path = project_root / "data" / "raw" / "market_data" / "sp500_historical.csv"
        sp500_df = pd.read_csv(sp500_path)
        sp500_df['year'] = pd.to_datetime(sp500_df['Date']).dt.year
        sp500_annual = sp500_df.set_index('year')['Annual_Return']

        bond_path = project_root / "data" / "raw" / "market_data" / "bond_yields.csv"
        bond_df = pd.read_csv(bond_path)
        bond_df['year'] = pd.to_datetime(bond_df['Date']).dt.year
        bond_df['yield_decimal'] = bond_df['10Y_Treasury']
        bond_df['yield_change'] = bond_df['yield_decimal'].diff().fillna(0)
        bond_df['bond_return'] = bond_df['yield_decimal'] - 7.0 * bond_df['yield_change']
        bond_annual = bond_df.set_index('year')['bond_return']

        # Find common years
        vix_years = set(monthly_vix_lagged.index.year)
        common_years = sorted(list(vix_years & set(sp500_annual.index) & set(bond_annual.index)))

        if len(common_years) < 10:
            logger.warning("Insufficient data in fallback, using defaults")
            return self._get_default_values()

        # Generate synthetic monthly returns
        np.random.seed(42)
        monthly_returns = {'bonds': [], 'us_stocks': [], 'intl_stocks': []}
        monthly_vix_aligned = []

        for year in common_years:
            us_annual_ret = sp500_annual[year]
            bond_annual_ret = bond_annual[year]

            us_monthly_mean = us_annual_ret / 12
            us_monthly_std = 0.165 / np.sqrt(12)
            bond_monthly_mean = bond_annual_ret / 12
            bond_monthly_std = 0.04 / np.sqrt(12)
            intl_corr = 0.79

            for month in range(1, 13):
                period = pd.Period(f"{year}-{month:02d}", freq='M')
                if period in monthly_vix_lagged.index:
                    monthly_vix_aligned.append(monthly_vix_lagged[period])

                    z_us = np.random.normal(0, 1)
                    z_bond = np.random.normal(0, 1)
                    z_intl = intl_corr * z_us + np.sqrt(1 - intl_corr**2) * np.random.normal(0, 1)

                    monthly_returns['us_stocks'].append(us_monthly_mean + us_monthly_std * z_us)
                    monthly_returns['bonds'].append(bond_monthly_mean + bond_monthly_std * z_bond)
                    monthly_returns['intl_stocks'].append(us_monthly_mean + us_monthly_std * z_intl)

        monthly_vix_aligned = np.array(monthly_vix_aligned)
        for key in monthly_returns:
            monthly_returns[key] = np.array(monthly_returns[key])

        # Run regression
        vix_normalized = (self.vix_theta - monthly_vix_aligned) / self.vix_theta
        X = vix_normalized.reshape(-1, 1)

        asset_betas = {}
        asset_deltas = {}

        for asset_name, returns in monthly_returns.items():
            # β regression
            reg_beta = LinearRegression().fit(X, returns)
            beta_monthly = reg_beta.coef_[0]

            # δ regression: uses FORWARD-LOOKING rolling volatility
            returns_series = pd.Series(returns)
            forward_vol = returns_series[::-1].rolling(window=self.ROLLING_WINDOW).std()[::-1]

            valid_idx = ~forward_vol.isna()
            X_valid = X[valid_idx]
            vol_valid = forward_vol[valid_idx].values

            reg_delta = LinearRegression().fit(X_valid, vol_valid)
            delta_monthly = reg_delta.coef_[0]

            asset_betas[asset_name] = beta_monthly * 12
            asset_deltas[asset_name] = abs(delta_monthly * np.sqrt(12))

        # Compute portfolio-specific values
        portfolio_weights = get_portfolio_weights(self.num_portfolios)
        asset_beta_array = np.array([asset_betas['bonds'], asset_betas['us_stocks'], asset_betas['intl_stocks']])
        asset_delta_array = np.array([asset_deltas['bonds'], asset_deltas['us_stocks'], asset_deltas['intl_stocks']])

        portfolio_betas = np.clip(portfolio_weights @ asset_beta_array, -0.15, 0.15)
        portfolio_deltas = np.clip(portfolio_weights @ asset_delta_array, 0.005, 0.20)

        logger.info(f"  Synthetic fallback: {len(monthly_vix_aligned)} months processed")

        return {
            'asset_betas': asset_beta_array,
            'asset_deltas': asset_delta_array,
            'portfolio_betas': portfolio_betas,
            'portfolio_deltas': portfolio_deltas,
            'portfolio_weights': portfolio_weights,
            'vix_theta': self.vix_theta
        }

    def _get_default_values(self) -> Dict[str, Any]:
        """Return default β/δ values when learning fails."""
        from src.data.efficient_frontier import get_portfolio_weights

        portfolio_weights = get_portfolio_weights(self.num_portfolios)

        # Default asset-class values (based on financial literature)
        # Note: β can be negative for mean reversion effect
        asset_betas = np.array([-0.01, -0.05, -0.04])   # [bonds, us, intl] - negative for mean reversion
        asset_deltas = np.array([0.01, 0.06, 0.055])     # [bonds, us, intl] - always positive

        portfolio_betas = portfolio_weights @ asset_betas
        portfolio_deltas = portfolio_weights @ asset_deltas

        return {
            'asset_betas': asset_betas,
            'asset_deltas': asset_deltas,
            'portfolio_betas': portfolio_betas,
            'portfolio_deltas': portfolio_deltas,
            'portfolio_weights': portfolio_weights,
            'vix_theta': self.vix_theta
        }

    @property
    def portfolio_betas(self) -> Optional[np.ndarray]:
        """Portfolio beta values (num_portfolios,)."""
        return self._portfolio_betas

    @property
    def portfolio_deltas(self) -> Optional[np.ndarray]:
        """Portfolio delta values (num_portfolios,)."""
        return self._portfolio_deltas


def learn_beta_delta(
    num_portfolios: int = 15,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    force_recompute: bool = False,
    volatility_method: str = 'rolling_vol'
) -> Dict[str, Any]:
    """
    Convenience function to learn β and δ parameters.

    Args:
        num_portfolios: Number of portfolios
        start_date: Start date for historical data (default: 1990-01-01)
        end_date: End date for historical data (default: current date)
        use_cache: Whether to use cached results
        force_recompute: If True, ignore cache and recompute
        volatility_method: Method for δ regression (default: 'rolling_vol')
            - 'rolling_vol': Regress 3-month forward-looking rolling volatility on VIX
            - 'return_squared': Regress R² (squared returns) on VIX, δ adjusts variance

    Returns:
        Dict with learned β/δ parameters
    """
    learner = BetaDeltaLearner(
        num_portfolios=num_portfolios,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        volatility_method=volatility_method
    )
    return learner.learn(force_recompute=force_recompute)


def get_default_beta_delta(num_portfolios: int = 15) -> Dict[str, Any]:
    """
    Get default β/δ values (no learning).

    Args:
        num_portfolios: Number of portfolios

    Returns:
        Dict with default β/δ parameters
    """
    learner = BetaDeltaLearner(num_portfolios=num_portfolios)
    return learner._get_default_values()
