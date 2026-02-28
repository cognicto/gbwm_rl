
"""
Historical Market Data Loader for GBWM Training

Loads historical market data and computes portfolio returns for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import logging
import json

from config.environment_config import DEFAULT_ENV_CONFIG


class HistoricalDataLoader:
    """
    Load and manage historical market data for GBWM training
    
    Handles:
    - Loading S&P 500 and bond historical data
    - Computing portfolio returns using efficient frontier weights
    - Providing random historical sequences for training episodes
    - Data validation and quality checks
    """
    
    def __init__(self, 
                 data_path: str = "data/raw/market_data/",
                 processed_path: str = "data/processed/",
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """
        Initialize historical data loader
        
        Args:
            data_path: Path to raw market data files
            processed_path: Path to processed portfolio parameters
            start_date: Start date for data (YYYY-MM-DD format)
            end_date: End date for data (YYYY-MM-DD format)
        """
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)
        self.start_date = start_date
        self.end_date = end_date
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Data containers
        self.sp500_data = None
        self.bond_data = None
        self.asset_returns = None
        self.portfolio_returns = None
        self.portfolio_weights = None
        
        # Load data
        self._load_market_data()
        self._load_portfolio_parameters()
        self._compute_portfolio_returns()
        
        self.logger.info(f"Historical data loaded: {len(self.portfolio_returns[0])} time periods")
    
    def _load_market_data(self):
        """Load historical market data from CSV files"""
        try:
            # Load S&P 500 data
            sp500_path = self.data_path / "sp500_historical.csv"
            self.sp500_data = pd.read_csv(sp500_path)
            self.sp500_data['Date'] = pd.to_datetime(self.sp500_data['Date'])
            self.sp500_data = self.sp500_data.sort_values('Date')
            
            # Load bond data
            bond_path = self.data_path / "bond_yields.csv"
            self.bond_data = pd.read_csv(bond_path)
            self.bond_data['Date'] = pd.to_datetime(self.bond_data['Date'])
            self.bond_data = self.bond_data.sort_values('Date')
            
            self.logger.info(f"Loaded S&P 500 data: {len(self.sp500_data)} records")
            self.logger.info(f"Loaded bond data: {len(self.bond_data)} records")
            
            # Filter by date range if specified
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                self.sp500_data = self.sp500_data[self.sp500_data['Date'] >= start_dt]
                self.bond_data = self.bond_data[self.bond_data['Date'] >= start_dt]
                
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                self.sp500_data = self.sp500_data[self.sp500_data['Date'] <= end_dt]
                self.bond_data = self.bond_data[self.bond_data['Date'] <= end_dt]
                
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            raise
    
    def _load_portfolio_parameters(self):
        """Load efficient frontier portfolio parameters"""
        try:
            # Load efficient frontier parameters
            ef_path = self.processed_path / "portfolio_parameters" / "efficient_frontier.json"
            with open(ef_path, 'r') as f:
                ef_data = json.load(f)
            
            # Extract portfolio parameters
            portfolios = ef_data['portfolios']
            self.num_portfolios = len(portfolios)
            
            # For this implementation, we'll use a simplified 3-asset model:
            # Asset 0: US Bonds, Asset 1: US Stocks (S&P 500), Asset 2: International Stocks
            # Create portfolio weights based on efficient frontier risk levels
            self.portfolio_weights = self._create_portfolio_weights(portfolios)
            
            self.logger.info(f"Loaded {self.num_portfolios} portfolio configurations")
            
        except Exception as e:
            self.logger.error(f"Failed to load portfolio parameters: {e}")
            raise
    
    def _create_portfolio_weights(self, portfolios: List[Dict]) -> np.ndarray:
        """
        Create portfolio weights for 3-asset model based on risk levels
        
        Assets: [US Bonds, US Stocks, International Stocks]
        Based on paper: Bonds (4.93%, 4.12%), US Stocks (7.70%, 19.90%), Intl Stocks (8.86%, 19.78%)
        
        Args:
            portfolios: Portfolio configurations from efficient frontier
            
        Returns:
            Array of shape (num_portfolios, 3) with portfolio weights
        """
        weights = np.zeros((len(portfolios), 3))  # 3 assets
        
        for i, portfolio in enumerate(portfolios):
            risk_level = portfolio['risk_level']
            expected_return = portfolio['expected_return']
            volatility = portfolio['volatility']
            
            if risk_level == "Conservative":
                # Higher bond allocation
                bond_weight = 0.7 - (i * 0.05)  # Decrease as we move up
                stock_weight = 0.25 + (i * 0.03)
                intl_weight = 0.05 + (i * 0.02)
                
            elif risk_level == "Moderate":
                # Balanced allocation
                bond_weight = 0.4 - ((i-5) * 0.03)  # Continue decreasing
                stock_weight = 0.45 + ((i-5) * 0.02)
                intl_weight = 0.15 + ((i-5) * 0.01)
                
            else:  # Aggressive
                # Higher equity allocation
                bond_weight = 0.15 - ((i-10) * 0.02)
                stock_weight = 0.55 + ((i-10) * 0.01)
                intl_weight = 0.30 + ((i-10) * 0.01)
            
            # Normalize weights to sum to 1
            total_weight = bond_weight + stock_weight + intl_weight
            weights[i] = [bond_weight/total_weight, stock_weight/total_weight, intl_weight/total_weight]
        
        return weights
    
    def _compute_asset_returns(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute ANNUAL returns for the 3 assets
        
        Returns:
            Tuple of (bond_returns, us_stock_returns, intl_stock_returns)
        """
        # Check if we have Annual_Return column (new format)
        if 'Annual_Return' in self.sp500_data.columns:
            # New format: Use pre-calculated annual returns
            us_stock_returns = pd.Series(self.sp500_data['Annual_Return'].values)
            self.logger.info("Using pre-calculated annual stock returns")
        else:
            # Legacy format: Compute returns from prices
            us_stock_returns = self.sp500_data['Close'].pct_change().dropna()
            self.logger.info("Computing returns from price data")
        
        # Bond returns from Treasury yields (ANNUAL basis)
        bond_yields = self.bond_data['10Y_Treasury'].values  # Already in decimal form
        
        # For bonds, annual return ≈ yield + capital gains/losses
        # Simplified: use yield as base return with some duration effect
        bond_returns = []
        for i in range(len(bond_yields)):
            base_return = bond_yields[i]  # Use yield as base return
            
            # Add capital gains/losses from yield changes (duration effect)
            if i > 0:
                yield_change = bond_yields[i] - bond_yields[i-1]
                duration = 7.0  # Approximate duration for 10Y bonds
                capital_change = -duration * yield_change
                total_return = base_return + capital_change
            else:
                total_return = base_return
                
            bond_returns.append(total_return)
        
        bond_returns = pd.Series(bond_returns)
        
        # International stock returns (correlated with US stocks)
        intl_correlation = 0.7866  # From paper
        noise_factor = 0.6  # Higher noise for annual data
        
        if len(us_stock_returns) > 0:
            intl_base = us_stock_returns * intl_correlation
            intl_noise = np.random.normal(0, us_stock_returns.std() * noise_factor, len(us_stock_returns))
            intl_stock_returns = intl_base + intl_noise
        else:
            # Fallback if no US stock data
            intl_stock_returns = np.random.normal(0.08, 0.20, len(bond_returns))  # 8% mean, 20% vol
        
        # Ensure all series have the same length
        min_length = min(len(us_stock_returns), len(bond_returns), len(intl_stock_returns))
        
        self.logger.info(f"Computed annual returns: {min_length} years")
        self.logger.info(f"US Stocks - Mean: {us_stock_returns.iloc[:min_length].mean():.3f}, Std: {us_stock_returns.iloc[:min_length].std():.3f}")
        self.logger.info(f"Bonds - Mean: {bond_returns.iloc[:min_length].mean():.3f}, Std: {bond_returns.iloc[:min_length].std():.3f}")
        
        return (bond_returns.iloc[:min_length],
                us_stock_returns.iloc[:min_length], 
                pd.Series(intl_stock_returns[:min_length]))
    
    def _compute_portfolio_returns(self):
        """Compute historical returns for each portfolio using asset weights"""
        try:
            # Get asset returns
            bond_returns, us_stock_returns, intl_stock_returns = self._compute_asset_returns()
            
            # Store asset returns
            self.asset_returns = {
                'bonds': bond_returns,
                'us_stocks': us_stock_returns,
                'intl_stocks': intl_stock_returns
            }
            
            # Compute portfolio returns
            self.portfolio_returns = []
            
            for i in range(self.num_portfolios):
                weights = self.portfolio_weights[i]
                
                # Weighted combination of asset returns
                portfolio_return = (weights[0] * bond_returns + 
                                  weights[1] * us_stock_returns + 
                                  weights[2] * intl_stock_returns)
                
                self.portfolio_returns.append(portfolio_return.values)
            
            # Convert to numpy array for easier indexing
            self.portfolio_returns = np.array(self.portfolio_returns)
            
            self.logger.info(f"Computed returns for {self.num_portfolios} portfolios")
            self.logger.info(f"Data length: {self.portfolio_returns.shape[1]} time periods")
            
        except Exception as e:
            self.logger.error(f"Failed to compute portfolio returns: {e}")
            raise
    
    def get_historical_sequence(self, start_idx: int, length: int = 16) -> np.ndarray:
        """
        Get specific historical sequence for all portfolios
        
        Args:
            start_idx: Starting index in the historical data
            length: Length of sequence (default 16 for 16 years)
            
        Returns:
            Array of shape (num_portfolios, length) with historical returns
        """
        if start_idx + length > self.portfolio_returns.shape[1]:
            raise ValueError(f"Not enough data: requested {start_idx + length}, "
                           f"available {self.portfolio_returns.shape[1]}")
        
        return self.portfolio_returns[:, start_idx:start_idx + length]
    
    def get_random_sequence(self, length: int = 16) -> np.ndarray:
        """
        Get random historical sequence for training
        
        Args:
            length: Length of sequence (default 16 for 16 years)
            
        Returns:
            Array of shape (num_portfolios, length) with historical returns
        """
        max_start = self.portfolio_returns.shape[1] - length
        if max_start <= 0:
            raise ValueError(f"Not enough historical data for sequence length {length}")
        
        start_idx = np.random.randint(0, max_start)
        return self.get_historical_sequence(start_idx, length)
    
    def get_available_sequences(self, length: int = 16) -> int:
        """Get number of available sequences of given length"""
        return max(0, self.portfolio_returns.shape[1] - length + 1)
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality and return statistics
        
        Returns:
            Dictionary with data quality metrics
        """
        stats = {
            'total_periods': self.portfolio_returns.shape[1],
            'num_portfolios': self.portfolio_returns.shape[0],
            'available_16y_sequences': self.get_available_sequences(16),
            'date_range': {
                'start': str(self.sp500_data['Date'].min()),
                'end': str(self.sp500_data['Date'].max())
            }
        }
        
        # Check for missing values
        missing_data = np.isnan(self.portfolio_returns).sum()
        stats['missing_values'] = int(missing_data)
        
        # Return statistics for each portfolio
        portfolio_stats = []
        for i in range(self.num_portfolios):
            returns = self.portfolio_returns[i]
            portfolio_stats.append({
                'portfolio_id': i,
                'mean_return': float(np.mean(returns)),
                'std_return': float(np.std(returns)),
                'min_return': float(np.min(returns)),
                'max_return': float(np.max(returns))
            })
        
        stats['portfolio_statistics'] = portfolio_stats
        
        return stats
    
    def get_portfolio_weights(self) -> np.ndarray:
        """Get portfolio weights for all portfolios"""
        return self.portfolio_weights.copy()
    
    def get_asset_names(self) -> List[str]:
        """Get names of underlying assets"""
        return ['US_Bonds', 'US_Stocks', 'International_Stocks']


# Convenience function for creating loader
def create_historical_loader(data_path: str = "data/raw/market_data/",
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> HistoricalDataLoader:
    """
    Create historical data loader with validation
    
    Args:
        data_path: Path to market data files
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        
    Returns:
        Initialized HistoricalDataLoader
    """
    loader = HistoricalDataLoader(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date
    )
    
    # Validate data quality
    stats = loader.validate_data_quality()
    logging.info(f"Historical data validation: {stats}")
    
    if stats['available_16y_sequences'] < 10:
        logging.warning(f"Limited historical sequences available: {stats['available_16y_sequences']}")
    
    return loader