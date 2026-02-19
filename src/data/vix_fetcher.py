"""
VIX Data Fetcher for GBWM Sentiment Integration

This module fetches VIX (CBOE Volatility Index) data from Yahoo Finance
for use in sentiment-aware goal-based wealth management.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIXFetcher:
    """
    Fetch VIX data from Yahoo Finance
    
    The VIX is often called the "fear gauge" as it measures market volatility
    expectations. Higher VIX values indicate higher fear/uncertainty in markets.
    """
    
    def __init__(self):
        self.vix_symbol = '^VIX'
        
    def fetch_historical(
        self, 
        start_date: str, 
        end_date: str,
        validate_data: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical VIX data from Yahoo Finance
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            validate_data: Whether to validate data quality
            
        Returns:
            DataFrame with columns:
                - date: datetime index
                - vix_open: float
                - vix_high: float
                - vix_low: float
                - vix_close: float
                - vix_volume: int
                
        Raises:
            ValueError: If date range is invalid
            RuntimeError: If data fetch fails
        """
        try:
            # Validate date inputs
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            if start_dt >= end_dt:
                raise ValueError(f"Start date {start_date} must be before end date {end_date}")
            
            # Check if date range is reasonable
            if start_dt < pd.to_datetime('1990-01-01'):
                logger.warning(f"Start date {start_date} is very early, VIX data may be limited")
            
            logger.info(f"Fetching VIX data from {start_date} to {end_date}")
            
            # Fetch data from Yahoo Finance
            vix_ticker = yf.Ticker(self.vix_symbol)
            vix_data = vix_ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if vix_data.empty:
                raise RuntimeError(f"No VIX data returned for period {start_date} to {end_date}")
            
            # Clean and format the data
            vix_df = self._clean_vix_data(vix_data)
            
            if validate_data:
                self._validate_vix_data(vix_df, start_date, end_date)
            
            logger.info(f"Successfully fetched {len(vix_df)} VIX data points")
            return vix_df
            
        except Exception as e:
            logger.error(f"Failed to fetch VIX data: {str(e)}")
            raise RuntimeError(f"VIX data fetch failed: {str(e)}")
    
    def get_current_vix(self) -> float:
        """
        Get the most recent VIX value
        
        Returns:
            Most recent VIX close price
            
        Raises:
            RuntimeError: If current data fetch fails
        """
        try:
            # Get last 5 trading days to ensure we get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            vix_data = self.fetch_historical(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                validate_data=False
            )
            
            if vix_data.empty:
                raise RuntimeError("No recent VIX data available")
            
            current_vix = float(vix_data['vix_close'].iloc[-1])
            logger.info(f"Current VIX: {current_vix:.2f}")
            
            return current_vix
            
        except Exception as e:
            logger.error(f"Failed to get current VIX: {str(e)}")
            raise RuntimeError(f"Current VIX fetch failed: {str(e)}")
    
    def _clean_vix_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and format raw VIX data from Yahoo Finance
        
        Args:
            raw_data: Raw DataFrame from yfinance
            
        Returns:
            Cleaned DataFrame with standardized column names
        """
        # Create clean DataFrame
        vix_df = pd.DataFrame()
        
        # Reset index to make date a column
        raw_data = raw_data.reset_index()
        
        # Map columns to standardized names
        vix_df['date'] = pd.to_datetime(raw_data['Date']).dt.tz_localize(None)
        vix_df['vix_open'] = raw_data['Open'].astype(float)
        vix_df['vix_high'] = raw_data['High'].astype(float)
        vix_df['vix_low'] = raw_data['Low'].astype(float)
        vix_df['vix_close'] = raw_data['Close'].astype(float)
        vix_df['vix_volume'] = raw_data['Volume'].fillna(0).astype(int)
        
        # Remove any rows with NaN values in OHLC
        vix_df = vix_df.dropna(subset=['vix_open', 'vix_high', 'vix_low', 'vix_close'])
        
        # Sort by date
        vix_df = vix_df.sort_values('date').reset_index(drop=True)
        
        # Set date as index for easier time series operations
        vix_df.set_index('date', inplace=True)
        
        return vix_df
    
    def _validate_vix_data(self, vix_df: pd.DataFrame, start_date: str, end_date: str) -> None:
        """
        Validate VIX data quality
        
        Args:
            vix_df: Cleaned VIX DataFrame
            start_date: Expected start date
            end_date: Expected end date
            
        Raises:
            ValueError: If data quality issues are found
        """
        if len(vix_df) == 0:
            raise ValueError("VIX DataFrame is empty")
        
        # Check for reasonable VIX values (historically 5-80 range)
        vix_values = vix_df['vix_close']
        
        if vix_values.min() < 5:
            logger.warning(f"Very low VIX values found (min: {vix_values.min():.2f})")
        
        if vix_values.max() > 100:
            logger.warning(f"Very high VIX values found (max: {vix_values.max():.2f})")
        
        # Check for missing data gaps (more than 5 business days)
        date_diff = vix_df.index.to_series().diff()
        large_gaps = date_diff[date_diff > timedelta(days=7)]
        
        if len(large_gaps) > 0:
            logger.warning(f"Found {len(large_gaps)} large data gaps in VIX data")
        
        # Check data coverage
        expected_start = pd.to_datetime(start_date)
        expected_end = pd.to_datetime(end_date)
        
        actual_start = vix_df.index.min()
        actual_end = vix_df.index.max()
        
        if actual_start > expected_start + timedelta(days=7):
            logger.warning(f"VIX data starts later than expected: {actual_start} vs {expected_start}")
        
        if actual_end < expected_end - timedelta(days=7):
            logger.warning(f"VIX data ends earlier than expected: {actual_end} vs {expected_end}")
        
        logger.info(f"VIX data validation complete: {len(vix_df)} records from {actual_start.date()} to {actual_end.date()}")
    
    def get_vix_statistics(self, vix_df: pd.DataFrame) -> dict:
        """
        Calculate summary statistics for VIX data
        
        Args:
            vix_df: VIX DataFrame
            
        Returns:
            Dictionary with statistical information
        """
        vix_close = vix_df['vix_close']
        
        stats = {
            'count': len(vix_close),
            'mean': float(vix_close.mean()),
            'std': float(vix_close.std()),
            'min': float(vix_close.min()),
            'max': float(vix_close.max()),
            'median': float(vix_close.median()),
            'p25': float(vix_close.quantile(0.25)),
            'p75': float(vix_close.quantile(0.75)),
            'p90': float(vix_close.quantile(0.90)),
            'p95': float(vix_close.quantile(0.95)),
            'start_date': str(vix_df.index.min().date()),
            'end_date': str(vix_df.index.max().date())
        }
        
        return stats


def test_vix_fetcher():
    """Test function for VIX fetcher"""
    fetcher = VIXFetcher()
    
    # Test historical data fetch
    try:
        vix_data = fetcher.fetch_historical('2020-01-01', '2020-12-31')
        print(f"✓ Fetched {len(vix_data)} VIX records for 2020")
        
        # Test statistics
        stats = fetcher.get_vix_statistics(vix_data)
        print(f"✓ VIX stats: mean={stats['mean']:.2f}, max={stats['max']:.2f}")
        
        # Test current VIX
        current = fetcher.get_current_vix()
        print(f"✓ Current VIX: {current:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ VIX fetcher test failed: {e}")
        return False


if __name__ == "__main__":
    test_vix_fetcher()