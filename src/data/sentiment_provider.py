"""
Sentiment Provider for GBWM System

This module provides the main interface for market sentiment data in the
Goals-Based Wealth Management reinforcement learning system.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Dict, Optional, Tuple
import logging
import warnings

from src.data.vix_fetcher import VIXFetcher
from src.data.vix_processor import VIXProcessor
from src.data.cache_manager import CacheManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentProvider:
    """
    Unified sentiment data provider for GBWM
    
    Provides market sentiment features based on VIX and optional news sentiment.
    Handles data fetching, caching, processing, and feature extraction for RL.
    """
    
    def __init__(
        self,
        cache_dir: str = './data/sentiment',
        vix_weight: float = 1.0,
        news_weight: float = 0.0,
        long_term_vix_mean: float = 20.0,
        lookback_days: int = 365
    ):
        """
        Initialize sentiment provider
        
        Args:
            cache_dir: Directory for caching sentiment data
            vix_weight: Weight for VIX component (0-1)
            news_weight: Weight for news component (0-1)
            long_term_vix_mean: Historical VIX average for normalization
            lookback_days: Default historical data window
        """
        self.cache_dir = cache_dir
        self.vix_weight = vix_weight
        self.news_weight = news_weight
        self.long_term_vix_mean = long_term_vix_mean
        self.lookback_days = lookback_days
        
        # Initialize components
        self.vix_fetcher = VIXFetcher()
        self.vix_processor = VIXProcessor(long_term_mean=long_term_vix_mean)
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Data storage
        self.vix_data: Optional[pd.DataFrame] = None
        self.processed_vix_data: Optional[pd.DataFrame] = None
        self.is_initialized = False
        
        # Validate weights
        total_weight = vix_weight + news_weight
        if total_weight <= 0:
            raise ValueError("At least one sentiment weight must be positive")
        
        # Normalize weights
        self.vix_weight = vix_weight / total_weight
        self.news_weight = news_weight / total_weight
        
        logger.info(f"SentimentProvider initialized: VIX weight={self.vix_weight:.2f}, News weight={self.news_weight:.2f}")
    
    def initialize(self, lookback_days: Optional[int] = None, force_refresh: bool = False) -> bool:
        """
        Initialize provider by fetching/loading data

        Args:
            lookback_days: Historical data window (uses default if None)
            force_refresh: Force refresh of cached data

        Returns:
            True if successful, False otherwise
        """
        try:
            lookback = lookback_days if lookback_days is not None else self.lookback_days

            logger.info(f"Initializing sentiment provider with {lookback} days lookback")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback + 30)  # Extra buffer for weekends

            # Try to load from cache first
            if not force_refresh:
                cached_vix = self.cache_manager.load_vix_cache(max_age_hours=24)
                if cached_vix is not None and self._validate_cached_data(cached_vix, start_date):
                    logger.info("Using cached VIX data")
                    self.vix_data = cached_vix
                    self.processed_vix_data = self._ensure_processed_features(cached_vix)
                    self.is_initialized = True
                    return True

            # Fetch fresh data if cache miss or forced refresh
            logger.info(f"Fetching fresh VIX data from {start_date.date()} to {end_date.date()}")
            self.vix_data = self.vix_fetcher.fetch_historical(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            # Process features
            self.processed_vix_data = self.vix_processor.calculate_features(self.vix_data)

            # Cache the processed data
            self.cache_manager.save_vix_cache(
                self.processed_vix_data,
                source_info={
                    'fetched_at': datetime.now().isoformat(),
                    'lookback_days': lookback,
                    'vix_weight': self.vix_weight,
                    'news_weight': self.news_weight
                }
            )

            self.is_initialized = True
            logger.info(f"Sentiment provider initialized successfully with {len(self.processed_vix_data)} VIX records")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize sentiment provider: {str(e)}")
            self.is_initialized = False
            return False

    def initialize_historical(self, start_date: str = '1990-01-02', end_date: Optional[str] = None,
                              force_refresh: bool = False) -> bool:
        """
        Initialize provider with historical VIX data for a specific date range.

        Use this method when running simulations that need historical VIX data
        (e.g., backtesting from 1990 to present).

        Args:
            start_date: Start date in YYYY-MM-DD format (default: 1990-01-02, VIX inception)
            end_date: End date in YYYY-MM-DD format (default: today)
            force_refresh: Force refresh of cached data

        Returns:
            True if successful, False otherwise
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            logger.info(f"Initializing sentiment provider with historical data from {start_date} to {end_date}")

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Try to load from cache first (check if cache covers the required range)
            if not force_refresh:
                cached_vix = self.cache_manager.load_vix_cache(max_age_hours=168)  # 1 week for historical data
                if cached_vix is not None:
                    cache_start = cached_vix.index.min()
                    cache_end = cached_vix.index.max()
                    # Check if cache covers the required range (with 30-day tolerance)
                    if cache_start <= start_dt + timedelta(days=30) and cache_end >= end_dt - timedelta(days=7):
                        logger.info(f"Using cached historical VIX data ({len(cached_vix)} records)")
                        self.vix_data = cached_vix
                        self.processed_vix_data = self._ensure_processed_features(cached_vix)
                        self.is_initialized = True
                        return True
                    else:
                        logger.info(f"Cache range ({cache_start.date()} to {cache_end.date()}) "
                                    f"doesn't cover required range ({start_date} to {end_date})")

            # Fetch fresh historical data
            logger.info(f"Fetching historical VIX data from {start_date} to {end_date}")
            self.vix_data = self.vix_fetcher.fetch_historical(start_date, end_date)

            # Process features
            self.processed_vix_data = self.vix_processor.calculate_features(self.vix_data)

            # Cache the processed data (non-blocking - failure doesn't affect initialization)
            try:
                self.cache_manager.save_vix_cache(
                    self.processed_vix_data,
                    source_info={
                        'fetched_at': datetime.now().isoformat(),
                        'start_date': start_date,
                        'end_date': end_date,
                        'is_historical': True,
                        'vix_weight': self.vix_weight,
                        'news_weight': self.news_weight
                    }
                )
            except Exception as cache_error:
                logger.warning(f"Failed to save VIX cache (non-critical): {cache_error}")

            self.is_initialized = True
            logger.info(f"Sentiment provider initialized with {len(self.processed_vix_data)} historical VIX records "
                        f"from {self.processed_vix_data.index.min().date()} to {self.processed_vix_data.index.max().date()}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize sentiment provider with historical data: {str(e)}")
            self.is_initialized = False
            return False
    
    def get_sentiment_features(self, date: Union[str, datetime]) -> np.ndarray:
        """
        Get sentiment features for specific date
        
        Args:
            date: Target date (YYYY-MM-DD string or datetime)
            
        Returns:
            np.ndarray of shape (2,): [vix_sentiment, vix_momentum]
            - vix_sentiment: float in [-1, 1], where -1=extreme fear, 1=extreme greed
            - vix_momentum: float in [-1, 1], normalized 5-day change
            
        Raises:
            RuntimeError: If provider not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("SentimentProvider not initialized. Call initialize() first.")
        
        try:
            # Convert to datetime if needed
            if isinstance(date, str):
                target_date = pd.to_datetime(date)
            else:
                target_date = pd.to_datetime(date)
            
            # Get VIX features
            vix_features = self.vix_processor.get_ml_features(self.processed_vix_data, target_date)
            
            # Currently only using VIX (news sentiment could be added later)
            sentiment_features = vix_features * self.vix_weight
            
            # Ensure proper shape and type
            sentiment_features = np.array(sentiment_features, dtype=np.float32)
            
            return sentiment_features
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment features for {date}: {str(e)}")
            # Return neutral sentiment on error
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def validate_data_coverage(self, start_date: str, end_date: str, episode_years: int = 16) -> bool:
        """
        Validate that VIX data covers the required date range for episode training
        
        Args:
            start_date: Required start date (YYYY-MM-DD)
            end_date: Required end date (YYYY-MM-DD) 
            episode_years: Length of episodes in years
            
        Returns:
            True if data coverage is sufficient, False otherwise
        """
        if not self.is_initialized:
            logger.warning("SentimentProvider not initialized, cannot validate coverage")
            return False
        
        try:
            required_start = pd.to_datetime(start_date)
            required_end = pd.to_datetime(end_date)
            
            # Check actual data coverage
            data_start = self.processed_vix_data.index.min()
            data_end = self.processed_vix_data.index.max()
            
            # Calculate required buffer for episode sequences
            buffer_days = episode_years * 365 + 30  # Extra buffer for weekends/holidays
            
            # Validate coverage with buffer
            coverage_sufficient = (
                data_start <= required_start - pd.Timedelta(days=30) and
                data_end >= required_end + pd.Timedelta(days=30)
            )
            
            if not coverage_sufficient:
                logger.error(f"Insufficient VIX data coverage:")
                logger.error(f"  Required: {required_start.date()} to {required_end.date()}")
                logger.error(f"  Available: {data_start.date()} to {data_end.date()}")
                logger.error(f"  Buffer needed: {buffer_days} days for {episode_years}-year episodes")
                
            return coverage_sufficient
            
        except Exception as e:
            logger.error(f"Failed to validate data coverage: {e}")
            return False
    
    def get_sentiment_info(self, date: Union[str, datetime]) -> Dict:
        """
        Get detailed sentiment information for logging
        
        Args:
            date: Target date
            
        Returns:
            Dictionary with keys:
                - 'date': datetime
                - 'vix_raw': float (raw VIX value)
                - 'vix_sentiment': float (normalized)
                - 'vix_regime': str ('LOW_FEAR'|'NORMAL'|'HIGH_FEAR')
                - 'vix_percentile': float (0-1)
                - 'features': np.ndarray (ML features)
        """
        if not self.is_initialized:
            raise RuntimeError("SentimentProvider not initialized. Call initialize() first.")
        
        try:
            # Convert to datetime if needed
            if isinstance(date, str):
                target_date = pd.to_datetime(date)
            else:
                target_date = pd.to_datetime(date)
            
            # Find closest date in data
            closest_date = self.vix_processor._find_closest_date(self.processed_vix_data, target_date)
            
            if closest_date is None:
                return {
                    'date': target_date,
                    'vix_raw': None,
                    'vix_sentiment': 0.0,
                    'vix_regime': 'UNKNOWN',
                    'vix_percentile': 0.5,
                    'features': np.array([0.0, 0.0])
                }
            
            # Get row data
            row = self.processed_vix_data.loc[closest_date]
            
            # Map regime codes to names
            regime_map = {-1: 'LOW_FEAR', 0: 'NORMAL', 1: 'HIGH_FEAR'}
            regime_name = regime_map.get(row.get('vix_regime_numeric', 0), 'NORMAL')
            
            # Get features
            features = self.get_sentiment_features(target_date)
            
            info = {
                'date': target_date,
                'closest_data_date': closest_date,
                'vix_raw': float(row['vix_close']),
                'vix_sentiment': float(features[0]),
                'vix_momentum': float(features[1]),
                'vix_regime': regime_name,
                'vix_percentile': float(row.get('vix_percentile', 0.5)),
                'vix_normalized': float(row.get('vix_normalized', 0.5)),
                'vix_centered': float(row.get('vix_centered', 0.0)),
                'features': features
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get sentiment info for {date}: {str(e)}")
            return {
                'date': target_date,
                'error': str(e),
                'features': np.array([0.0, 0.0])
            }
    
    def update_cache(self, force: bool = False) -> None:
        """
        Update cached VIX data
        
        Args:
            force: Force update even if cache is recent
        """
        try:
            logger.info(f"Updating sentiment cache (force={force})")
            
            # Check if update needed
            if not force:
                cache_info = self.cache_manager.get_vix_cache_info()
                if cache_info and cache_info.get('is_fresh', False):
                    logger.info("Cache is fresh, skipping update")
                    return
            
            # Re-initialize with fresh data
            self.initialize(force_refresh=True)
            
        except Exception as e:
            logger.error(f"Failed to update cache: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive statistics about sentiment data
        
        Returns:
            Dictionary with sentiment statistics
        """
        if not self.is_initialized:
            return {'error': 'Provider not initialized'}
        
        try:
            stats = {}
            
            # VIX statistics
            if self.processed_vix_data is not None:
                vix_stats = self.vix_processor.get_feature_summary(self.processed_vix_data)
                stats['vix'] = vix_stats
            
            # Cache statistics
            cache_info = self.cache_manager.get_vix_cache_info()
            if cache_info:
                stats['cache'] = cache_info
            
            # Provider configuration
            stats['config'] = {
                'vix_weight': self.vix_weight,
                'news_weight': self.news_weight,
                'long_term_vix_mean': self.long_term_vix_mean,
                'lookback_days': self.lookback_days,
                'is_initialized': self.is_initialized
            }
            
            # Recent features (last 5 trading days)
            if self.processed_vix_data is not None and len(self.processed_vix_data) > 0:
                recent_dates = self.processed_vix_data.index[-5:]
                recent_features = []
                for date in recent_dates:
                    features = self.get_sentiment_features(date)
                    recent_features.append({
                        'date': str(date.date()),
                        'vix_sentiment': float(features[0]),
                        'vix_momentum': float(features[1])
                    })
                stats['recent_features'] = recent_features
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {'error': str(e)}
    
    def _validate_cached_data(self, cached_data: pd.DataFrame, required_start_date: datetime) -> bool:
        """
        Validate that cached data meets requirements
        
        Args:
            cached_data: Cached VIX DataFrame
            required_start_date: Minimum required start date
            
        Returns:
            True if data is sufficient, False otherwise
        """
        if cached_data is None or len(cached_data) == 0:
            return False
        
        # Check date coverage
        data_start = cached_data.index.min()
        if data_start > required_start_date + timedelta(days=7):
            logger.info(f"Cached data starts too late: {data_start.date()} > {required_start_date.date()}")
            return False
        
        # Check data recency (last 7 days)
        data_end = cached_data.index.max()
        if data_end < datetime.now() - timedelta(days=7):
            logger.info(f"Cached data is too old: {data_end.date()}")
            return False
        
        # Check required columns
        required_columns = ['vix_close']
        if not all(col in cached_data.columns for col in required_columns):
            logger.info("Cached data missing required columns")
            return False
        
        return True
    
    def _ensure_processed_features(self, vix_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure VIX data has all processed features
        
        Args:
            vix_data: Raw or partially processed VIX data
            
        Returns:
            Fully processed VIX data with all features
        """
        required_features = ['vix_centered', 'vix_change_5d', 'vix_regime']
        
        if all(col in vix_data.columns for col in required_features):
            return vix_data
        else:
            logger.info("Reprocessing VIX features")
            return self.vix_processor.calculate_features(vix_data)


def test_sentiment_provider():
    """Test function for sentiment provider"""
    try:
        # Initialize provider
        provider = SentimentProvider(
            cache_dir='./test_sentiment_cache',
            vix_weight=1.0,
            news_weight=1.0,
            lookback_days=30
        )
        
        # Test initialization
        success = provider.initialize()
        assert success, "Failed to initialize sentiment provider"
        print("✓ Sentiment provider initialized")
        
        # Test feature extraction
        test_date = datetime.now() - timedelta(days=5)
        features = provider.get_sentiment_features(test_date)
        assert features.shape == (2,), f"Wrong feature shape: {features.shape}"
        assert -1 <= features[0] <= 1, f"VIX sentiment out of range: {features[0]}"
        assert -1 <= features[1] <= 1, f"VIX momentum out of range: {features[1]}"
        print(f"✓ Features extracted: {features}")
        
        # Test detailed info
        info = provider.get_sentiment_info(test_date)
        assert 'vix_raw' in info, "Missing VIX raw value in info"
        assert 'vix_regime' in info, "Missing VIX regime in info"
        print(f"✓ Sentiment info: VIX={info['vix_raw']:.2f}, regime={info['vix_regime']}")
        
        # Test statistics
        stats = provider.get_statistics()
        assert 'vix' in stats, "Missing VIX statistics"
        assert 'config' in stats, "Missing config in statistics"
        print("✓ Statistics generated")
        
        # Cleanup
        import shutil
        import os
        if os.path.exists('./test_sentiment_cache'):
            shutil.rmtree('./test_sentiment_cache')
        
        return True
        
    except Exception as e:
        print(f"✗ Sentiment provider test failed: {e}")
        # Cleanup on failure
        import shutil
        import os
        if os.path.exists('./test_sentiment_cache'):
            shutil.rmtree('./test_sentiment_cache')
        return False


if __name__ == "__main__":
    test_sentiment_provider()