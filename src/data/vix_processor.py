"""
VIX Data Processor for Feature Engineering

This module processes raw VIX data into machine learning features suitable
for reinforcement learning in the GBWM system.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VIXProcessor:
    """
    Process raw VIX data into ML-ready features
    
    Converts VIX time series into normalized features suitable for RL:
    - Sentiment indicators (fear vs greed)
    - Momentum indicators (short-term changes)
    - Regime indicators (low/normal/high volatility)
    """
    
    def __init__(self, long_term_mean: float = 20.0):
        """
        Initialize VIX processor
        
        Args:
            long_term_mean: Historical VIX long-term average for normalization
        """
        self.long_term_mean = long_term_mean
        
        # VIX regime thresholds (based on historical analysis)
        self.vix_low_threshold = 15.0    # Below 15: Low fear/complacency
        self.vix_high_threshold = 25.0   # Above 25: High fear/stress
        
    def calculate_features(self, vix_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive VIX features for ML
        
        Input DataFrame columns:
            - date: datetime index
            - vix_close: float (closing VIX values)
            
        Output DataFrame additional columns:
            - vix_normalized: VIX normalized to [0, 1] range
            - vix_centered: VIX centered around mean, clipped to [-1, 1]
            - vix_regime: Categorical regime ('LOW', 'NORMAL', 'HIGH')
            - vix_regime_numeric: Numeric regime (-1, 0, 1)
            - vix_change_1d: 1-day percentage change
            - vix_change_5d: 5-day percentage change
            - vix_change_20d: 20-day percentage change
            - vix_sma_5: 5-day simple moving average
            - vix_sma_20: 20-day simple moving average
            - vix_sma_50: 50-day simple moving average
            - vix_volatility_20d: 20-day rolling volatility
            - vix_percentile: Historical percentile rank [0, 1]
            
        Returns:
            DataFrame with all original columns plus calculated features
        """
        # Create working copy
        result_df = vix_df.copy()
        
        # Ensure we have the required column
        if 'vix_close' not in result_df.columns:
            raise ValueError("Input DataFrame must contain 'vix_close' column")
        
        vix_close = result_df['vix_close']
        
        # 1. Basic normalization features
        result_df['vix_normalized'] = self._normalize_vix(vix_close)
        result_df['vix_centered'] = self._center_vix(vix_close)
        
        # 2. Regime classification
        result_df['vix_regime'] = self._classify_regime(vix_close)
        result_df['vix_regime_numeric'] = self._regime_to_numeric(result_df['vix_regime'])
        
        # 3. Change indicators (momentum)
        result_df['vix_change_1d'] = vix_close.pct_change(1)
        result_df['vix_change_5d'] = vix_close.pct_change(5)
        result_df['vix_change_20d'] = vix_close.pct_change(20)
        result_df['vix_change_90d'] = vix_close.pct_change(90)  # 3-4 month momentum for annual decisions
        
        # 4. Moving averages (trend)
        result_df['vix_sma_5'] = vix_close.rolling(window=5, min_periods=1).mean()
        result_df['vix_sma_20'] = vix_close.rolling(window=20, min_periods=1).mean()
        result_df['vix_sma_50'] = vix_close.rolling(window=50, min_periods=1).mean()
        
        # 5. Volatility of VIX (meta-volatility)
        result_df['vix_volatility_20d'] = vix_close.rolling(window=20, min_periods=1).std()
        
        # 6. Percentile ranking
        result_df['vix_percentile'] = self._calculate_percentile_rank(vix_close)
        
        # Fill any remaining NaN values
        result_df = self._handle_missing_values(result_df)
        
        logger.info(f"Calculated VIX features for {len(result_df)} records")
        return result_df
    
    def get_ml_features(self, vix_df: pd.DataFrame, date: pd.Timestamp,
                        momentum_lookback: int = 90) -> np.ndarray:
        """
        Extract ML-ready features for a specific date

        Args:
            vix_df: Processed VIX DataFrame with features
            date: Target date for feature extraction
            momentum_lookback: Days for momentum calculation (default 90 = ~3-4 months)
                             Use 90 for annual decisions, 5 for daily decisions

        Returns:
            np.ndarray of shape (2,): [vix_sentiment, vix_momentum]
            - vix_sentiment: float in [-1, 1] (fear/greed indicator)
            - vix_momentum: float in [-1, 1] (momentum normalized)
        """
        # Find closest date if exact match not available
        closest_date = self._find_closest_date(vix_df, date)

        if closest_date is None:
            logger.warning(f"No VIX data available near {date}, returning neutral features")
            return np.array([0.0, 0.0], dtype=np.float32)

        row = vix_df.loc[closest_date]

        # Sentiment: Use centered VIX (high VIX = fear = negative sentiment)
        vix_sentiment = -row['vix_centered']  # Invert: high VIX = negative sentiment

        # Momentum: Use appropriate lookback period
        # For annual decisions, use 90-day (3-4 month) momentum
        # This captures meaningful trend changes relevant to annual investment decisions
        momentum_col = f'vix_change_{momentum_lookback}d'

        if momentum_col in row.index and not pd.isna(row[momentum_col]):
            vix_change = row[momentum_col]
        elif 'vix_change_90d' in row.index and not pd.isna(row['vix_change_90d']):
            # Fall back to 90-day if available
            vix_change = row['vix_change_90d']
        elif 'vix_change_20d' in row.index and not pd.isna(row['vix_change_20d']):
            # Fall back to 20-day
            vix_change = row['vix_change_20d']
        elif 'vix_change_5d' in row.index and not pd.isna(row['vix_change_5d']):
            # Fall back to 5-day
            vix_change = row['vix_change_5d']
        else:
            vix_change = 0.0

        # Normalize to [-1, 1]: 50% change = full range
        vix_momentum = np.clip(vix_change / 0.5, -1.0, 1.0)

        features = np.array([vix_sentiment, vix_momentum], dtype=np.float32)

        return features
    
    def _normalize_vix(self, vix_series: pd.Series) -> pd.Series:
        """
        Normalize VIX to [0, 1] range
        
        Maps typical VIX range (10-80) to [0, 1]
        """
        # Use robust normalization based on historical range
        vix_min = 10.0  # Historical minimum (rarely below this)
        vix_max = 80.0  # Historical maximum (2008 crisis peak was ~80)
        
        normalized = (vix_series - vix_min) / (vix_max - vix_min)
        return np.clip(normalized, 0.0, 1.0)
    
    def _center_vix(self, vix_series: pd.Series) -> pd.Series:
        """
        Center VIX around long-term mean and clip to [-1, 1]
        
        Positive values indicate above-average fear
        Negative values indicate below-average fear (complacency)
        """
        # Center around long-term mean (typically 20)
        centered = (vix_series - self.long_term_mean) / 30.0  # ±30 points = full range
        return np.clip(centered, -1.0, 1.0)
    
    def _classify_regime(self, vix_series: pd.Series) -> pd.Series:
        """
        Classify VIX into market regimes
        
        Returns:
            Series with values 'LOW', 'NORMAL', 'HIGH'
        """
        conditions = [
            vix_series < self.vix_low_threshold,
            vix_series > self.vix_high_threshold
        ]
        choices = ['LOW', 'HIGH']
        
        regime = np.select(conditions, choices, default='NORMAL')
        return pd.Series(regime, index=vix_series.index)
    
    def _regime_to_numeric(self, regime_series: pd.Series) -> pd.Series:
        """
        Convert regime to numeric values
        
        LOW -> -1 (complacency)
        NORMAL -> 0 (neutral)
        HIGH -> 1 (fear)
        """
        mapping = {'LOW': -1, 'NORMAL': 0, 'HIGH': 1}
        return regime_series.map(mapping)
    
    def _calculate_percentile_rank(self, vix_series: pd.Series) -> pd.Series:
        """
        Calculate rolling percentile rank over full history
        
        Returns values in [0, 1] where 1 = highest VIX in history
        """
        # Use expanding window to get percentile relative to all historical data
        percentiles = vix_series.expanding(min_periods=20).rank(pct=True)
        return percentiles.fillna(0.5)  # Neutral for early periods
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in calculated features
        
        Uses forward fill and neutral values for critical features
        """
        result = df.copy()
        
        # Forward fill most features
        ffill_columns = [
            'vix_sma_5', 'vix_sma_20', 'vix_sma_50', 'vix_volatility_20d'
        ]
        for col in ffill_columns:
            if col in result.columns:
                result[col] = result[col].ffill()
        
        # Fill change indicators with 0 (neutral)
        change_columns = ['vix_change_1d', 'vix_change_5d', 'vix_change_20d', 'vix_change_90d']
        for col in change_columns:
            if col in result.columns:
                result[col] = result[col].fillna(0.0)
        
        # Fill remaining with neutral values
        result['vix_normalized'] = result['vix_normalized'].fillna(0.5)
        result['vix_centered'] = result['vix_centered'].fillna(0.0)
        result['vix_percentile'] = result['vix_percentile'].fillna(0.5)
        
        return result
    
    def _find_closest_date(self, vix_df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Find the closest available date in VIX data
        
        Args:
            vix_df: DataFrame with datetime index
            target_date: Target date to find
            
        Returns:
            Closest available date or None if no data
        """
        if len(vix_df) == 0:
            return None
        
        # Get date index
        dates = vix_df.index
        
        # Find closest date
        date_diffs = np.abs(dates - target_date)
        closest_idx = date_diffs.argmin()
        closest_date = dates[closest_idx]
        
        # Check if it's reasonably close (within 30 days)
        if date_diffs[closest_idx].days > 30:
            logger.warning(f"Closest VIX date {closest_date.date()} is {date_diffs[closest_idx].days} days from target {target_date.date()}")
        
        return closest_date
    
    def get_feature_summary(self, vix_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all VIX features
        
        Args:
            vix_df: Processed VIX DataFrame
            
        Returns:
            Dictionary with feature statistics
        """
        feature_columns = [
            'vix_close', 'vix_normalized', 'vix_centered', 'vix_regime_numeric',
            'vix_change_5d', 'vix_sma_20', 'vix_volatility_20d', 'vix_percentile'
        ]
        
        summary = {}
        for col in feature_columns:
            if col in vix_df.columns:
                series = vix_df[col]
                summary[col] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'missing_pct': float(series.isna().mean() * 100)
                }
        
        # Regime distribution
        if 'vix_regime' in vix_df.columns:
            regime_dist = vix_df['vix_regime'].value_counts(normalize=True).to_dict()
            summary['regime_distribution'] = regime_dist
        
        return summary


def test_vix_processor():
    """Test function for VIX processor"""
    # Create sample VIX data for testing
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    
    # Generate realistic VIX data with different regimes
    np.random.seed(42)
    base_vix = 20 + np.cumsum(np.random.randn(len(dates)) * 0.5)  # Random walk around 20
    
    # Add some volatility spikes
    spike_dates = np.random.choice(len(dates), size=10, replace=False)
    base_vix[spike_dates] += np.random.uniform(15, 40, size=10)
    
    # Ensure positive values
    base_vix = np.maximum(base_vix, 5.0)
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'vix_close': base_vix
    }, index=dates)
    
    # Test processor
    processor = VIXProcessor()
    
    try:
        # Test feature calculation
        processed_df = processor.calculate_features(test_df)
        print(f"✓ Calculated features for {len(processed_df)} records")
        
        # Test ML feature extraction
        target_date = pd.Timestamp('2020-06-15')
        features = processor.get_ml_features(processed_df, target_date)
        print(f"✓ ML features for {target_date.date()}: {features}")
        
        # Test feature summary
        summary = processor.get_feature_summary(processed_df)
        print(f"✓ Feature summary generated with {len(summary)} metrics")
        
        # Verify feature ranges
        assert -1 <= features[0] <= 1, "VIX sentiment out of range"
        assert -1 <= features[1] <= 1, "VIX momentum out of range"
        print("✓ All feature ranges validated")
        
        return True
        
    except Exception as e:
        print(f"✗ VIX processor test failed: {e}")
        return False


if __name__ == "__main__":
    test_vix_processor()