"""
Cache Manager for Sentiment Data

This module manages caching of sentiment data (VIX, etc.) to avoid
repeated API calls and improve performance.
"""

import pickle
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manage sentiment data caching for efficient data access
    
    Provides persistent caching of VIX data and other sentiment indicators
    with automatic cache invalidation and update mechanisms.
    """
    
    def __init__(self, cache_dir: str = './data/sentiment'):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file names
        self.vix_cache_file = self.cache_dir / 'vix_data.pkl'
        self.vix_metadata_file = self.cache_dir / 'vix_metadata.json'
        self.config_file = self.cache_dir / 'cache_config.json'
        
        # Default cache settings
        self.default_cache_hours = 24  # Refresh daily
        self.max_cache_age_days = 7   # Force refresh after a week
        
        # Initialize cache config
        self._initialize_config()
        
    def save_vix_cache(
        self, 
        vix_df: pd.DataFrame, 
        source_info: Optional[Dict] = None
    ) -> None:
        """
        Save VIX DataFrame to cache with metadata
        
        Args:
            vix_df: VIX DataFrame to cache
            source_info: Additional information about data source
        """
        try:
            # Prepare metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'record_count': len(vix_df),
                'date_range': {
                    'start': str(vix_df.index.min().date()) if len(vix_df) > 0 else None,
                    'end': str(vix_df.index.max().date()) if len(vix_df) > 0 else None
                },
                'columns': list(vix_df.columns),
                'source_info': source_info or {}
            }
            
            # Save data
            with open(self.vix_cache_file, 'wb') as f:
                pickle.dump(vix_df, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            with open(self.vix_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"VIX cache saved: {len(vix_df)} records from {metadata['date_range']['start']} to {metadata['date_range']['end']}")
            
        except Exception as e:
            logger.error(f"Failed to save VIX cache: {str(e)}")
            raise RuntimeError(f"Cache save failed: {str(e)}")
    
    def load_vix_cache(self, max_age_hours: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Load VIX cache if it exists and is fresh enough
        
        Args:
            max_age_hours: Maximum cache age in hours (uses default if None)
            
        Returns:
            VIX DataFrame if cache is valid, None otherwise
        """
        try:
            # Check if cache files exist
            if not (self.vix_cache_file.exists() and self.vix_metadata_file.exists()):
                logger.info("VIX cache files not found")
                return None
            
            # Load metadata
            with open(self.vix_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check cache age
            cache_time = datetime.fromisoformat(metadata['timestamp'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            max_age = max_age_hours if max_age_hours is not None else self.default_cache_hours
            
            if age_hours > max_age:
                logger.info(f"VIX cache expired: {age_hours:.1f} hours > {max_age} hours")
                return None
            
            # Load data
            with open(self.vix_cache_file, 'rb') as f:
                vix_df = pickle.load(f)
            
            logger.info(f"VIX cache loaded: {len(vix_df)} records, age: {age_hours:.1f} hours")
            return vix_df
            
        except Exception as e:
            logger.warning(f"Failed to load VIX cache: {str(e)}")
            return None
    
    def get_vix_cache_info(self) -> Optional[Dict]:
        """
        Get information about cached VIX data
        
        Returns:
            Dictionary with cache information or None if no cache
        """
        try:
            if not self.vix_metadata_file.exists():
                return None
            
            with open(self.vix_metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Add age information
            cache_time = datetime.fromisoformat(metadata['timestamp'])
            age_hours = (datetime.now() - cache_time).total_seconds() / 3600
            
            metadata['age_hours'] = age_hours
            metadata['is_fresh'] = age_hours <= self.default_cache_hours
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to get VIX cache info: {str(e)}")
            return None
    
    def invalidate_vix_cache(self) -> None:
        """
        Remove VIX cache files to force refresh
        """
        try:
            if self.vix_cache_file.exists():
                self.vix_cache_file.unlink()
                logger.info("VIX data cache file removed")
            
            if self.vix_metadata_file.exists():
                self.vix_metadata_file.unlink()
                logger.info("VIX metadata cache file removed")
                
        except Exception as e:
            logger.error(f"Failed to invalidate VIX cache: {str(e)}")
    
    def cleanup_old_cache(self, max_age_days: Optional[int] = None) -> None:
        """
        Remove cache files older than specified age
        
        Args:
            max_age_days: Maximum age in days (uses default if None)
        """
        max_age = max_age_days if max_age_days is not None else self.max_cache_age_days
        cutoff_time = datetime.now() - timedelta(days=max_age)
        
        cache_files = list(self.cache_dir.glob('*.pkl')) + list(self.cache_dir.glob('*.json'))
        
        removed_count = 0
        for cache_file in cache_files:
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_time < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed old cache file: {cache_file.name}")
                    
            except Exception as e:
                logger.warning(f"Failed to check/remove cache file {cache_file.name}: {str(e)}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old cache files")
    
    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get cache directory size and file information
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_size = 0
            file_count = 0
            file_info = {}
            
            for cache_file in self.cache_dir.rglob('*'):
                if cache_file.is_file():
                    file_size = cache_file.stat().st_size
                    total_size += file_size
                    file_count += 1
                    
                    file_info[cache_file.name] = {
                        'size_bytes': file_size,
                        'size_mb': file_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(cache_file.stat().st_mtime).isoformat()
                    }
            
            return {
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'files': file_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache size: {str(e)}")
            return {'error': str(e)}
    
    def _initialize_config(self) -> None:
        """
        Initialize cache configuration file
        """
        default_config = {
            'default_cache_hours': self.default_cache_hours,
            'max_cache_age_days': self.max_cache_age_days,
            'auto_cleanup_enabled': True,
            'created': datetime.now().isoformat()
        }
        
        if not self.config_file.exists():
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.debug(f"Cache config initialized: {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to create cache config: {str(e)}")
        
        # Load existing config
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            # Update settings from config
            self.default_cache_hours = config.get('default_cache_hours', self.default_cache_hours)
            self.max_cache_age_days = config.get('max_cache_age_days', self.max_cache_age_days)
            
        except Exception as e:
            logger.warning(f"Failed to load cache config: {str(e)}")
    
    def update_config(self, **kwargs) -> None:
        """
        Update cache configuration
        
        Args:
            **kwargs: Configuration parameters to update
        """
        try:
            # Load current config
            config = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            
            # Update with new values
            config.update(kwargs)
            config['last_updated'] = datetime.now().isoformat()
            
            # Save updated config
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update instance variables
            if 'default_cache_hours' in kwargs:
                self.default_cache_hours = kwargs['default_cache_hours']
            if 'max_cache_age_days' in kwargs:
                self.max_cache_age_days = kwargs['max_cache_age_days']
            
            logger.info(f"Cache config updated: {list(kwargs.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to update cache config: {str(e)}")


def test_cache_manager():
    """Test function for cache manager"""
    # Create test cache directory
    test_cache_dir = './test_cache'
    cache_manager = CacheManager(test_cache_dir)
    
    try:
        # Create test VIX data
        test_dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        test_vix_df = pd.DataFrame({
            'vix_close': [20.5, 21.2, 19.8, 22.1, 23.5, 20.0, 18.9, 21.7, 22.3, 20.8],
            'vix_normalized': [0.4, 0.42, 0.38, 0.44, 0.47, 0.4, 0.36, 0.43, 0.45, 0.42]
        }, index=test_dates)
        
        # Test save
        cache_manager.save_vix_cache(test_vix_df, {'source': 'test'})
        print("✓ VIX cache saved successfully")
        
        # Test load
        loaded_df = cache_manager.load_vix_cache()
        assert loaded_df is not None, "Failed to load cache"
        assert len(loaded_df) == len(test_vix_df), "Loaded data size mismatch"
        print("✓ VIX cache loaded successfully")
        
        # Test cache info
        cache_info = cache_manager.get_vix_cache_info()
        assert cache_info is not None, "Failed to get cache info"
        assert cache_info['record_count'] == len(test_vix_df), "Cache info mismatch"
        print("✓ Cache info retrieved successfully")
        
        # Test cache size
        size_info = cache_manager.get_cache_size()
        assert size_info['file_count'] > 0, "No cache files found"
        print(f"✓ Cache size: {size_info['total_size_mb']:.2f} MB")
        
        # Test invalidation
        cache_manager.invalidate_vix_cache()
        loaded_df_after = cache_manager.load_vix_cache()
        assert loaded_df_after is None, "Cache not properly invalidated"
        print("✓ Cache invalidation successful")
        
        # Cleanup test directory
        import shutil
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)
        print("✓ Test cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Cache manager test failed: {e}")
        # Cleanup on failure
        import shutil
        if os.path.exists(test_cache_dir):
            shutil.rmtree(test_cache_dir)
        return False


if __name__ == "__main__":
    test_cache_manager()