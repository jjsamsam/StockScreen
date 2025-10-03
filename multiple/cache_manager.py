"""
Centralized caching system for stock data and technical indicators
Optimizes performance by avoiding redundant API calls and calculations
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import pickle
import os
from pathlib import Path
import hashlib


class StockDataCache:
    """Singleton cache manager for stock data"""

    _instance = None
    _cache = {}
    _cache_dir = Path("cache")
    _max_cache_age_hours = 24

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StockDataCache, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize cache directory"""
        self._cache_dir.mkdir(exist_ok=True)
        self._memory_cache = {}
        self._indicator_cache = {}

    def get_stock_data(self, symbol, period="1y", interval="1d", force_refresh=False):
        """
        Get stock data with caching

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force refresh from API

        Returns:
            DataFrame with stock data
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Check memory cache first
        if not force_refresh and cache_key in self._memory_cache:
            cached_data, timestamp = self._memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=self._max_cache_age_hours):
                return cached_data

        # Check disk cache
        if not force_refresh:
            disk_data = self._load_from_disk(cache_key)
            if disk_data is not None:
                self._memory_cache[cache_key] = (disk_data, datetime.now())
                return disk_data

        # Fetch fresh data from API
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data is not None and not data.empty:
                # Cache in memory and disk
                self._memory_cache[cache_key] = (data, datetime.now())
                self._save_to_disk(cache_key, data)
                return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

        return None

    def get_ticker_info(self, symbol, force_refresh=False):
        """
        Get ticker info with caching

        Args:
            symbol: Stock ticker symbol
            force_refresh: Force refresh from API

        Returns:
            Dictionary with ticker info
        """
        cache_key = f"{symbol}_info"

        # Check memory cache
        if not force_refresh and cache_key in self._memory_cache:
            cached_data, timestamp = self._memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=self._max_cache_age_hours):
                return cached_data

        # Check disk cache
        if not force_refresh:
            disk_data = self._load_from_disk(cache_key)
            if disk_data is not None:
                self._memory_cache[cache_key] = (disk_data, datetime.now())
                return disk_data

        # Fetch fresh data
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info:
                self._memory_cache[cache_key] = (info, datetime.now())
                self._save_to_disk(cache_key, info)
                return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {e}")

        return {}

    def cache_technical_indicator(self, symbol, indicator_name, data, params=None):
        """
        Cache calculated technical indicator

        Args:
            symbol: Stock ticker symbol
            indicator_name: Name of the indicator (e.g., 'SMA', 'RSI', 'MACD')
            data: Calculated indicator data
            params: Parameters used for calculation (e.g., period=14)
        """
        params_str = str(params) if params else ""
        cache_key = f"{symbol}_{indicator_name}_{params_str}"
        self._indicator_cache[cache_key] = (data, datetime.now())

    def get_technical_indicator(self, symbol, indicator_name, params=None, max_age_hours=1):
        """
        Get cached technical indicator

        Args:
            symbol: Stock ticker symbol
            indicator_name: Name of the indicator
            params: Parameters used for calculation
            max_age_hours: Maximum age of cache in hours

        Returns:
            Cached indicator data or None if not found/expired
        """
        params_str = str(params) if params else ""
        cache_key = f"{symbol}_{indicator_name}_{params_str}"

        if cache_key in self._indicator_cache:
            data, timestamp = self._indicator_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=max_age_hours):
                return data

        return None

    def _save_to_disk(self, cache_key, data):
        """Save data to disk cache"""
        try:
            cache_file = self._cache_dir / f"{self._hash_key(cache_key)}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now(),
                    'key': cache_key
                }, f)
        except Exception as e:
            print(f"Error saving cache to disk: {e}")

    def _load_from_disk(self, cache_key):
        """Load data from disk cache"""
        try:
            cache_file = self._cache_dir / f"{self._hash_key(cache_key)}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)

                # Check if cache is still valid
                if datetime.now() - cached['timestamp'] < timedelta(hours=self._max_cache_age_hours):
                    return cached['data']
        except Exception as e:
            print(f"Error loading cache from disk: {e}")

        return None

    def _hash_key(self, key):
        """Generate hash for cache key to use as filename"""
        return hashlib.md5(key.encode()).hexdigest()

    def clear_cache(self, symbol=None):
        """
        Clear cache

        Args:
            symbol: If provided, only clear cache for this symbol. Otherwise clear all.
        """
        if symbol:
            # Clear specific symbol from memory
            keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._memory_cache[key]

            keys_to_remove = [k for k in self._indicator_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._indicator_cache[key]
        else:
            # Clear all caches
            self._memory_cache.clear()
            self._indicator_cache.clear()

    def cleanup_old_cache(self):
        """Remove old cache files from disk"""
        try:
            for cache_file in self._cache_dir.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached = pickle.load(f)
                        if datetime.now() - cached['timestamp'] > timedelta(hours=self._max_cache_age_hours):
                            cache_file.unlink()
                except:
                    # Remove corrupted cache files
                    cache_file.unlink()
        except Exception as e:
            print(f"Error cleaning up cache: {e}")

    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self._memory_cache),
            'indicator_cache_size': len(self._indicator_cache),
            'disk_cache_files': len(list(self._cache_dir.glob("*.pkl")))
        }


# Global cache instance
_cache = StockDataCache()


def get_stock_data(symbol, period="1y", interval="1d", force_refresh=False):
    """
    Convenience function to get stock data with caching

    Usage:
        from cache_manager import get_stock_data
        data = get_stock_data('AAPL', period='1y')
    """
    return _cache.get_stock_data(symbol, period, interval, force_refresh)


def get_ticker_info(symbol, force_refresh=False):
    """
    Convenience function to get ticker info with caching

    Usage:
        from cache_manager import get_ticker_info
        info = get_ticker_info('AAPL')
    """
    return _cache.get_ticker_info(symbol, force_refresh)


def clear_cache(symbol=None):
    """
    Convenience function to clear cache

    Usage:
        from cache_manager import clear_cache
        clear_cache('AAPL')  # Clear specific symbol
        clear_cache()  # Clear all
    """
    _cache.clear_cache(symbol)


def get_cache_instance():
    """Get the global cache instance"""
    return _cache
