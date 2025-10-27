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
import pytz
from logger_config import get_logger

logger = get_logger(__name__)


class StockDataCache:
    """Singleton cache manager for stock data"""

    _instance = None
    _cache = {}
    _cache_dir = Path("cache")
    _max_cache_age_hours = 24
    _max_memory_cache_size = 500  # 최대 메모리 캐시 항목 수 (메모리 누수 방지)
    _max_indicator_cache_size = 1000  # 최대 지표 캐시 항목 수

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

    def get_stock_data(self, symbol, period="1y", interval="1d", force_refresh=False, validate_cache=True):
        """
        Get stock data with caching

        Args:
            symbol: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            force_refresh: Force refresh from API
            validate_cache: Validate cached data for anomalies (default: True)

        Returns:
            DataFrame with stock data
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Check memory cache first
        if not force_refresh and cache_key in self._memory_cache:
            cached_data, timestamp = self._memory_cache[cache_key]
            if datetime.now() - timestamp < timedelta(hours=self._max_cache_age_hours):
                # Validate cached data if requested
                if validate_cache and not self._validate_data_quality(cached_data, symbol):
                    logger.warning(f"Cached data for {symbol} failed validation, forcing refresh")
                    force_refresh = True
                else:
                    return cached_data

        # Check disk cache
        if not force_refresh:
            disk_data = self._load_from_disk(cache_key)
            if disk_data is not None:
                # Validate disk data if requested
                if validate_cache and not self._validate_data_quality(disk_data, symbol):
                    logger.warning(f"Disk cached data for {symbol} failed validation, forcing refresh")
                    force_refresh = True
                else:
                    self._memory_cache[cache_key] = (disk_data, datetime.now())
                    return disk_data

        # Fetch fresh data from API
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data is not None and not data.empty:
                # Remove today's incomplete data if market hasn't closed yet
                data = self._remove_incomplete_today_data(data, symbol)

                # Validate fresh data
                if validate_cache and not self._validate_data_quality(data, symbol):
                    logger.error(f"Fresh data for {symbol} failed validation")
                    return None

                # Cache in memory and disk
                self._memory_cache[cache_key] = (data, datetime.now())
                self._save_to_disk(cache_key, data)
                self._cleanup_memory_cache()  # 메모리 캐시 크기 제한
                return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")

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
                self._cleanup_memory_cache()  # 메모리 캐시 크기 제한
                return info
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")

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
        self._cleanup_indicator_cache()  # 지표 캐시 크기 제한

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
            logger.error(f"Error saving cache to disk: {e}")

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
            logger.error(f"Error loading cache from disk: {e}")

        return None

    def _hash_key(self, key):
        """Generate hash for cache key to use as filename"""
        return hashlib.md5(key.encode()).hexdigest()

    def _remove_incomplete_today_data(self, data, symbol):
        """
        Remove today's data if the market hasn't closed yet to avoid incomplete/incorrect data

        Args:
            data: DataFrame with stock data
            symbol: Stock ticker symbol

        Returns:
            DataFrame with incomplete today's data removed if necessary
        """
        if data is None or data.empty:
            return data

        try:
            # Get market info from symbol suffix
            market_info = self._get_market_info(symbol)
            timezone = market_info['timezone']
            close_hour = market_info['close_hour']
            close_minute = market_info['close_minute']

            # Get current time in market's timezone
            tz = pytz.timezone(timezone)
            now_market_time = datetime.now(tz)

            # Get today's market close time
            today_close = now_market_time.replace(
                hour=close_hour,
                minute=close_minute,
                second=0,
                microsecond=0
            )

            # Check if market has closed today
            market_closed = now_market_time >= today_close

            # Get the last data point's date
            if len(data) > 0:
                last_date = data.index[-1]

                # Convert to market timezone for comparison
                if last_date.tzinfo is None:
                    last_date = tz.localize(last_date)
                else:
                    last_date = last_date.astimezone(tz)

                # Check if last data point is from today
                is_today = last_date.date() == now_market_time.date()

                if is_today and not market_closed:
                    # Market hasn't closed yet, remove today's incomplete data
                    logger.warning(f"🕐 Market not closed yet for {symbol}. Removing incomplete today's data.")
                    logger.info(f"   Current time: {now_market_time.strftime('%Y-%m-%d %H:%M %Z')}")
                    logger.info(f"   Market closes: {today_close.strftime('%H:%M %Z')}")
                    logger.info(f"   Last data date: {last_date.strftime('%Y-%m-%d %H:%M')}")

                    # Log the data being removed for debugging
                    today_data = data.iloc[-1]
                    logger.debug(f"   Removing: O={today_data['Open']:.2f} H={today_data['High']:.2f} "
                               f"L={today_data['Low']:.2f} C={today_data['Close']:.2f}")

                    # Remove the last row (today's incomplete data)
                    data = data.iloc[:-1]

                    if len(data) > 0:
                        logger.info(f"   ✅ Using data up to: {data.index[-1].strftime('%Y-%m-%d')}")
                    else:
                        logger.warning(f"   ⚠️ No data remaining after removing today's data")
                elif is_today and market_closed:
                    logger.debug(f"✅ Market closed for {symbol}. Today's data is complete.")
                else:
                    logger.debug(f"✅ Last data is from {last_date.strftime('%Y-%m-%d')}, not today. Data is complete.")

            return data

        except Exception as e:
            logger.warning(f"Error checking market close time for {symbol}: {e}. Using data as-is.")
            return data

    def _get_market_info(self, symbol):
        """
        Get market timezone and closing time based on symbol suffix

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with timezone and close time
        """
        # Default to US market
        market_info = {
            'timezone': 'America/New_York',
            'close_hour': 16,
            'close_minute': 0
        }

        # Korean markets (.KS, .KQ)
        if symbol.endswith('.KS') or symbol.endswith('.KQ'):
            market_info = {
                'timezone': 'Asia/Seoul',
                'close_hour': 15,
                'close_minute': 30
            }
        # Japanese market (.T)
        elif symbol.endswith('.T'):
            market_info = {
                'timezone': 'Asia/Tokyo',
                'close_hour': 15,
                'close_minute': 0
            }
        # Hong Kong market (.HK)
        elif symbol.endswith('.HK'):
            market_info = {
                'timezone': 'Asia/Hong_Kong',
                'close_hour': 16,
                'close_minute': 0
            }
        # London market (.L)
        elif symbol.endswith('.L'):
            market_info = {
                'timezone': 'Europe/London',
                'close_hour': 16,
                'close_minute': 30
            }
        # German market (.DE)
        elif symbol.endswith('.DE'):
            market_info = {
                'timezone': 'Europe/Berlin',
                'close_hour': 17,
                'close_minute': 30
            }
        # Shanghai market (.SS)
        elif symbol.endswith('.SS'):
            market_info = {
                'timezone': 'Asia/Shanghai',
                'close_hour': 15,
                'close_minute': 0
            }
        # Shenzhen market (.SZ)
        elif symbol.endswith('.SZ'):
            market_info = {
                'timezone': 'Asia/Shanghai',
                'close_hour': 15,
                'close_minute': 0
            }
        # US markets (default - NASDAQ, NYSE, etc.)
        # No suffix or common US suffixes

        return market_info

    def _validate_data_quality(self, data, symbol):
        """
        Validate data quality to detect anomalies, splits, or data corruption

        Args:
            data: DataFrame with stock data
            symbol: Stock ticker symbol

        Returns:
            True if data passes validation, False otherwise
        """
        if data is None or data.empty:
            logger.warning(f"Data validation failed for {symbol}: Empty data")
            return False

        try:
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                logger.warning(f"Data validation failed for {symbol}: Missing required columns")
                return False

            # Check for NaN values in critical columns
            if data[['Close', 'Open', 'High', 'Low']].isnull().any().any():
                logger.warning(f"Data validation failed for {symbol}: NaN values in price data")
                return False

            # Check for zero or negative prices
            if (data['Close'] <= 0).any() or (data['Open'] <= 0).any():
                logger.warning(f"Data validation failed for {symbol}: Zero or negative prices")
                return False

            # Check for suspicious price jumps (potential stock splits not adjusted)
            # A single-day change > 50% up or > 40% down without high volume might indicate data issue
            price_changes = data['Close'].pct_change().abs()
            suspicious_jumps = price_changes > 0.5

            if suspicious_jumps.sum() > 0:
                # Check if these jumps coincide with volume spikes
                for idx in data[suspicious_jumps].index:
                    if idx in data.index:
                        pos = data.index.get_loc(idx)
                        if pos > 0:
                            prev_volume = data['Volume'].iloc[pos - 1] if pos > 0 else data['Volume'].iloc[pos]
                            curr_volume = data['Volume'].iloc[pos]
                            # If big price jump without volume spike, likely data issue
                            if curr_volume < prev_volume * 1.5:
                                logger.warning(f"Data validation warning for {symbol}: Suspicious price jump at {idx} "
                                             f"({price_changes.loc[idx]*100:.1f}%) without volume spike")
                                # Note: We log warning but don't fail validation as this could be legitimate

            # Check for High < Low (impossible)
            if (data['High'] < data['Low']).any():
                logger.warning(f"Data validation failed for {symbol}: High < Low detected")
                return False

            # Check for Close outside High-Low range (with small tolerance for rounding errors)
            # Allow 0.1% tolerance for floating point precision issues
            tolerance = 0.001
            close_too_high = data['Close'] > data['High'] * (1 + tolerance)
            close_too_low = data['Close'] < data['Low'] * (1 - tolerance)

            if (close_too_high | close_too_low).any():
                # Log which rows have issues
                problem_rows = data[close_too_high | close_too_low]
                logger.warning(f"Data validation issue for {symbol}: Close outside High-Low range")
                logger.warning(f"Problem dates: {problem_rows.index.tolist()}")

                # Try to fix small discrepancies (< 1%)
                for idx in problem_rows.index:
                    close_val = data.loc[idx, 'Close']
                    high_val = data.loc[idx, 'High']
                    low_val = data.loc[idx, 'Low']

                    deviation_high = (close_val - high_val) / high_val if high_val > 0 else 0
                    deviation_low = (low_val - close_val) / low_val if low_val > 0 else 0

                    # If deviation is small (< 1%), adjust High/Low instead of failing
                    if abs(deviation_high) < 0.01 or abs(deviation_low) < 0.01:
                        if close_val > high_val:
                            data.loc[idx, 'High'] = close_val
                            logger.info(f"Fixed {symbol} on {idx}: Adjusted High from {high_val} to {close_val}")
                        if close_val < low_val:
                            data.loc[idx, 'Low'] = close_val
                            logger.info(f"Fixed {symbol} on {idx}: Adjusted Low from {low_val} to {close_val}")
                    else:
                        # Large deviation - this is a real data problem
                        logger.error(f"Cannot fix {symbol} on {idx}: Close={close_val}, High={high_val}, Low={low_val}")
                        return False

                # Re-check after fixes
                close_too_high = data['Close'] > data['High'] * (1 + tolerance)
                close_too_low = data['Close'] < data['Low'] * (1 - tolerance)
                if (close_too_high | close_too_low).any():
                    logger.error(f"Data validation failed for {symbol}: Close outside High-Low range even after fixes")
                    return False
                else:
                    logger.info(f"Successfully fixed data validation issues for {symbol}")

            # Check for minimum data points
            if len(data) < 5:
                logger.warning(f"Data validation failed for {symbol}: Insufficient data points ({len(data)})")
                return False

            logger.debug(f"Data validation passed for {symbol}: {len(data)} data points")
            return True

        except Exception as e:
            logger.error(f"Data validation error for {symbol}: {e}")
            return False

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

    def _cleanup_memory_cache(self):
        """메모리 캐시가 너무 커지면 오래된 항목 제거 (LRU 방식)"""
        if len(self._memory_cache) > self._max_memory_cache_size:
            # 타임스탬프 기준으로 정렬하여 오래된 항목 제거
            sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
            items_to_remove = len(self._memory_cache) - self._max_memory_cache_size

            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self._memory_cache[key]

            logger.debug(f"메모리 캐시 정리: {items_to_remove}개 항목 제거 (현재: {len(self._memory_cache)}개)")

    def _cleanup_indicator_cache(self):
        """지표 캐시가 너무 커지면 오래된 항목 제거"""
        if len(self._indicator_cache) > self._max_indicator_cache_size:
            # 타임스탬프 기준으로 정렬하여 오래된 항목 제거
            sorted_items = sorted(self._indicator_cache.items(), key=lambda x: x[1][1])
            items_to_remove = len(self._indicator_cache) - self._max_indicator_cache_size

            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self._indicator_cache[key]

            logger.debug(f"지표 캐시 정리: {items_to_remove}개 항목 제거 (현재: {len(self._indicator_cache)}개)")

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
            logger.error(f"Error cleaning up cache: {e}")

    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'memory_cache_size': len(self._memory_cache),
            'indicator_cache_size': len(self._indicator_cache),
            'disk_cache_files': len(list(self._cache_dir.glob("*.pkl"))),
            'memory_cache_limit': self._max_memory_cache_size,
            'indicator_cache_limit': self._max_indicator_cache_size
        }

    def force_cleanup(self):
        """
        강제 메모리 정리 (대량 스크리닝 후 호출 권장)

        메모리 캐시와 지표 캐시를 공격적으로 정리하여
        메모리 사용량을 줄입니다.
        """
        # 메모리 캐시를 절반으로 줄임
        if len(self._memory_cache) > 100:
            sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
            keep_count = max(100, len(self._memory_cache) // 2)

            # 오래된 항목 제거
            for i in range(len(self._memory_cache) - keep_count):
                key = sorted_items[i][0]
                del self._memory_cache[key]

            logger.info(f"🧹 강제 메모리 정리: 메모리 캐시 {len(self._memory_cache)}개로 축소")

        # 지표 캐시를 절반으로 줄임
        if len(self._indicator_cache) > 100:
            sorted_items = sorted(self._indicator_cache.items(), key=lambda x: x[1][1])
            keep_count = max(100, len(self._indicator_cache) // 2)

            for i in range(len(self._indicator_cache) - keep_count):
                key = sorted_items[i][0]
                del self._indicator_cache[key]

            logger.info(f"🧹 강제 메모리 정리: 지표 캐시 {len(self._indicator_cache)}개로 축소")

        # Python 가비지 컬렉터 실행
        import gc
        gc.collect()
        logger.info("🧹 메모리 강제 정리 완료")


# Global cache instance
_cache = StockDataCache()


def get_stock_data(symbol, period="1y", interval="1d", force_refresh=False, validate_cache=True):
    """
    Convenience function to get stock data with caching

    Usage:
        from cache_manager import get_stock_data
        data = get_stock_data('AAPL', period='1y')
        # For model training, use force_refresh and validation
        data = get_stock_data('AAPL', period='1y', force_refresh=True, validate_cache=True)
    """
    return _cache.get_stock_data(symbol, period, interval, force_refresh, validate_cache)


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


def force_cleanup_cache():
    """
    강제 캐시 정리 (대량 스크리닝 후 호출 권장)

    Usage:
        from cache_manager import force_cleanup_cache
        # 2000개 종목 스크리닝 후
        force_cleanup_cache()
    """
    _cache.force_cleanup()


def get_cache_stats():
    """
    캐시 통계 조회

    Usage:
        from cache_manager import get_cache_stats
        stats = get_cache_stats()
        print(f"메모리 캐시: {stats['memory_cache_size']}/{stats['memory_cache_limit']}")
    """
    return _cache.get_cache_stats()
