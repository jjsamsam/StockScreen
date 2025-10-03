"""
Technical indicators calculator with built-in caching
Optimizes performance by caching calculated indicators
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from cache_manager import get_cache_instance


class TechnicalIndicators:
    """기술적 지표 계산 클래스 (캐싱 지원)"""

    def __init__(self):
        self.cache = get_cache_instance()

    def calculate_sma(self, data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """
        단순 이동평균 (Simple Moving Average)

        Args:
            data: OHLCV 데이터프레임
            period: 기간
            column: 계산할 컬럼명

        Returns:
            SMA 시리즈
        """
        if data is None or data.empty or column not in data.columns:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(
            symbol, 'SMA', {'period': period, 'column': column}
        )
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        sma = data[column].rolling(window=period).mean()

        # 캐시 저장
        self.cache.cache_technical_indicator(
            symbol, 'SMA', sma, {'period': period, 'column': column}
        )

        return sma

    def calculate_ema(self, data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """
        지수 이동평균 (Exponential Moving Average)

        Args:
            data: OHLCV 데이터프레임
            period: 기간
            column: 계산할 컬럼명

        Returns:
            EMA 시리즈
        """
        if data is None or data.empty or column not in data.columns:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(
            symbol, 'EMA', {'period': period, 'column': column}
        )
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        ema = data[column].ewm(span=period, adjust=False).mean()

        # 캐시 저장
        self.cache.cache_technical_indicator(
            symbol, 'EMA', ema, {'period': period, 'column': column}
        )

        return ema

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
        """
        상대강도지수 (Relative Strength Index)

        Args:
            data: OHLCV 데이터프레임
            period: 기간
            column: 계산할 컬럼명

        Returns:
            RSI 시리즈
        """
        if data is None or data.empty or column not in data.columns:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(
            symbol, 'RSI', {'period': period, 'column': column}
        )
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # 캐시 저장
        self.cache.cache_technical_indicator(
            symbol, 'RSI', rsi, {'period': period, 'column': column}
        )

        return rsi

    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26,
                      signal: int = 9, column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)

        Args:
            data: OHLCV 데이터프레임
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 시그널 라인 기간
            column: 계산할 컬럼명

        Returns:
            (MACD, Signal, Histogram) 튜플
        """
        if data is None or data.empty or column not in data.columns:
            return pd.Series(), pd.Series(), pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        params = {'fast': fast, 'slow': slow, 'signal': signal, 'column': column}
        cached = self.cache.get_technical_indicator(symbol, 'MACD', params)

        if cached is not None and isinstance(cached, tuple) and len(cached) == 3:
            if len(cached[0]) == len(data):
                return cached

        # 계산
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        result = (macd_line, signal_line, histogram)

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'MACD', result, params)

        return result

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20,
                                  std_dev: float = 2.0, column: str = 'Close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        볼린저 밴드 (Bollinger Bands)

        Args:
            data: OHLCV 데이터프레임
            period: 기간
            std_dev: 표준편차 배수
            column: 계산할 컬럼명

        Returns:
            (Upper Band, Middle Band, Lower Band) 튜플
        """
        if data is None or data.empty or column not in data.columns:
            return pd.Series(), pd.Series(), pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        params = {'period': period, 'std_dev': std_dev, 'column': column}
        cached = self.cache.get_technical_indicator(symbol, 'BB', params)

        if cached is not None and isinstance(cached, tuple) and len(cached) == 3:
            if len(cached[0]) == len(data):
                return cached

        # 계산
        middle_band = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        result = (upper_band, middle_band, lower_band)

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'BB', result, params)

        return result

    def calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14,
                            d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        스토캐스틱 오실레이터 (Stochastic Oscillator)

        Args:
            data: OHLCV 데이터프레임
            k_period: %K 기간
            d_period: %D 기간

        Returns:
            (%K, %D) 튜플
        """
        if data is None or data.empty:
            return pd.Series(), pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        params = {'k_period': k_period, 'd_period': d_period}
        cached = self.cache.get_technical_indicator(symbol, 'STOCH', params)

        if cached is not None and isinstance(cached, tuple) and len(cached) == 2:
            if len(cached[0]) == len(data):
                return cached

        # 계산
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()

        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        result = (k_percent, d_percent)

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'STOCH', result, params)

        return result

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        평균 진폭 범위 (Average True Range)

        Args:
            data: OHLCV 데이터프레임
            period: 기간

        Returns:
            ATR 시리즈
        """
        if data is None or data.empty:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(symbol, 'ATR', {'period': period})
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'ATR', atr, {'period': period})

        return atr

    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """
        거래량 균형 지표 (On-Balance Volume)

        Args:
            data: OHLCV 데이터프레임

        Returns:
            OBV 시리즈
        """
        if data is None or data.empty:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(symbol, 'OBV', {})
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'OBV', obv, {})

        return obv

    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        평균 방향성 지수 (Average Directional Index)

        Args:
            data: OHLCV 데이터프레임
            period: 기간

        Returns:
            ADX 시리즈
        """
        if data is None or data.empty:
            return pd.Series()

        # 캐시 확인
        symbol = self._get_symbol_from_data(data)
        cached = self.cache.get_technical_indicator(symbol, 'ADX', {'period': period})
        if cached is not None and len(cached) == len(data):
            return cached

        # 계산
        high_diff = data['High'].diff()
        low_diff = data['Low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = self.calculate_atr(data, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        # 캐시 저장
        self.cache.cache_technical_indicator(symbol, 'ADX', adx, {'period': period})

        return adx

    def _get_symbol_from_data(self, data: pd.DataFrame) -> str:
        """데이터프레임에서 심볼 추출 (없으면 해시 사용)"""
        if hasattr(data, 'symbol'):
            return data.symbol

        # 데이터프레임의 해시를 심볼로 사용
        return f"data_{hash(str(data.index[0]) + str(data.index[-1]))}"


# 전역 인스턴스
_indicators = TechnicalIndicators()


def get_indicators() -> TechnicalIndicators:
    """기술적 지표 계산기 인스턴스 가져오기"""
    return _indicators
