"""
technical_analysis.py
Technical Analysis Indicators Calculation
Separated from utils.py to allow usage in headless environments (e.g. Docker backend)
without PyQt5 dependencies.
"""

import pandas as pd
import numpy as np

class TechnicalAnalysis:
    """기술적 분석 클래스"""

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 계산 (결측 보정 포함)"""
        if data is None or data.empty:
            return data
            
        data = data.copy()
        
        # 이동평균선
        data['MA5'] = data['Close'].rolling(5).mean()
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA60'] = data['Close'].rolling(60).mean()
        data['MA120'] = data['Close'].rolling(120).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data['RSI'] = 100 - (100 / (1 + rs))

        # 볼린저밴드
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

        # 스토캐스틱
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        denom = (high_14 - low_14).replace(0, np.nan)
        data['%K'] = 100 * ((data['Close'] - low_14) / denom)
        data['%D'] = data['%K'].rolling(3).mean()

        # 윌리엄스 %R
        data['Williams_R'] = -100 * ((high_14 - data['Close']) / denom)

        # 거래량 지표
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['OBV'] = (data['Volume'] * np.where(data['Close'] > data['Close'].shift(1), 1, -1)).cumsum()

        # CCI
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))

        # ATR (Average True Range) - 변동성 측정
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()

        # ADX (Average Directional Index) - 추세 강도 측정
        # +DM, -DM 계산
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()

        plus_dm = high_diff.copy()
        plus_dm[(high_diff < 0) | (high_diff <= low_diff)] = 0

        minus_dm = low_diff.copy()
        minus_dm[(low_diff < 0) | (low_diff <= high_diff)] = 0

        # +DI, -DI 계산
        plus_di = 100 * (plus_dm.rolling(14).mean() / data['ATR'])
        minus_di = 100 * (minus_dm.rolling(14).mean() / data['ATR'])

        # DX, ADX 계산
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        data['ADX'] = dx.rolling(14).mean()
        data['+DI'] = plus_di
        data['-DI'] = minus_di

        # Parabolic SAR - 추세 추적 (간단한 버전)
        # 실제 구현은 복잡하므로 기본 로직만
        data['PSAR'] = data['Close'].copy()  # 초기값

        # 결측값 처리
        try:
            data = data.ffill().bfill()
        except Exception:
            data = data.fillna(method='ffill').fillna(method='bfill')

        return data
