"""
stock_analysis_service.py
주식 기술적 분석 서비스 - update_info_panel 로직의 웹앱 버전

이 서비스는 chart_window.py의 update_info_panel 로직을 
웹앱에서 사용 가능하도록 분리한 것입니다.
"""

import sys
import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# 프로젝트 루트 추가 (core -> backend -> web_app -> multiple)
current_dir = os.path.dirname(os.path.abspath(__file__))  # core
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

# 경로 추가 (중복 방지)
for path in [project_root, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from cache_manager import get_stock_data
from logger_config import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """기술적 지표 계산 유틸리티"""
    
    @staticmethod
    def calculate_all(data: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 계산"""
        df = data.copy()
        
        # 이동평균선
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        df['MA120'] = df['Close'].rolling(window=120).mean()
        
        # 볼린저 밴드 (20일 기준)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # RSI (14일 기준)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # ADX (14일 기준) - 간략화된 버전
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        df['+DM'] = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        df['-DM'] = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR'])
        df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR'])
        
        dx = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
        df['ADX'] = dx.rolling(window=14).mean()
        
        return df


class StockAnalysisService:
    """주식 기술적 분석 서비스"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StockAnalysisService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        logger.info("StockAnalysisService 초기화 완료")
    
    def get_stock_info(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        종목의 기술적 분석 정보를 반환
        
        Args:
            symbol: 종목 코드
            period: 기간 (1mo, 3mo, 6mo, 1y, 2y)
        
        Returns:
            기술적 분석 정보 딕셔너리
        """
        try:
            # 데이터 로드
            data = get_stock_data(symbol, period=period, validate_cache=False)
            
            if data is None or len(data) < 20:
                return {
                    'success': False,
                    'error': f"'{symbol}' 데이터를 불러올 수 없습니다"
                }
            
            # 시간대 처리
            if data.index.tz is not None:
                # UTC로 변환하지 않고 바로 시간대 정보만 제거하여 해당 시장의 로컬 날짜 유지
                # 예: 2026-02-02 00:00 KST -> 2026-02-02 00:00 (UTC 변환 시 2026-02-01 15:00이 되어 날짜가 하루 밀림 방지)
                data.index = data.index.tz_localize(None)
            
            # 무효 데이터 필터링
            # 무효 데이터 필터링
            # Close가 0이거나 NaN인 경우만 제거 (Open/High/Low는 0이어도 Close가 있으면 유효한 것으로 간주)
            invalid_mask = (data['Close'] == 0) | (data['Close'].isna())
            
            if invalid_mask.any():
                data = data[~invalid_mask].copy()
            
            if data.empty or len(data) < 2:
                return {
                    'success': False,
                    'error': "유효한 가격 데이터가 없습니다"
                }
            
            # 기술적 지표 계산
            data = TechnicalIndicators.calculate_all(data)
            
            # 분석 결과 생성
            analysis = self._analyze_stock(data, symbol)
            
            return {
                'success': True,
                'data': analysis
            }
            
        except Exception as e:
            logger.error(f"주식 분석 오류 ({symbol}): {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_stock(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """기술적 분석 수행"""
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        # === 가격 정보 ===
        try:
            price_change = float(current['Close']) - float(prev['Close'])
            price_change_pct = (price_change / float(prev['Close'])) * 100 if prev['Close'] else 0.0
        except:
            price_change, price_change_pct = 0.0, 0.0
        
        price_info = {
            'current_price': float(current['Close']),
            'prev_close': float(prev['Close']),
            'change': float(price_change),
            'change_percent': float(price_change_pct),
            'high': float(current['High']),
            'low': float(current['Low']),
            'open': float(current['Open']),
        }
        
        # === RSI 분석 ===
        rsi = float(current.get('RSI', 50.0))
        if rsi >= 80:
            rsi_signal = "extreme_overbought"
            rsi_desc = "극도 과매수 (즉시 매도 고려)"
        elif rsi >= 70:
            rsi_signal = "overbought"
            rsi_desc = "과매수 (매도 준비)"
        elif rsi >= 60:
            rsi_signal = "bullish"
            rsi_desc = "강세 구간 (상승 지속 가능)"
        elif rsi >= 40:
            rsi_signal = "neutral"
            rsi_desc = "중립 구간 (방향성 애매)"
        elif rsi >= 30:
            rsi_signal = "bearish"
            rsi_desc = "약세 구간 (하락 지속 가능)"
        elif rsi >= 20:
            rsi_signal = "oversold"
            rsi_desc = "과매도 (매수 준비)"
        else:
            rsi_signal = "extreme_oversold"
            rsi_desc = "극도 과매도 (적극 매수 고려)"
        
        rsi_info = {
            'value': rsi,
            'signal': rsi_signal,
            'description': rsi_desc
        }
        
        # === MACD 분석 ===
        macd_now = float(current.get('MACD', 0.0))
        macd_sig_now = float(current.get('MACD_Signal', 0.0))
        macd_prev = float(prev.get('MACD', 0.0))
        macd_sig_prev = float(prev.get('MACD_Signal', 0.0))
        
        macd_cross_up = (macd_now > macd_sig_now) and (macd_prev <= macd_sig_prev)
        macd_cross_down = (macd_now < macd_sig_now) and (macd_prev >= macd_sig_prev)
        
        if macd_cross_up:
            macd_signal = "golden_cross"
            macd_desc = "골든크로스 발생 (강력한 매수 신호)"
        elif macd_cross_down:
            macd_signal = "death_cross"
            macd_desc = "데드크로스 발생 (강력한 매도 신호)"
        elif macd_now > macd_sig_now:
            macd_signal = "bullish"
            macd_desc = "MACD > Signal (상승 모멘텀)"
        else:
            macd_signal = "bearish"
            macd_desc = "MACD < Signal (하락 모멘텀)"
        
        macd_info = {
            'macd': macd_now,
            'signal_line': macd_sig_now,
            'histogram': float(current.get('MACD_Histogram', 0.0)),
            'signal': macd_signal,
            'description': macd_desc
        }
        
        # === 볼린저밴드 분석 ===
        try:
            band_range = float(current['BB_Upper']) - float(current['BB_Lower'])
            bb_position = (float(current['Close']) - float(current['BB_Lower'])) / band_range if band_range != 0 else 0.5
        except:
            bb_position = 0.5
        
        if bb_position > 0.8:
            bb_signal = "upper"
            bb_desc = "상단 근접 (매도 관심)"
        elif bb_position < 0.2:
            bb_signal = "lower"
            bb_desc = "하단 근접 (매수 관심)"
        else:
            bb_signal = "middle"
            bb_desc = "중앙 영역 (관망)"
        
        bollinger_info = {
            'upper': float(current.get('BB_Upper', 0)),
            'middle': float(current.get('BB_Middle', 0)),
            'lower': float(current.get('BB_Lower', 0)),
            'position': bb_position,
            'signal': bb_signal,
            'description': bb_desc
        }
        
        # === 이동평균선 분석 ===
        ma20 = float(current.get('MA20', 0))
        ma60 = float(current.get('MA60', 0))
        ma120 = float(current.get('MA120', 0))
        
        if ma20 > ma60 > ma120 and ma120 > 0:
            ma_signal = "strong_bullish"
            ma_desc = "완전 정배열 (강한 상승 추세)"
            trend_strength = "매우 강함"
        elif ma20 > ma60:
            ma_signal = "bullish"
            ma_desc = "부분 정배열 (단기 상승 추세)"
            trend_strength = "보통"
        elif ma20 < ma60 < ma120 and ma120 > 0:
            ma_signal = "strong_bearish"
            ma_desc = "완전 역배열 (강한 하락 추세)"
            trend_strength = "매우 약함"
        elif ma20 < ma60:
            ma_signal = "bearish"
            ma_desc = "부분 역배열 (단기 하락 추세)"
            trend_strength = "약함"
        else:
            ma_signal = "neutral"
            ma_desc = "혼재 (방향성 불분명)"
            trend_strength = "중립"
        
        ma_info = {
            'ma20': ma20,
            'ma60': ma60,
            'ma120': ma120,
            'signal': ma_signal,
            'description': ma_desc,
            'trend_strength': trend_strength
        }
        
        # === 거래량 분석 ===
        vol_now = float(current.get('Volume', 0))
        vol_ma20 = float(data['Volume'].rolling(20, min_periods=1).mean().iloc[-1])
        vol_ratio = (vol_now / vol_ma20) if vol_ma20 > 0 else 1.0
        
        if vol_ratio > 3.0:
            vol_signal = "extreme_high"
            vol_desc = "대량 거래 (주목 필요)"
        elif vol_ratio > 2.0:
            vol_signal = "high"
            vol_desc = "높은 거래량 (관심 증가)"
        elif vol_ratio > 1.5:
            vol_signal = "above_average"
            vol_desc = "보통 이상 거래량"
        elif vol_ratio > 0.8:
            vol_signal = "normal"
            vol_desc = "보통 거래량"
        else:
            vol_signal = "low"
            vol_desc = "낮은 거래량 (관심 부족)"
        
        volume_info = {
            'current': vol_now,
            'average_20d': vol_ma20,
            'ratio': vol_ratio,
            'signal': vol_signal,
            'description': vol_desc
        }
        
        # === ADX (추세 강도) ===
        adx_value = float(current.get('ADX', 0))
        plus_di = float(current.get('+DI', 0))
        minus_di = float(current.get('-DI', 0))
        atr_value = float(current.get('ATR', 0))
        
        if adx_value > 25:
            adx_signal = "strong"
            adx_desc = "강한 추세"
        else:
            adx_signal = "weak"
            adx_desc = "약한 추세 (횡보)"
        
        trend_info = {
            'adx': adx_value,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'atr': atr_value,
            'signal': adx_signal,
            'description': adx_desc,
            'direction': 'bullish' if plus_di > minus_di else 'bearish'
        }
        
        # === 종합 점수 ===
        bullish_points = 0
        bearish_points = 0
        
        if macd_signal in ['golden_cross', 'bullish']: bullish_points += 1
        if rsi < 30: bullish_points += 1
        if bb_position < 0.2: bullish_points += 1
        if ma_signal == 'strong_bullish': bullish_points += 2
        elif ma_signal == 'bullish': bullish_points += 1
        if vol_ratio > 1.5: bullish_points += 1
        
        if macd_signal in ['death_cross', 'bearish']: bearish_points += 1
        if rsi > 70: bearish_points += 1
        if bb_position > 0.8: bearish_points += 1
        if ma_signal == 'strong_bearish': bearish_points += 2
        elif ma_signal == 'bearish': bearish_points += 1
        
        if bullish_points >= 4:
            overall_signal = "strong_buy"
            overall_desc = "강력 매수 추천"
        elif bullish_points >= 2 and bullish_points > bearish_points:
            overall_signal = "buy"
            overall_desc = "매수 관심 구간"
        elif bearish_points >= 4:
            overall_signal = "strong_sell"
            overall_desc = "강력 매도 추천"
        elif bearish_points >= 2 and bearish_points > bullish_points:
            overall_signal = "sell"
            overall_desc = "매도 관심 구간"
        else:
            overall_signal = "neutral"
            overall_desc = "중립/관망 구간"
        
        summary = {
            'bullish_points': bullish_points,
            'bearish_points': bearish_points,
            'signal': overall_signal,
            'description': overall_desc
        }
        
        # === 리스크 관리 ===
        if atr_value > 0:
            stop_loss = float(current['Close']) - (atr_value * 2)
            take_profit = float(current['Close']) + (atr_value * 3)
            risk_reward = 1.5
        else:
            stop_loss = float(current['Close']) * 0.95
            take_profit = float(current['Close']) * 1.10
            risk_reward = 2.0
        
        risk_management = {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward
        }
        
        # 최종 결과
        return {
            'symbol': symbol,
            'last_update': data.index[-1].strftime('%Y-%m-%d'),
            'price': price_info,
            'rsi': rsi_info,
            'macd': macd_info,
            'bollinger': bollinger_info,
            'moving_averages': ma_info,
            'volume': volume_info,
            'trend': trend_info,
            'summary': summary,
            'risk_management': risk_management
        }


# 전역 인스턴스
stock_analysis_service = StockAnalysisService()
