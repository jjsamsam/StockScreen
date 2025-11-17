"""
Smart Signal Generator
가중치 기반 스마트 매매 신호 생성 시스템
"""

import pandas as pd
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)


class SmartSignalGenerator:
    """
    지표 중요도를 고려한 스마트 신호 생성

    시장 환경(추세장 vs 횡보장)에 따라 지표별 가중치를 다르게 적용하여
    더 정확한 매매 신호를 생성합니다.
    """

    # 시장 환경별 지표 가중치
    WEIGHTS = {
        'trend': {  # 추세장 (ADX > 25)
            'adx': 3.0,          # 추세 강도가 가장 중요
            'ma_alignment': 2.5,  # 이동평균 정렬
            'macd': 2.0,         # MACD 신호
            '+di_-di': 1.8,      # +DI/-DI 관계
            'rsi': 1.5,          # RSI
            'volume': 1.2,       # 거래량
            'bb': 1.0,           # 볼린저 밴드
            'stochastic': 0.8    # 스토캐스틱
        },
        'range': {  # 횡보장 (ADX < 25)
            'rsi': 3.0,          # 과매수/과매도가 가장 중요
            'stochastic': 2.5,   # 스토캐스틱도 중요
            'bb': 2.0,           # 볼린저 밴드
            'volume': 1.5,       # 거래량
            'macd': 1.2,         # MACD
            '+di_-di': 1.0,      # +DI/-DI
            'ma_alignment': 0.5, # 이동평균은 덜 중요
            'adx': 0.3           # ADX는 최소
        }
    }

    def __init__(self):
        """초기화"""
        pass

    def detect_market_regime(self, indicators):
        """
        시장 환경 감지: 추세장 vs 횡보장

        Args:
            indicators: 기술적 지표 딕셔너리

        Returns:
            str: 'trend' (추세장) 또는 'range' (횡보장)
        """
        adx = indicators.get('adx', 20)

        if adx > 25:
            return 'trend'  # 추세장
        else:
            return 'range'  # 횡보장

    def analyze_indicators(self, data):
        """
        현재 기술적 지표 상태 분석

        Args:
            data: 기술적 지표가 계산된 DataFrame

        Returns:
            dict: 분석된 지표 상태
        """
        if len(data) < 2:
            return None

        current = data.iloc[-1]
        prev = data.iloc[-2]

        indicators = {}

        # ADX - 추세 강도
        indicators['adx'] = float(current.get('ADX', 20))
        indicators['+di'] = float(current.get('+DI', 20))
        indicators['-di'] = float(current.get('-DI', 20))

        # ATR - 변동성
        indicators['atr'] = float(current.get('ATR', 0))

        # RSI
        indicators['rsi'] = float(current.get('RSI', 50))

        # MACD
        indicators['macd'] = float(current.get('MACD', 0))
        indicators['macd_signal'] = float(current.get('MACD_Signal', 0))
        indicators['macd_hist'] = float(current.get('MACD_Histogram', 0))

        # MACD 크로스 감지
        macd_prev = float(prev.get('MACD', 0))
        macd_sig_prev = float(prev.get('MACD_Signal', 0))
        indicators['macd_cross_up'] = (indicators['macd'] > indicators['macd_signal']) and (macd_prev <= macd_sig_prev)
        indicators['macd_cross_down'] = (indicators['macd'] < indicators['macd_signal']) and (macd_prev >= macd_sig_prev)

        # 이동평균
        indicators['ma20'] = float(current.get('MA20', 0))
        indicators['ma60'] = float(current.get('MA60', 0))
        indicators['ma120'] = float(current.get('MA120', 0))
        indicators['close'] = float(current.get('Close', 0))

        # 볼린저 밴드
        bb_upper = float(current.get('BB_Upper', 0))
        bb_lower = float(current.get('BB_Lower', 0))
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            indicators['bb_position'] = (indicators['close'] - bb_lower) / bb_range
        else:
            indicators['bb_position'] = 0.5

        # 스토캐스틱
        indicators['stoch_k'] = float(current.get('%K', 50))
        indicators['stoch_d'] = float(current.get('%D', 50))

        # 거래량
        indicators['volume_ratio'] = float(current.get('Volume_Ratio', 1.0))

        return indicators

    def calculate_signal_scores(self, indicators, regime):
        """
        지표별 매수/매도 점수 계산

        Args:
            indicators: 분석된 지표 딕셔너리
            regime: 시장 환경 ('trend' 또는 'range')

        Returns:
            tuple: (bullish_score, bearish_score, details)
        """
        weights = self.WEIGHTS[regime]
        bullish_score = 0
        bearish_score = 0
        details = []

        # 1. ADX - 추세 강도
        if indicators['adx'] > 25:
            # 추세가 강함 - +DI/-DI 관계 확인
            if indicators['+di'] > indicators['-di']:
                bullish_score += weights['adx']
                details.append(f"✅ ADX({indicators['adx']:.1f}) 강한 상승추세 (+{weights['adx']:.1f}점)")
            else:
                bearish_score += weights['adx']
                details.append(f"❌ ADX({indicators['adx']:.1f}) 강한 하락추세 (-{weights['adx']:.1f}점)")
        else:
            details.append(f"⚪ ADX({indicators['adx']:.1f}) 횡보장 (추세 약함)")

        # 2. +DI / -DI 관계
        if indicators['+di'] > indicators['-di']:
            score = weights['+di_-di'] * (indicators['+di'] / indicators['-di'] if indicators['-di'] > 0 else 1)
            score = min(score, weights['+di_-di'] * 2)  # 최대 2배
            bullish_score += score
            details.append(f"✅ +DI({indicators['+di']:.1f}) > -DI({indicators['-di']:.1f}) 매수 우세 (+{score:.1f}점)")
        else:
            score = weights['+di_-di'] * (indicators['-di'] / indicators['+di'] if indicators['+di'] > 0 else 1)
            score = min(score, weights['+di_-di'] * 2)
            bearish_score += score
            details.append(f"❌ -DI({indicators['-di']:.1f}) > +DI({indicators['+di']:.1f}) 매도 우세 (-{score:.1f}점)")

        # 3. RSI
        rsi = indicators['rsi']
        if rsi < 30:
            score = weights['rsi'] * (30 - rsi) / 10  # 과매도 강도에 비례
            bullish_score += score
            details.append(f"✅ RSI({rsi:.1f}) 과매도 (+{score:.1f}점)")
        elif rsi > 70:
            score = weights['rsi'] * (rsi - 70) / 10  # 과매수 강도에 비례
            bearish_score += score
            details.append(f"❌ RSI({rsi:.1f}) 과매수 (-{score:.1f}점)")
        elif 40 <= rsi <= 60:
            details.append(f"⚪ RSI({rsi:.1f}) 중립")
        elif rsi < 50:
            details.append(f"🟡 RSI({rsi:.1f}) 약세 구간")
        else:
            details.append(f"🟡 RSI({rsi:.1f}) 강세 구간")

        # 4. MACD
        if indicators['macd_cross_up']:
            bullish_score += weights['macd'] * 1.5  # 크로스는 강한 신호
            details.append(f"✅ MACD 골든크로스 발생 (+{weights['macd'] * 1.5:.1f}점)")
        elif indicators['macd_cross_down']:
            bearish_score += weights['macd'] * 1.5
            details.append(f"❌ MACD 데드크로스 발생 (-{weights['macd'] * 1.5:.1f}점)")
        elif indicators['macd'] > indicators['macd_signal']:
            score = weights['macd'] * 0.7
            bullish_score += score
            details.append(f"✅ MACD > Signal 상승 모멘텀 (+{score:.1f}점)")
        else:
            score = weights['macd'] * 0.7
            bearish_score += score
            details.append(f"❌ MACD < Signal 하락 모멘텀 (-{score:.1f}점)")

        # 5. 이동평균 정렬
        ma20, ma60, ma120 = indicators['ma20'], indicators['ma60'], indicators['ma120']
        close = indicators['close']

        if ma20 > ma60 > ma120 and close > ma20:
            bullish_score += weights['ma_alignment'] * 1.5
            details.append(f"✅ 완전 정배열 + 가격 상단 (+{weights['ma_alignment'] * 1.5:.1f}점)")
        elif ma20 > ma60 and close > ma20:
            bullish_score += weights['ma_alignment']
            details.append(f"✅ 부분 정배열 (+{weights['ma_alignment']:.1f}점)")
        elif ma20 < ma60 < ma120 and close < ma20:
            bearish_score += weights['ma_alignment'] * 1.5
            details.append(f"❌ 완전 역배열 + 가격 하단 (-{weights['ma_alignment'] * 1.5:.1f}점)")
        elif ma20 < ma60 and close < ma20:
            bearish_score += weights['ma_alignment']
            details.append(f"❌ 부분 역배열 (-{weights['ma_alignment']:.1f}점)")
        else:
            details.append(f"⚪ 이동평균 혼재")

        # 6. 볼린저 밴드
        bb_pos = indicators['bb_position']
        if bb_pos < 0.2:
            score = weights['bb'] * (0.2 - bb_pos) * 5
            bullish_score += score
            details.append(f"✅ BB 하단 근접({bb_pos:.2f}) (+{score:.1f}점)")
        elif bb_pos > 0.8:
            score = weights['bb'] * (bb_pos - 0.8) * 5
            bearish_score += score
            details.append(f"❌ BB 상단 근접({bb_pos:.2f}) (-{score:.1f}점)")
        else:
            details.append(f"⚪ BB 중앙 영역({bb_pos:.2f})")

        # 7. 스토캐스틱
        stoch_k = indicators['stoch_k']
        if stoch_k < 20:
            score = weights['stochastic'] * (20 - stoch_k) / 10
            bullish_score += score
            details.append(f"✅ Stoch({stoch_k:.1f}) 과매도 (+{score:.1f}점)")
        elif stoch_k > 80:
            score = weights['stochastic'] * (stoch_k - 80) / 10
            bearish_score += score
            details.append(f"❌ Stoch({stoch_k:.1f}) 과매수 (-{score:.1f}점)")

        # 8. 거래량
        vol_ratio = indicators['volume_ratio']
        if vol_ratio > 2.0:
            # 거래량 급증은 현재 추세를 강화
            if bullish_score > bearish_score:
                score = weights['volume'] * 1.5
                bullish_score += score
                details.append(f"✅ 대량 거래({vol_ratio:.2f}x) 상승 강화 (+{score:.1f}점)")
            else:
                score = weights['volume'] * 1.5
                bearish_score += score
                details.append(f"❌ 대량 거래({vol_ratio:.2f}x) 하락 강화 (-{score:.1f}점)")
        elif vol_ratio > 1.5:
            if bullish_score > bearish_score:
                score = weights['volume']
                bullish_score += score
                details.append(f"✅ 높은 거래량({vol_ratio:.2f}x) (+{score:.1f}점)")
            else:
                score = weights['volume']
                bearish_score += score
                details.append(f"❌ 높은 거래량({vol_ratio:.2f}x) (-{score:.1f}점)")
        elif vol_ratio < 0.5:
            details.append(f"⚠️ 낮은 거래량({vol_ratio:.2f}x) - 신뢰도 하락")

        return bullish_score, bearish_score, details

    def generate_signal(self, data):
        """
        스마트 매매 신호 생성

        Args:
            data: 기술적 지표가 계산된 DataFrame

        Returns:
            dict: {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0-100,
                'regime': 'trend' | 'range',
                'regime_kr': '추세장' | '횡보장',
                'bullish_score': float,
                'bearish_score': float,
                'reasoning': list of str,
                'recommendation': str
            }
        """
        try:
            # 지표 분석
            indicators = self.analyze_indicators(data)
            if not indicators:
                return self._get_no_data_result()

            # 시장 환경 감지
            regime = self.detect_market_regime(indicators)
            regime_kr = '추세장' if regime == 'trend' else '횡보장'

            # 신호 점수 계산
            bullish_score, bearish_score, details = self.calculate_signal_scores(indicators, regime)

            # 총 가능 점수 계산
            weights = self.WEIGHTS[regime]
            total_weight = sum(weights.values()) * 1.5  # 최대 가중치 고려

            # 신뢰도 계산
            if bullish_score > bearish_score:
                confidence = min((bullish_score / total_weight) * 100, 100)
                signal = 'BUY' if confidence >= 40 else 'HOLD'
            elif bearish_score > bullish_score:
                confidence = min((bearish_score / total_weight) * 100, 100)
                signal = 'SELL' if confidence >= 40 else 'HOLD'
            else:
                confidence = 0
                signal = 'HOLD'

            # 추천 메시지 생성
            if signal == 'BUY':
                if confidence >= 70:
                    recommendation = f"🟢 강력 매수 추천 ({confidence:.0f}% 신뢰도)"
                elif confidence >= 50:
                    recommendation = f"🟢 매수 관심 ({confidence:.0f}% 신뢰도)"
                else:
                    recommendation = f"🟡 약한 매수 신호 ({confidence:.0f}% 신뢰도)"
            elif signal == 'SELL':
                if confidence >= 70:
                    recommendation = f"🔴 강력 매도 추천 ({confidence:.0f}% 신뢰도)"
                elif confidence >= 50:
                    recommendation = f"🔴 매도 관심 ({confidence:.0f}% 신뢰도)"
                else:
                    recommendation = f"🟡 약한 매도 신호 ({confidence:.0f}% 신뢰도)"
            else:
                recommendation = f"⚪ 관망 권장 (매수 {bullish_score:.1f}점 vs 매도 {bearish_score:.1f}점)"

            # ATR 기반 손절/목표가 제안
            atr = indicators['atr']
            close = indicators['close']
            stop_loss = close - (atr * 2)
            take_profit = close + (atr * 3)

            return {
                'signal': signal,
                'confidence': confidence,
                'regime': regime,
                'regime_kr': regime_kr,
                'adx': indicators['adx'],
                'bullish_score': bullish_score,
                'bearish_score': bearish_score,
                'reasoning': details,
                'recommendation': recommendation,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': 1.5  # ATR 2:3 비율
            }

        except Exception as e:
            logger.error(f"신호 생성 오류: {e}")
            return self._get_error_result(str(e))

    def _get_no_data_result(self):
        """데이터 부족 시 결과"""
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'regime': 'unknown',
            'regime_kr': '분석불가',
            'adx': 0,
            'bullish_score': 0,
            'bearish_score': 0,
            'reasoning': ['데이터가 부족합니다'],
            'recommendation': '⚪ 데이터 부족',
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0
        }

    def _get_error_result(self, error_msg):
        """오류 발생 시 결과"""
        return {
            'signal': 'HOLD',
            'confidence': 0,
            'regime': 'error',
            'regime_kr': '오류',
            'adx': 0,
            'bullish_score': 0,
            'bearish_score': 0,
            'reasoning': [f'분석 오류: {error_msg}'],
            'recommendation': f'⚠️ 분석 오류',
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0
        }


# 테스트 함수
if __name__ == "__main__":
    print("Smart Signal Generator Test")
    print("=" * 60)

    # 테스트용 데이터 생성
    import yfinance as yf
    from utils import TechnicalAnalysis

    symbol = "AAPL"
    print(f"\n테스트 종목: {symbol}")

    # 데이터 다운로드
    data = yf.download(symbol, period="6mo", progress=False)

    if not data.empty:
        # 기술적 지표 계산
        ta = TechnicalAnalysis()
        data = ta.calculate_all_indicators(data)

        # 스마트 신호 생성
        generator = SmartSignalGenerator()
        result = generator.generate_signal(data)

        print(f"\n📊 분석 결과:")
        print(f"  시장 환경: {result['regime_kr']} (ADX: {result['adx']:.1f})")
        print(f"  신호: {result['signal']}")
        print(f"  신뢰도: {result['confidence']:.1f}%")
        print(f"  매수 점수: {result['bullish_score']:.1f}")
        print(f"  매도 점수: {result['bearish_score']:.1f}")
        print(f"  종합 의견: {result['recommendation']}")
        print(f"\n📍 리스크 관리:")
        print(f"  손절가: ${result['stop_loss']:.2f}")
        print(f"  목표가: ${result['take_profit']:.2f}")
        print(f"  손익비: 1:{result['risk_reward_ratio']:.1f}")
        print(f"\n🔍 상세 근거:")
        for reason in result['reasoning']:
            print(f"  {reason}")
    else:
        print("❌ 데이터를 불러올 수 없습니다")
