"""
trend_analysis.py
추세 판단 및 매매 타이밍 분석 모듈

사용법:
1. multiple/ 폴더에 이 파일을 저장
2. screener.py에서 import해서 사용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TrendTimingAnalyzer:
    def __init__(self):
        self.trend_thresholds = {
            'strong_uptrend': 75,
            'weak_uptrend': 55,
            'sideways': 45,
            'weak_downtrend': 35,
            'strong_downtrend': 25
        }
    
    def analyze_trend_and_timing(self, data):
        """종합적인 추세 및 타이밍 분석"""
        if len(data) < 120:
            return None
            
        # 1. 추세 방향 및 강도 판단
        trend_info = self.determine_trend_strength(data)
        
        # 2. 매매 타이밍 분석
        buy_timing = self.analyze_buy_timing(data, trend_info)
        sell_timing = self.analyze_sell_timing(data, trend_info)
        
        return {
            'trend_direction': trend_info['direction'],
            'trend_strength': trend_info['strength'],
            'trend_score': trend_info['score'],
            'buy_timing': buy_timing,
            'sell_timing': sell_timing,
            'recommendation': self.get_recommendation(trend_info, buy_timing, sell_timing)
        }
    
    def determine_trend_strength(self, data):
        """추세 방향과 강도를 종합 판단"""
        current = data.iloc[-1]
        
        # 1. 이동평균선 배열 점수 (40점)
        ma_score = self.calculate_ma_alignment_score(data)
        
        # 2. 가격 모멘텀 점수 (30점)  
        momentum_score = self.calculate_momentum_score(data)
        
        # 3. 거래량 확인 점수 (20점)
        volume_score = self.calculate_volume_score(data)
        
        # 4. 기술적 지표 점수 (10점)
        technical_score = self.calculate_technical_score(data)
        
        # 종합 점수 계산
        total_score = ma_score + momentum_score + volume_score + technical_score
        
        # 추세 방향 판단
        if total_score >= self.trend_thresholds['strong_uptrend']:
            direction = "강한 상승추세"
        elif total_score >= self.trend_thresholds['weak_uptrend']:
            direction = "약한 상승추세"
        elif total_score >= self.trend_thresholds['sideways']:
            direction = "횡보"
        elif total_score >= self.trend_thresholds['weak_downtrend']:
            direction = "약한 하락추세"
        else:
            direction = "강한 하락추세"
            
        return {
            'direction': direction,
            'strength': 'strong' if total_score >= 75 or total_score <= 25 else 'weak',
            'score': round(total_score, 1),
            'components': {
                'ma_score': ma_score,
                'momentum_score': momentum_score,
                'volume_score': volume_score,
                'technical_score': technical_score
            }
        }
    
    def calculate_ma_alignment_score(self, data):
        """이동평균선 배열 상태 점수화 (40점 만점)"""
        current = data.iloc[-1]
        
        # MA20, MA60, MA120 관계
        ma20 = current.get('MA20', 0)
        ma60 = current.get('MA60', 0)
        ma120 = current.get('MA120', 0)
        close = current['Close']
        
        score = 0
        
        # 완전 정배열: MA20 > MA60 > MA120 (20점)
        if ma20 > ma60 > ma120:
            score += 20
        elif ma20 > ma60:  # 부분 정배열 (10점)
            score += 10
        elif ma20 < ma60 < ma120:  # 완전 역배열 (-20점)
            score -= 20
        elif ma20 < ma60:  # 부분 역배열 (-10점)
            score -= 10
            
        # 현재가와 이동평균선 관계 (20점)
        if close > ma20 > ma60 > ma120:
            score += 20  # 모든 이평선 위
        elif close > ma20:
            score += 10  # 단기 이평선 위
        elif close < ma120:
            score -= 20  # 장기 이평선 아래
        elif close < ma60:
            score -= 10  # 중기 이평선 아래
            
        return max(0, min(40, score + 20))  # 0-40점으로 정규화
    
    def calculate_momentum_score(self, data):
        """가격 모멘텀 점수화 (30점 만점)"""
        if len(data) < 21:
            return 15  # 중립
            
        current = data.iloc[-1]
        
        # 1주, 2주, 3주 수익률
        returns = []
        for days in [5, 10, 20]:
            if len(data) > days:
                prev_price = data.iloc[-(days+1)]['Close']
                return_pct = (current['Close'] / prev_price - 1) * 100
                returns.append(return_pct)
        
        if not returns:
            return 15
            
        avg_return = np.mean(returns)
        
        # 수익률에 따른 점수 부여
        if avg_return > 10:
            return 30  # 매우 강한 상승
        elif avg_return > 5:
            return 25  # 강한 상승
        elif avg_return > 2:
            return 20  # 상승
        elif avg_return > -2:
            return 15  # 중립
        elif avg_return > -5:
            return 10  # 하락
        elif avg_return > -10:
            return 5   # 강한 하락
        else:
            return 0   # 매우 강한 하락
    
    def calculate_volume_score(self, data):
        """거래량 확인 점수화 (20점 만점)"""
        current = data.iloc[-1]
        
        # 거래량 비율 (최근 vs 평균)
        volume_ratio = current.get('Volume_Ratio', 1.0)
        
        if volume_ratio > 2.0:
            return 20  # 매우 높은 거래량
        elif volume_ratio > 1.5:
            return 16  # 높은 거래량
        elif volume_ratio > 1.2:
            return 12  # 약간 높은 거래량
        elif volume_ratio > 0.8:
            return 10  # 보통 거래량
        elif volume_ratio > 0.5:
            return 6   # 낮은 거래량
        else:
            return 2   # 매우 낮은 거래량
    
    def calculate_technical_score(self, data):
        """기술적 지표 종합 점수 (10점 만점)"""
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        score = 0
        
        # RSI 점수 (5점)
        rsi = current.get('RSI', 50)
        if 30 <= rsi <= 70:  # 적정 구간
            score += 3
        elif rsi > 70:  # 과매수
            score += 1
        elif rsi < 30:  # 과매도
            score += 2
            
        # MACD 점수 (5점)
        macd = current.get('MACD', 0)
        macd_signal = current.get('MACD_Signal', 0)
        macd_prev = prev.get('MACD', 0)
        macd_signal_prev = prev.get('MACD_Signal', 0)
        
        # MACD 골든크로스
        if macd > macd_signal and macd_prev <= macd_signal_prev:
            score += 5
        elif macd > macd_signal:
            score += 3
        elif macd < macd_signal and macd_prev >= macd_signal_prev:
            score += 0  # 데드크로스
        else:
            score += 1
            
        return score
    
    def analyze_buy_timing(self, data, trend_info):
        """매수 타이밍 분석"""
        trend_direction = trend_info['direction']
        trend_score = trend_info['score']
        
        if "상승추세" in trend_direction:
            return self.analyze_uptrend_buy_timing(data, trend_score)
        elif "하락추세" in trend_direction:
            return self.analyze_downtrend_buy_timing(data, trend_score)
        else:  # 횡보
            return self.analyze_sideways_buy_timing(data, trend_score)
    
    def analyze_uptrend_buy_timing(self, data, trend_score):
        """상승추세에서의 매수 타이밍"""
        current = data.iloc[-1]
        
        timing_score = 0
        reasons = []
        
        # 1. 조정 후 재반등 확인
        if self.is_pullback_complete(data):
            timing_score += 30
            reasons.append("조정 후 재반등")
            
        # 2. 지지선 근처 매수
        if self.is_near_support_level(data):
            timing_score += 25
            reasons.append("지지선 근처")
            
        # 3. 거래량 증가 확인
        volume_ratio = current.get('Volume_Ratio', 1.0)
        if volume_ratio > 1.3:
            timing_score += 20
            reasons.append("거래량 증가")
            
        # 4. RSI 적정 구간
        rsi = current.get('RSI', 50)
        if 40 <= rsi <= 60:
            timing_score += 15
            reasons.append("RSI 적정")
            
        # 5. 추세 강도 보너스
        if trend_score > 80:
            timing_score += 10
            reasons.append("매우 강한 추세")
            
        return {
            'score': min(100, timing_score),
            'grade': self.get_timing_grade(timing_score),
            'reasons': reasons,
            'recommendation': "매수 검토" if timing_score >= 60 else "대기"
        }
    
    def analyze_downtrend_buy_timing(self, data, trend_score):
        """하락추세에서의 매수 타이밍 (바닥 매수)"""
        timing_score = 0
        reasons = []
        
        # 하락추세에서는 보수적 접근
        if self.is_oversold_bounce_signal(data):
            timing_score += 40
            reasons.append("과매도 반등 신호")
            
        if self.is_volume_climax(data):
            timing_score += 30
            reasons.append("거래량 급증 (바닥 신호)")
            
        return {
            'score': min(100, timing_score),
            'grade': self.get_timing_grade(timing_score),
            'reasons': reasons,
            'recommendation': "신중한 매수" if timing_score >= 50 else "대기"
        }
    
    def analyze_sideways_buy_timing(self, data, trend_score):
        """횡보에서의 매수 타이밍"""
        timing_score = 0
        reasons = []
        
        if self.is_support_bounce(data):
            timing_score += 35
            reasons.append("지지선 반등")
            
        if self.is_breakout_signal(data):
            timing_score += 40
            reasons.append("돌파 신호")
            
        return {
            'score': min(100, timing_score),
            'grade': self.get_timing_grade(timing_score),
            'reasons': reasons,
            'recommendation': "선별 매수" if timing_score >= 55 else "관망"
        }
    
    def analyze_sell_timing(self, data, trend_info):
        """매도 타이밍 분석"""
        current = data.iloc[-1]
        trend_direction = trend_info['direction']
        
        timing_score = 0
        reasons = []
        
        # 1. 추세 전환 신호
        if "하락추세" in trend_direction:
            timing_score += 40
            reasons.append("하락 추세 전환")
            
        # 2. 과매수 구간
        rsi = current.get('RSI', 50)
        if rsi > 75:
            timing_score += 30
            reasons.append("과매수 구간")
            
        # 3. 거래량 급증 (매도 압력)
        volume_ratio = current.get('Volume_Ratio', 1.0)
        if volume_ratio > 2.0 and current['Close'] < current.get('MA20', 0):
            timing_score += 25
            reasons.append("대량 거래 + 이평선 이탈")
            
        # 4. 저항선 근처
        if self.is_near_resistance_level(data):
            timing_score += 20
            reasons.append("저항선 근처")
            
        return {
            'score': min(100, timing_score),
            'grade': self.get_timing_grade(timing_score),
            'reasons': reasons,
            'recommendation': "매도 검토" if timing_score >= 60 else "보유"
        }
    
    def get_timing_grade(self, score):
        """타이밍 점수를 등급으로 변환"""
        if score >= 80:
            return "★★★ 최적"
        elif score >= 60:
            return "★★ 양호"
        elif score >= 40:
            return "★ 보통"
        else:
            return "대기"
    
    def is_pullback_complete(self, data):
        """조정 완료 여부 판단"""
        if len(data) < 10:
            return False
            
        recent_data = data.iloc[-10:]
        ma20 = recent_data['MA20'].iloc[-1]
        
        # 최근 며칠간 MA20 근처에서 지지받는지 확인
        support_count = sum(recent_data['Close'] >= ma20 * 0.98)
        return support_count >= 3
    
    def is_near_support_level(self, data):
        """지지선 근처 여부 판단"""
        current = data.iloc[-1]
        ma60 = current.get('MA60', 0)
        
        # 60일선 근처 (±2%)
        return abs(current['Close'] - ma60) / ma60 <= 0.02 if ma60 > 0 else False
    
    def is_oversold_bounce_signal(self, data):
        """과매도 반등 신호 확인"""
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        rsi = current.get('RSI', 50)
        rsi_prev = prev.get('RSI', 50)
        
        # RSI가 30 아래에서 상승 전환
        return rsi < 35 and rsi > rsi_prev and rsi_prev < 30
    
    def is_volume_climax(self, data):
        """거래량 급증 (바닥 신호) 확인"""
        current = data.iloc[-1]
        volume_ratio = current.get('Volume_Ratio', 1.0)
        
        return volume_ratio > 3.0  # 평균의 3배 이상
    
    def is_support_bounce(self, data):
        """지지선 반등 확인"""
        if len(data) < 5:
            return False
            
        recent_lows = data['Low'].iloc[-5:].min()
        current_close = data.iloc[-1]['Close']
        
        # 최근 저점 대비 2% 이상 반등
        return (current_close - recent_lows) / recent_lows > 0.02 if recent_lows > 0 else False
    
    def is_breakout_signal(self, data):
        """돌파 신호 확인"""
        if len(data) < 20:
            return False
            
        current = data.iloc[-1]
        recent_high = data['High'].iloc[-20:-1].max()  # 최근 20일 고점 (오늘 제외)
        
        return current['Close'] > recent_high
    
    def is_near_resistance_level(self, data):
        """저항선 근처 여부 판단"""
        if len(data) < 20:
            return False
            
        current = data.iloc[-1]
        recent_high = data['High'].iloc[-20:].max()
        
        # 최근 고점 근처 (±2%)
        return abs(current['Close'] - recent_high) / recent_high <= 0.02 if recent_high > 0 else False
    
    def get_recommendation(self, trend_info, buy_timing, sell_timing):
        """종합 추천 의견"""
        trend_score = trend_info['score']
        buy_score = buy_timing['score']
        sell_score = sell_timing['score']
        
        if buy_score >= 70 and trend_score >= 60:
            return "적극 매수 추천"
        elif buy_score >= 50 and trend_score >= 50:
            return "매수 검토"
        elif sell_score >= 70:
            return "매도 검토"
        elif sell_score >= 50:
            return "보유 관망"
        else:
            return "중립"