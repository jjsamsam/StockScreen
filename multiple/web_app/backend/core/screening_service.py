"""
screening_service.py
UI 독립적인 순수 스크리닝 서비스 - 웹 API용

기존 screener.py의 스크리닝 로직을 추출하여 FastAPI에서 사용
"""

import sys
import os
import pandas as pd
from typing import List, Dict, Optional, Tuple

# ✅ 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # core
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cache_manager import get_stock_data
from technical_analysis import TechnicalAnalysis
from enhanced_screening_conditions import EnhancedScreeningConditions
from logger_config import get_logger

logger = get_logger(__name__)


class ScreeningService:
    """스크리닝 서비스"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalysis()
        self.enhanced_conditions = EnhancedScreeningConditions()
    
    def screen_stocks(
        self,
        symbols: List[str],
        buy_conditions: Optional[List[str]] = None,
        sell_conditions: Optional[List[str]] = None,
        period: str = "1y",
        match_mode: str = "any"
    ) -> Dict:
        """
        종목 스크리닝 실행
        
        Args:
            symbols: 종목 코드 리스트
            buy_conditions: 매수 조건 리스트
            sell_conditions: 매도 조건 리스트
            period: 데이터 기간
            match_mode: 'all' (모두 만족), 'any' (하나라도 만족)
        
        Returns:
            dict: 스크리닝 결과
        """
        try:
            logger.info(f"스크리닝 시작: {len(symbols)}개 종목 (모드: {match_mode})")
            
            buy_results = []
            sell_results = []
            
            # 종목 정보를 위한 데이터 서비스 활용
            from core.data_service import data_service
            
            # 모든 시장 데이터 통합 (심볼 -> 이름 매핑용)
            symbol_to_name = {}
            for market in data_service.get_markets():
                df = data_service.master_data[market]
                if df is not None:
                    # 'ticker'와 'name' 컬럼 매핑
                    for _, row in df.iterrows():
                        symbol_to_name[str(row['ticker'])] = str(row['name'])

            # 조건명 매핑 (ID -> 한글명)
            condition_names = {
                "golden_cross": "골든크로스",
                "death_cross": "데드크로스",
                "rsi_oversold": "RSI과매도",
                "rsi_overbought": "RSI과매수",
                "volume_surge": "거래량급증",
                "enhanced_ma_buy": "강화된 MA 매수",
                "enhanced_bb_rsi_buy": "강화된 BB+RSI 매수",
                "enhanced_macd_volume_buy": "강화된 MACD+거래량",
                "enhanced_momentum_buy": "강화된 모멘텀 매수",
                "balanced_buy": "균형형 매수 점수",
                "ichimoku_bullish": "일목 상승 전환",
                "enhanced_technical_sell": "강화된 기술적 매도",
                "enhanced_bb_rsi_sell": "강화된 BB+RSI 매도",
                "balanced_sell": "균형형 매도 점수",
                "ichimoku_bearish": "일목 하락 전환"
            }
            
            for symbol in symbols:
                try:
                    # 데이터 가져오기
                    data = get_stock_data(symbol, period=period)
                    
                    if data is None or data.empty:
                        logger.warning(f"{symbol}: 데이터 없음")
                        continue
                    
                    # 기술적 지표 계산
                    data = self.technical_analyzer.calculate_all_indicators(data)
                    
                    # ✅ EnhancedScreeningConditions 기대 컬럼명 맞춤
                    if 'MACD_Histogram' in data.columns and 'MACD_Hist' not in data.columns:
                        data['MACD_Hist'] = data['MACD_Histogram']
                    
                    # 종목 한글명 찾기
                    stock_name = symbol_to_name.get(symbol, symbol)
                    
                    # 매수 조건 체크
                    if buy_conditions:
                        matched = self._check_conditions(data, buy_conditions, symbol, match_mode)
                        if matched:
                            score_info = self._calculate_signal_score(data)
                            # ID를 한글명으로 변환
                            readable_conditions = [condition_names.get(c, c) for c in matched]
                            buy_results.append({
                                'symbol': symbol,
                                'name': stock_name,
                                'current_price': float(data['Close'].iloc[-1]),
                                'volume': int(data['Volume'].iloc[-1]),
                                'matched_conditions': readable_conditions,
                                'matched_ids': matched,
                                'screening_score': score_info['buy_score'],
                                'risk_reward': score_info['risk_reward'],
                                'signal_details': score_info['buy_reasons']
                            })
                    
                    # 매도 조건 체크
                    if sell_conditions:
                        matched = self._check_conditions(data, sell_conditions, symbol, match_mode)
                        if matched:
                            score_info = self._calculate_signal_score(data)
                            readable_conditions = [condition_names.get(c, c) for c in matched]
                            sell_results.append({
                                'symbol': symbol,
                                'name': stock_name,
                                'current_price': float(data['Close'].iloc[-1]),
                                'volume': int(data['Volume'].iloc[-1]),
                                'matched_conditions': readable_conditions,
                                'matched_ids': matched,
                                'screening_score': score_info['sell_score'],
                                'risk_reward': score_info['risk_reward'],
                                'signal_details': score_info['sell_reasons']
                            })
                
                except Exception as e:
                    logger.error(f"{symbol} 처리 중 오류: {str(e)}")
                    continue
            
            logger.info(f"스크리닝 완료: 매수 {len(buy_results)}개, 매도 {len(sell_results)}개")
            
            return {
                'success': True,
                'buy_signals': buy_results,
                'sell_signals': sell_results,
                'total_screened': len(symbols),
                'match_mode': match_mode
            }
            
        except Exception as e:
            logger.error(f"스크리닝 중 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_conditions(self, data: pd.DataFrame, conditions: List[str], symbol: str = "Unknown", match_mode: str = "any") -> Optional[List[str]]:
        """
        조건 체크
        
        Args:
            data: 주가 데이터 (지표 포함)
            conditions: 조건 리스트
            symbol: 종목 심볼
            match_mode: 'all' (모두 만족), 'any' (하나라도 만족)
        
        Returns:
            list: 만족된 조건 리스트 (모든 조건 만족 시), 아니면 None
        """
        if data.empty or len(data) < 2:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        # 지표 존재 여부 확인 및 로깅
        missing_cols = [c for c in ['MA5', 'MA20', 'MA60', 'MA120', 'RSI', 'Volume_Ratio'] if c not in latest]
        if missing_cols:
            logger.info(f"⚠️ {symbol}: 지표 계산을 위한 데이터 부족 (누락: {missing_cols}). {len(data)}일치의 데이터만 있음.")
            return None

        matched_ids = []
        for condition in conditions:
            is_matched, detail = self._check_single_condition_enhanced(data, latest, prev, condition)
            if is_matched:
                matched_ids.append(condition)
            elif match_mode == 'all':
                # '모두 만족' 모드에서 하나라도 실패하면 바로 종료
                return None
        
        if matched_ids:
             logger.info(f"✨ {symbol} 조건 만족: {matched_ids}")
             return matched_ids
        
        return None
    
    def _check_single_condition_enhanced(self, data: pd.DataFrame, latest, prev, condition: str) -> Tuple[bool, Optional[str]]:
        """개별 조건 체크 (강화된 조건 포함)"""
        try:
            # 1. 기존 단순 조건들
            if condition == "golden_cross":
                # utils.py에는 MA5, MA20이 없을 수 있으므로 체크 후 계산 호환성 유지
                ma5_key = 'MA5' if 'MA5' in latest else 'MA_5' if 'MA_5' in latest else None
                ma20_key = 'MA20' if 'MA20' in latest else 'MA_20' if 'MA_20' in latest else None
                
                if ma5_key and ma20_key:
                    return (latest[ma5_key] > latest[ma20_key] and prev[ma5_key] <= prev[ma20_key]), "골든크로스"
                return False, None

            elif condition == "death_cross":
                ma5_key = 'MA5' if 'MA5' in latest else 'MA_5' if 'MA_5' in latest else None
                ma20_key = 'MA20' if 'MA20' in latest else 'MA_20' if 'MA_20' in latest else None
                
                if ma5_key and ma20_key:
                    return (latest[ma5_key] < latest[ma20_key] and prev[ma5_key] >= prev[ma20_key]), "데드크로스"
                return False, None

            elif condition == "rsi_oversold":
                return latest['RSI'] < 30, "RSI과매도"

            elif condition == "rsi_overbought":
                return latest['RSI'] > 70, "RSI과매수"

            elif condition == "volume_surge":
                if 'Volume_Ratio' in latest:
                    return latest['Volume_Ratio'] > 2.0, "거래량급증"
                return latest['Volume'] > prev['Volume'] * 2, "거래량급증"

            # 2. 강화된 조건들 (multiple/enhanced_screening_conditions.py)
            elif condition == "enhanced_ma_buy":
                return self.enhanced_conditions.check_ma_buy_condition_enhanced(data, latest, prev)
            
            elif condition == "enhanced_bb_rsi_buy":
                return self.enhanced_conditions.check_bb_rsi_buy_condition_enhanced(data, latest, prev)
            
            elif condition == "enhanced_macd_volume_buy":
                return self.enhanced_conditions.check_macd_volume_buy_condition_enhanced(data, latest, prev)
            
            elif condition == "enhanced_momentum_buy":
                return self.enhanced_conditions.check_momentum_buy_condition_enhanced(data, latest, prev)

            elif condition == "balanced_buy":
                score_info = self._calculate_signal_score(data)
                return score_info['buy_score'] >= 65, f"균형형 매수 점수 {score_info['buy_score']}"

            elif condition == "ichimoku_bullish":
                return self._check_ichimoku_signal(data, latest, prev, bullish=True)
            
            elif condition == "enhanced_technical_sell":
                return self.enhanced_conditions.check_technical_sell_condition_enhanced(data, latest, prev)
            
            elif condition == "enhanced_bb_rsi_sell":
                return self.enhanced_conditions.check_bb_rsi_sell_condition_enhanced(data, latest, prev)

            elif condition == "balanced_sell":
                score_info = self._calculate_signal_score(data)
                return score_info['sell_score'] >= 65, f"균형형 매도 점수 {score_info['sell_score']}"

            elif condition == "ichimoku_bearish":
                return self._check_ichimoku_signal(data, latest, prev, bullish=False)

            # 알 수 없는 조건
            else:
                logger.warning(f"알 수 없는 조건: {condition}")
                return False, None
                
        except Exception as e:
            logger.error(f"조건 체크 오류 ({condition}): {str(e)}")
            return False, None

    def _check_ichimoku_signal(self, data: pd.DataFrame, latest, prev, bullish: bool) -> Tuple[bool, Optional[str]]:
        """일목균형표 기반 추세 전환 확인."""
        required = ['Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Span_A', 'Ichimoku_Span_B']
        if any(col not in data.columns for col in required):
            return False, None

        close = float(latest['Close'])
        prev_close = float(prev['Close'])
        tenkan = float(latest['Ichimoku_Tenkan'])
        kijun = float(latest['Ichimoku_Kijun'])
        prev_tenkan = float(prev['Ichimoku_Tenkan'])
        prev_kijun = float(prev['Ichimoku_Kijun'])
        cloud_top = max(float(latest['Ichimoku_Span_A']), float(latest['Ichimoku_Span_B']))
        cloud_bottom = min(float(latest['Ichimoku_Span_A']), float(latest['Ichimoku_Span_B']))

        if bullish:
            price_above_cloud = close > cloud_top and prev_close <= cloud_top
            tk_cross = tenkan > kijun and prev_tenkan <= prev_kijun
            trend_filter = close > float(latest.get('MA60', close))
            return (price_above_cloud or tk_cross) and trend_filter, "일목 상승 전환"

        price_below_cloud = close < cloud_bottom and prev_close >= cloud_bottom
        tk_cross_down = tenkan < kijun and prev_tenkan >= prev_kijun
        trend_filter = close < float(latest.get('MA60', close))
        return (price_below_cloud or tk_cross_down) and trend_filter, "일목 하락 전환"

    def _calculate_signal_score(self, data: pd.DataFrame) -> Dict:
        """여러 지표를 종합한 보수적 스크리닝 점수."""
        current = data.iloc[-1]
        prev = data.iloc[-2]
        close = float(current['Close'])
        atr = float(current.get('ATR', 0) or 0)
        rsi = float(current.get('RSI', 50) or 50)
        adx = float(current.get('ADX', 0) or 0)
        volume_ratio = float(current.get('Volume_Ratio', 1) or 1)
        macd = float(current.get('MACD', 0) or 0)
        macd_signal = float(current.get('MACD_Signal', 0) or 0)
        macd_prev = float(prev.get('MACD', 0) or 0)
        macd_signal_prev = float(prev.get('MACD_Signal', 0) or 0)
        ma20 = float(current.get('MA20', close) or close)
        ma60 = float(current.get('MA60', close) or close)
        ma120 = float(current.get('MA120', close) or close)

        buy_score = 0
        sell_score = 0
        buy_reasons = []
        sell_reasons = []

        if close > ma20 > ma60 and ma60 >= ma120:
            buy_score += 25
            buy_reasons.append("정배열 추세")
        if close < ma20 < ma60 and ma60 <= ma120:
            sell_score += 25
            sell_reasons.append("역배열 추세")

        if 45 <= rsi <= 65:
            buy_score += 15
            buy_reasons.append("RSI 건전 구간")
        elif rsi < 35 and close > ma60:
            buy_score += 10
            buy_reasons.append("상승추세 내 과매도")
        elif rsi >= 72:
            sell_score += 15
            sell_reasons.append("RSI 과열")
        elif rsi <= 38 and close < ma60:
            sell_score += 15
            sell_reasons.append("약세 RSI")

        if macd > macd_signal and macd_prev <= macd_signal_prev:
            buy_score += 20
            buy_reasons.append("MACD 상승 전환")
        elif macd < macd_signal and macd_prev >= macd_signal_prev:
            sell_score += 20
            sell_reasons.append("MACD 하락 전환")
        elif macd > macd_signal:
            buy_score += 10
            buy_reasons.append("MACD 상승 우위")
        else:
            sell_score += 10
            sell_reasons.append("MACD 하락 우위")

        if 1.1 <= volume_ratio <= 3.0:
            buy_score += 15
            buy_reasons.append("거래량 확인")
        elif volume_ratio > 3.0 and close < float(prev['Close']):
            sell_score += 15
            sell_reasons.append("하락 대량거래")

        if adx >= 20 and close > ma60:
            buy_score += 10
            buy_reasons.append("추세 강도 양호")
        elif adx >= 20 and close < ma60:
            sell_score += 10
            sell_reasons.append("하락 추세 강도")

        cloud_top = max(float(current.get('Ichimoku_Span_A', close) or close), float(current.get('Ichimoku_Span_B', close) or close))
        cloud_bottom = min(float(current.get('Ichimoku_Span_A', close) or close), float(current.get('Ichimoku_Span_B', close) or close))
        if close > cloud_top:
            buy_score += 15
            buy_reasons.append("일목 구름 상단")
        elif close < cloud_bottom:
            sell_score += 15
            sell_reasons.append("일목 구름 하단")

        risk_reward = 1.5
        if atr > 0:
            stop_loss = close - (atr * 2)
            take_profit = close + (atr * 3)
            risk_reward = (take_profit - close) / max(close - stop_loss, 1e-9)

        return {
            'buy_score': min(100, int(buy_score)),
            'sell_score': min(100, int(sell_score)),
            'buy_reasons': buy_reasons[:5],
            'sell_reasons': sell_reasons[:5],
            'risk_reward': round(float(risk_reward), 2)
        }


# 전역 인스턴스
screening_service = ScreeningService()
