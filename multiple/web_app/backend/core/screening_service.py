"""
screening_service.py
UI 독립적인 순수 스크리닝 서비스 - 웹 API용

기존 screener.py의 스크리닝 로직을 추출하여 FastAPI에서 사용
"""

import sys
import os
import pandas as pd
from typing import List, Dict, Optional

# ✅ 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # core
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cache_manager import get_stock_data
from utils import TechnicalAnalysis
from logger_config import get_logger

logger = get_logger(__name__)


class ScreeningService:
    """스크리닝 서비스"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalysis()
    
    def screen_stocks(
        self,
        symbols: List[str],
        buy_conditions: Optional[List[str]] = None,
        sell_conditions: Optional[List[str]] = None,
        period: str = "1y"
    ) -> Dict:
        """
        종목 스크리닝 실행
        
        Args:
            symbols: 종목 코드 리스트
            buy_conditions: 매수 조건 리스트
            sell_conditions: 매도 조건 리스트
            period: 데이터 기간
        
        Returns:
            dict: 스크리닝 결과
        """
        try:
            logger.info(f"스크리닝 시작: {len(symbols)}개 종목")
            
            buy_results = []
            sell_results = []
            
            for symbol in symbols:
                try:
                    # 데이터 가져오기
                    data = get_stock_data(symbol, period=period)
                    
                    if data is None or data.empty:
                        logger.warning(f"{symbol}: 데이터 없음")
                        continue
                    
                    # 기술적 지표 계산
                    data = self.technical_analyzer.calculate_all_indicators(data)
                    
                    # 매수 조건 체크
                    if buy_conditions:
                        if self._check_conditions(data, buy_conditions):
                            buy_results.append({
                                'symbol': symbol,
                                'current_price': float(data['Close'].iloc[-1]),
                                'volume': int(data['Volume'].iloc[-1]),
                                'matched_conditions': buy_conditions
                            })
                    
                    # 매도 조건 체크
                    if sell_conditions:
                        if self._check_conditions(data, sell_conditions):
                            sell_results.append({
                                'symbol': symbol,
                                'current_price': float(data['Close'].iloc[-1]),
                                'volume': int(data['Volume'].iloc[-1]),
                                'matched_conditions': sell_conditions
                            })
                
                except Exception as e:
                    logger.error(f"{symbol} 처리 중 오류: {str(e)}")
                    continue
            
            logger.info(f"스크리닝 완료: 매수 {len(buy_results)}개, 매도 {len(sell_results)}개")
            
            return {
                'success': True,
                'buy_signals': buy_results,
                'sell_signals': sell_results,
                'total_screened': len(symbols)
            }
            
        except Exception as e:
            logger.error(f"스크리닝 중 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_conditions(self, data: pd.DataFrame, conditions: List[str]) -> bool:
        """
        조건 체크
        
        Args:
            data: 주가 데이터 (지표 포함)
            conditions: 조건 리스트
        
        Returns:
            bool: 모든 조건 만족 여부
        """
        if data.empty or len(data) < 2:
            return False
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        for condition in conditions:
            if not self._check_single_condition(latest, prev, condition):
                return False
        
        return True
    
    def _check_single_condition(self, latest, prev, condition: str) -> bool:
        """개별 조건 체크"""
        try:
            # 골든 크로스
            if condition == "golden_cross":
                return (latest['MA_5'] > latest['MA_20'] and 
                       prev['MA_5'] <= prev['MA_20'])
            
            # 데드 크로스
            elif condition == "death_cross":
                return (latest['MA_5'] < latest['MA_20'] and 
                       prev['MA_5'] >= prev['MA_20'])
            
            # RSI 과매도
            elif condition == "rsi_oversold":
                return latest['RSI'] < 30
            
            # RSI 과매수
            elif condition == "rsi_overbought":
                return latest['RSI'] > 70
            
            # 거래량 급증
            elif condition == "volume_surge":
                avg_volume = prev['Volume']
                return latest['Volume'] > avg_volume * 2
            
            # 기본적으로 True 반환 (알 수 없는 조건)
            else:
                logger.warning(f"알 수 없는 조건: {condition}")
                return True
                
        except Exception as e:
            logger.error(f"조건 체크 오류: {str(e)}")
            return False


# 전역 인스턴스
screening_service = ScreeningService()
