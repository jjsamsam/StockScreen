"""
prediction_service.py
UI 독립적인 순수 예측 서비스 - 웹 API용

기존 enhanced_screener.py의 EnhancedCPUPredictor를 래핑하여
FastAPI에서 사용 가능하도록 구성
"""

import sys
import os
import numpy as np

def convert_numpy_to_python(obj):
    """Numpy 타입을 표준 파이썬 타입으로 변환 (FastAPI JSON 직렬화용)"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(i) for i in obj]
    return obj

# ✅ 프로젝트 루트 디렉토리를 Python 경로에 추가
# 현재: web_app/backend/core/prediction_service.py
# 목표: multiple/ (enhanced_screener.py가 있는 곳)
current_dir = os.path.dirname(os.path.abspath(__file__))  # core
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from enhanced_screener import EnhancedCPUPredictor
from logger_config import get_logger

logger = get_logger(__name__)


class PredictionService:
    """예측 서비스 - 싱글톤 패턴"""
    
    _instance = None
    _predictor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """예측기 초기화 (lazy loading)"""
        if PredictionService._predictor is None:
            logger.info("예측 서비스 초기화 중...")
            PredictionService._predictor = EnhancedCPUPredictor()
            logger.info("예측 서비스 초기화 완료")
    
    def predict(self, ticker: str, forecast_days: int = 7) -> dict:
        """주식 예측 실행"""
        try:
            # ✅ 한국 티커 자동 보완 (숫자 6자리인 경우 .KS 추가)
            if ticker.isdigit() and len(ticker) == 6:
                original_ticker = ticker
                ticker = f"{ticker}.KS"
                logger.info(f"티커 보완: {original_ticker} -> {ticker}")

            logger.info(f"예측 요청: {ticker}, {forecast_days}일")
            
            result, error = self._predictor.predict_stock(
                ticker=ticker,
                forecast_days=forecast_days
            )
            
            if error:
                logger.error(f"예측 실패: {error}")
                return {
                    'success': False,
                    'error': error
                }
            
            logger.info(f"예측 성공: {ticker}")
            # JSON 직렬화를 위해 Numpy 타입을 파이썬 타입으로 변환
            python_result = convert_numpy_to_python(result)
            
            return {
                'success': True,
                'data': python_result
            }
            
        except Exception as e:
            logger.error(f"예측 중 예외 발생: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def clear_cache(self):
        """캐시 정리"""
        if self._predictor:
            self._predictor.clear_cache()
            logger.info("캐시 정리 완료")
    
    def get_settings(self) -> dict:
        """현재 설정 반환"""
        if self._predictor:
            return self._predictor.settings
        return {}


# 전역 인스턴스
prediction_service = PredictionService()
