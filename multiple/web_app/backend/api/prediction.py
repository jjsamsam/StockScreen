"""
Prediction API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# ✅ 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # api
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.prediction_service import prediction_service

router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str
    forecast_days: Optional[int] = 7


@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    """
    주식 예측
    
    Args:
        ticker: 종목 코드
        forecast_days: 예측 기간 (기본 7일)
    
    Returns:
        예측 결과
    """
    result = prediction_service.predict(
        ticker=request.ticker,
        forecast_days=request.forecast_days
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))
    
    return result


@router.get("/predict/{ticker}")
async def predict_stock_get(ticker: str, forecast_days: int = 7):
    """
    주식 예측 (GET 방식)
    
    Args:
        ticker: 종목 코드
        forecast_days: 예측 기간
    
    Returns:
        예측 결과
    """
    result = prediction_service.predict(
        ticker=ticker,
        forecast_days=forecast_days
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))
    
    return result


@router.post("/predict/clear-cache")
async def clear_prediction_cache():
    """예측 캐시 정리"""
    prediction_service.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/predict/settings")
async def get_prediction_settings():
    """예측 설정 조회"""
    settings = prediction_service.get_settings()
    return {"settings": settings}
