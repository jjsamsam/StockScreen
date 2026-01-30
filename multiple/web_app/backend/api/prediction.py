"""
Prediction API endpoints - 비동기 처리 지원 버전

특징:
- 비동기 예측 API (POST /predict/async)
- 작업 상태 조회 (GET /predict/status/{task_id})
- 작업 결과 조회 (GET /predict/result/{task_id})
- 작업 취소 (POST /predict/cancel/{task_id})
- 기존 동기 API도 유지 (하위 호환성)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import sys
import os
import asyncio

# ✅ 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # api
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.prediction_service import prediction_service
from core.task_manager import task_manager, TaskStatus

router = APIRouter()


class PredictionRequest(BaseModel):
    ticker: str
    forecast_days: Optional[int] = 7


class AsyncPredictionRequest(BaseModel):
    ticker: str
    forecast_days: Optional[int] = 7
    mode: Optional[str] = "fast"  # "fast", "standard", "precise"


# =============================================================================
# 🚀 비동기 예측 API (신규)
# =============================================================================

@router.post("/predict/async")
async def predict_stock_async(request: AsyncPredictionRequest, background_tasks: BackgroundTasks):
    """
    비동기 주식 예측 시작
    
    Args:
        ticker: 종목 코드
        forecast_days: 예측 기간 (기본 7일)
        mode: 예측 모드 - "fast"(빠름), "standard"(표준), "precise"(정밀)
    
    Returns:
        task_id: 작업 ID (상태 조회에 사용)
    """
    # 작업 생성
    task_id = task_manager.create_task(
        task_type="prediction",
        params={
            "ticker": request.ticker.upper(),
            "forecast_days": request.forecast_days,
            "mode": request.mode
        }
    )
    
    # 백그라운드에서 예측 실행
    async def run_prediction():
        await task_manager.run_prediction_async(
            task_id=task_id,
            ticker=request.ticker.upper(),
            forecast_days=request.forecast_days,
            predict_func=prediction_service.predict,
            mode=request.mode or "standard"
        )
    
    # asyncio.create_task로 백그라운드 실행
    asyncio.create_task(run_prediction())
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "예측 작업이 시작되었습니다",
        "status_url": f"/api/predict/status/{task_id}",
        "result_url": f"/api/predict/result/{task_id}"
    }


@router.get("/predict/status/{task_id}")
async def get_prediction_status(task_id: str):
    """
    예측 작업 상태 조회
    
    Returns:
        status: "pending", "running", "completed", "failed", "cancelled"
        progress: 0-100
        message: 현재 상태 메시지
        elapsed_seconds: 소요 시간 (초)
    """
    status = task_manager.get_task_status(task_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")
    
    return status


@router.get("/predict/result/{task_id}")
async def get_prediction_result(task_id: str):
    """
    예측 작업 결과 조회
    
    완료된 작업의 결과를 반환합니다.
    작업이 아직 완료되지 않았으면 에러를 반환합니다.
    """
    task = task_manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")
    
    if task.status == TaskStatus.RUNNING:
        return {
            "success": False,
            "status": "running",
            "progress": task.progress,
            "message": "예측 진행 중입니다. 잠시 후 다시 조회해주세요."
        }
    
    if task.status == TaskStatus.PENDING:
        return {
            "success": False,
            "status": "pending",
            "message": "예측 대기 중입니다."
        }
    
    if task.status == TaskStatus.CANCELLED:
        return {
            "success": False,
            "status": "cancelled",
            "message": "예측이 취소되었습니다."
        }
    
    if task.status == TaskStatus.FAILED:
        return {
            "success": False,
            "status": "failed",
            "error": task.error,
            "message": "예측이 실패했습니다."
        }
    
    # 완료됨
    return {
        "success": True,
        "status": "completed",
        "data": task.result
    }


@router.post("/predict/cancel/{task_id}")
async def cancel_prediction(task_id: str):
    """
    예측 작업 취소
    
    실행 중인 작업을 취소합니다.
    """
    task = task_manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail=f"작업을 찾을 수 없습니다: {task_id}")
    
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        return {
            "success": False,
            "message": f"이미 종료된 작업입니다 (상태: {task.status.value})"
        }
    
    success = task_manager.cancel_task(task_id)
    
    return {
        "success": success,
        "message": "작업 취소 요청이 처리되었습니다" if success else "작업을 취소할 수 없습니다"
    }


# =============================================================================
# 📌 기존 동기 API (하위 호환성 유지)
# =============================================================================

@router.post("/predict")
async def predict_stock(request: PredictionRequest):
    """
    주식 예측 (동기 방식 - 기존 API)
    
    ⚠️ 주의: CPU 집약적 작업으로 인해 응답이 느릴 수 있습니다.
    빠른 응답이 필요하면 /predict/async를 사용하세요.
    
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
    주식 예측 (GET 방식 - 기존 API)
    
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
