"""
Screening API endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# ✅ 프로젝트 루트 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # api
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.screening_service import screening_service

router = APIRouter()


class ScreeningRequest(BaseModel):
    symbols: List[str]
    buy_conditions: Optional[List[str]] = None
    sell_conditions: Optional[List[str]] = None
    period: Optional[str] = "1y"
    match_mode: Optional[str] = "any"  # 'all' (AND) 또는 'any' (OR)


@router.post("/screen")
async def screen_stocks(request: ScreeningRequest):
    """
    종목 스크리닝
    
    Args:
        symbols: 종목 코드 리스트
        buy_conditions: 매수 조건
        sell_conditions: 매도 조건
        period: 데이터 기간
    
    Returns:
        스크리닝 결과
    """
    result = screening_service.screen_stocks(
        symbols=request.symbols,
        buy_conditions=request.buy_conditions,
        sell_conditions=request.sell_conditions,
        period=request.period,
        match_mode=request.match_mode
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Screening failed'))
    
    return result


@router.get("/conditions")
async def get_available_conditions():
    """사용 가능한 스크리닝 조건 목록"""
    return {
        "buy_conditions": [
            {"id": "golden_cross", "name": "골든 크로스"},
            {"id": "rsi_oversold", "name": "RSI 과매도"},
            {"id": "volume_surge", "name": "거래량 급증"},
            {"id": "enhanced_ma_buy", "name": "강화된 MA 매수"},
            {"id": "enhanced_bb_rsi_buy", "name": "강화된 BB+RSI 매수"},
            {"id": "enhanced_macd_volume_buy", "name": "강화된 MACD+거래량"},
            {"id": "enhanced_momentum_buy", "name": "강화된 모멘텀 매수"}
        ],
        "sell_conditions": [
            {"id": "death_cross", "name": "데드 크로스"},
            {"id": "rsi_overbought", "name": "RSI 과매수"},
            {"id": "enhanced_technical_sell", "name": "강화된 기술적 매도"},
            {"id": "enhanced_bb_rsi_sell", "name": "강화된 BB+RSI 매도"}
        ]
    }
