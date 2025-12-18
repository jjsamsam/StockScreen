"""
Data API endpoints
"""

from fastapi import APIRouter, HTTPException, Query
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

from core.data_service import data_service

router = APIRouter()


@router.get("/markets")
async def get_markets():
    """사용 가능한 시장 목록"""
    markets = data_service.get_markets()
    return {"markets": markets}


@router.get("/stocks/{market}")
async def get_stocks(market: str, limit: Optional[int] = Query(None, ge=1, le=1000)):
    """
    특정 시장의 종목 리스트
    
    Args:
        market: 시장 이름 (korea, usa, sweden)
        limit: 최대 종목 수
    
    Returns:
        종목 리스트
    """
    result = data_service.get_stocks(market, limit)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'Market not found'))
    
    return result


@router.get("/chart/{symbol}")
async def get_chart_data(
    symbol: str,
    period: str = Query("1y", regex="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$"),
    interval: str = Query("1d", regex="^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$")
):
    """
    차트 데이터 조회
    
    Args:
        symbol: 종목 코드
        period: 기간
        interval: 간격
    
    Returns:
        OHLCV 데이터
    """
    result = data_service.get_stock_data(symbol, period, interval)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'Data not found'))
    
    return result


@router.get("/search")
async def search_stocks(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    """
    종목 검색
    
    Args:
        q: 검색어
        limit: 최대 결과 수
    
    Returns:
        검색 결과
    """
    result = data_service.search_stocks(q, limit)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result.get('error', 'Search failed'))
    
    return result
