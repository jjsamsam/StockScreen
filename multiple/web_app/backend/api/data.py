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
async def get_stocks(
    market: str,
    limit: Optional[int] = Query(None, ge=1, le=10000),
    source: str = Query("csv", regex="^(csv|dynamic|hybrid)$")
):
    """
    특정 시장의 종목 리스트
    
    Args:
        market: 시장 이름 (korea, usa, sweden)
        limit: 최대 종목 수
        source: csv, dynamic, hybrid
    
    Returns:
        종목 리스트
    """
    result = data_service.get_stocks_by_source(market, limit, source)
    
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


# =============================================================================
# 📊 기술적 분석 API (NEW!)
# =============================================================================

from core.stock_analysis_service import stock_analysis_service


@router.get("/analysis/{symbol}")
async def get_stock_analysis(
    symbol: str,
    period: str = Query("6mo", regex="^(1mo|3mo|6mo|1y|2y)$")
):
    """
    종목 기술적 분석 정보 조회
    
    chart_window.py의 update_info_panel 로직을 웹앱에서 사용 가능하도록 제공
    
    Args:
        symbol: 종목 코드 (예: AAPL, 005930.KS)
        period: 분석 기간 (1mo, 3mo, 6mo, 1y, 2y)
    
    Returns:
        기술적 분석 정보:
        - price: 가격 정보 (현재가, 전일대비, 고가/저가)
        - rsi: RSI 분석 (값, 신호, 설명)
        - macd: MACD 분석 (MACD, Signal, 히스토그램)
        - bollinger: 볼린저밴드 (상단/중단/하단, 위치)
        - moving_averages: 이동평균선 (MA20, MA60, MA120, 정배열/역배열)
        - volume: 거래량 분석 (현재, 평균, 비율)
        - trend: 추세 강도 (ADX, +DI/-DI, ATR)
        - summary: 종합 의견 (매수/매도 점수, 신호)
        - risk_management: 리스크 관리 (손절가, 목표가)
    """
    result = stock_analysis_service.get_stock_info(symbol, period)
    
    if not result['success']:
        raise HTTPException(status_code=404, detail=result.get('error', 'Analysis failed'))
    
    return result


# =============================================================================
# 💹 실시간 시세 API (NEW!)
# =============================================================================

import yfinance as yf


@router.get("/quote/{symbol}")
async def get_stock_quote(symbol: str):
    """
    종목 현재가 정보 조회
    
    Args:
        symbol: 종목 코드 (예: AAPL, 005930.KS)
    
    Returns:
        현재가 정보:
        - symbol: 종목 코드
        - name: 종목명
        - price: 현재가
        - change: 전일대비 변동
        - change_percent: 전일대비 변동률(%)
        - volume: 거래량
        - prev_close: 전일 종가
        - open: 시가
        - high: 고가
        - low: 저가
        - market_cap: 시가총액
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}  # info가 None인 경우 빈 딕셔너리로 대응

        
        # 기본 정보 추출
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
        
        if current_price is None:
            # 히스토리 데이터에서 마지막 가격 가져오기
            hist = ticker.history(period='5d')
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                if len(hist) >= 2:
                    prev_close = float(hist['Close'].iloc[-2])
        
        if current_price is None:
            raise HTTPException(status_code=404, detail=f"'{symbol}' 시세 정보를 찾을 수 없습니다")
        
        # 변동 계산
        change = 0
        change_percent = 0
        if prev_close and prev_close != 0:
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
        
        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'name': info.get('shortName') or info.get('longName') or symbol,
                'price': current_price,
                'change': round(change, 2),
                'change_percent': round(change_percent, 2),
                'volume': info.get('volume') or info.get('regularMarketVolume') or 0,
                'prev_close': prev_close or 0,
                'open': info.get('open') or info.get('regularMarketOpen') or 0,
                'high': info.get('dayHigh') or info.get('regularMarketDayHigh') or 0,
                'low': info.get('dayLow') or info.get('regularMarketDayLow') or 0,
                'market_cap': info.get('marketCap') or 0,
                'currency': info.get('currency') or 'USD'
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
