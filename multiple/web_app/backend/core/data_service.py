"""
data_service.py
데이터 관리 서비스 - 웹 API용

종목 리스트, 시장 데이터 등을 제공
"""

import sys
import os
import pandas as pd
from typing import List, Dict, Optional
import yfinance as yf

# ✅ 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))  # core
backend_dir = os.path.dirname(current_dir)  # backend
webapp_dir = os.path.dirname(backend_dir)  # web_app
project_root = os.path.dirname(webapp_dir)  # multiple

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from csv_manager import CSVDataManager
from cache_manager import get_stock_data, get_ticker_info
from technical_analysis import TechnicalAnalysis
from logger_config import get_logger

logger = get_logger(__name__)


class DataService:
    """데이터 서비스"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        # 마스터 CSV 로드
        self.master_data = self.csv_manager.load_all_master_csvs()
        logger.info(f"데이터 서비스 초기화 완료: {len(self.master_data)}개 시장")
    
    def get_markets(self) -> List[str]:
        """사용 가능한 시장 목록 반환"""
        return list(self.master_data.keys())
    
    def get_stocks(self, market: str, limit: Optional[int] = None) -> Dict:
        """
        특정 시장의 종목 리스트 반환
        
        Args:
            market: 시장 이름 ('korea', 'usa', 'sweden')
            limit: 반환할 최대 종목 수
        
        Returns:
            dict: 종목 리스트
        """
        try:
            return self._get_csv_stocks(market, limit)
            
        except Exception as e:
            logger.error(f"종목 리스트 조회 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_stocks_by_source(self, market: str, limit: Optional[int] = None, source: str = "csv") -> Dict:
        """시장별 종목 리스트 조회.

        source:
            csv: 기존 마스터 CSV 사용
            dynamic: 내장 대표 종목을 yfinance 거래대금 기준으로 정렬
            hybrid: dynamic 실패 시 csv로 자동 폴백
        """
        source = (source or "csv").lower()
        if source == "csv":
            return self._get_csv_stocks(market, limit)

        if source in {"dynamic", "hybrid"}:
            dynamic = self._get_dynamic_stocks(market, limit)
            if dynamic.get("success") or source == "dynamic":
                return dynamic
            logger.warning(f"{market}: 동적 유니버스 실패, CSV로 폴백: {dynamic.get('error')}")
            fallback = self._get_csv_stocks(market, limit)
            fallback["source"] = "csv_fallback"
            fallback["source_warning"] = dynamic.get("error")
            return fallback

        return {
            'success': False,
            'error': f"지원하지 않는 종목 소스: {source}"
        }

    def _get_csv_stocks(self, market: str, limit: Optional[int] = None) -> Dict:
        """마스터 CSV 기반 종목 리스트 반환"""
        try:
            if market not in self.master_data:
                return {
                    'success': False,
                    'error': f'{market} 시장 데이터 없음'
                }
            
            stocks = self.master_data[market]
            
            if stocks is None or stocks.empty:
                return {
                    'success': False,
                    'error': f'{market} 시장 데이터 없음'
                }
            
            # limit 적용
            if limit:
                stocks = stocks.head(limit)
            
            # DataFrame을 dict 리스트로 변환
            stocks_list = stocks.to_dict('records')
            
            return {
                'success': True,
                'market': market,
                'source': 'csv',
                'count': len(stocks_list),
                'stocks': stocks_list
            }
            
        except Exception as e:
            logger.error(f"종목 리스트 조회 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _get_dynamic_stocks(self, market: str, limit: Optional[int] = None) -> Dict:
        """CSV 없이 대표 종목을 최근 거래대금 기준으로 구성.

        완전한 거래소 전 종목 대체는 아니지만, 도커 환경에서 별도 파일 갱신 없이
        유동성이 높은 감시 목록을 바로 만들 수 있는 실용적인 보조 소스입니다.
        """
        try:
            symbols = self._get_dynamic_seed_symbols(market)
            if not symbols:
                return {'success': False, 'error': f'{market} 동적 유니버스 미지원'}

            requested_limit = limit or len(symbols)
            symbols = symbols[:max(requested_limit * 2, requested_limit)]

            hist = yf.download(
                tickers=symbols,
                period="1mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )

            csv_names = self._build_symbol_name_map()
            rows = []
            for symbol in symbols:
                try:
                    if isinstance(hist.columns, pd.MultiIndex):
                        stock_hist = hist[symbol].dropna(how="all") if symbol in hist.columns.get_level_values(0) else pd.DataFrame()
                    else:
                        stock_hist = hist.dropna(how="all")

                    if stock_hist.empty or "Close" not in stock_hist or "Volume" not in stock_hist:
                        continue

                    latest = stock_hist.iloc[-1]
                    avg_turnover = float((stock_hist["Close"] * stock_hist["Volume"]).tail(20).mean())
                    rows.append({
                        'ticker': symbol,
                        'name': csv_names.get(symbol, symbol),
                        'avg_turnover_20d': avg_turnover,
                        'last_close': float(latest["Close"]),
                        'source': 'dynamic'
                    })
                except Exception as symbol_error:
                    logger.debug(f"{symbol}: 동적 유니버스 계산 스킵: {symbol_error}")

            if not rows:
                return {'success': False, 'error': '동적 종목 데이터를 가져오지 못했습니다'}

            rows = sorted(rows, key=lambda x: x.get('avg_turnover_20d', 0), reverse=True)
            rows = rows[:requested_limit]

            return {
                'success': True,
                'market': market,
                'source': 'dynamic',
                'count': len(rows),
                'stocks': rows
            }
        except Exception as e:
            logger.error(f"동적 종목 리스트 조회 오류: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _build_symbol_name_map(self) -> Dict[str, str]:
        symbol_to_name = {}
        for stocks in self.master_data.values():
            if stocks is None or stocks.empty:
                continue
            if 'ticker' not in stocks or 'name' not in stocks:
                continue
            for _, row in stocks.iterrows():
                symbol_to_name[str(row['ticker'])] = str(row['name'])
        return symbol_to_name

    def _get_dynamic_seed_symbols(self, market: str) -> List[str]:
        seeds = {
            'usa': [
                'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'BRK-B', 'JPM',
                'LLY', 'V', 'UNH', 'XOM', 'MA', 'COST', 'WMT', 'HD', 'PG', 'NFLX',
                'AMD', 'BAC', 'CRM', 'KO', 'PEP', 'ADBE', 'CSCO', 'ORCL', 'MCD', 'TMO',
                'INTC', 'QCOM', 'IBM', 'GE', 'CAT', 'GS', 'NOW', 'AMAT', 'TXN', 'SPY',
                'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'SMH'
            ],
            'korea': [
                '005930.KS', '000660.KS', '373220.KS', '207940.KS', '005380.KS', '000270.KS',
                '068270.KS', '035420.KS', '105560.KS', '012330.KS', '055550.KS', '028260.KS',
                '035720.KS', '066570.KS', '032830.KS', '086790.KS', '003550.KS', '015760.KS',
                '017670.KS', '051910.KS', '096770.KS', '034020.KS', '009150.KS', '316140.KS'
            ],
            'sweden': [
                'VOLV-B.ST', 'ERIC-B.ST', 'ATCO-A.ST', 'ASSA-B.ST', 'INVE-B.ST', 'SEB-A.ST',
                'SHB-A.ST', 'SWED-A.ST', 'HM-B.ST', 'SAND.ST', 'SKF-B.ST', 'TELIA.ST',
                'ALFA.ST', 'EPI-A.ST', 'ABB.ST', 'AZN.ST', 'ESSITY-B.ST', 'HEXA-B.ST',
                'SCA-B.ST', 'SAAB-B.ST'
            ]
        }
        return seeds.get(market, [])
    
    def get_stock_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict:
        """
        종목 데이터 조회 (기술적 지표 포함)
        
        Args:
            symbol: 종목 코드
            period: 기간
            interval: 간격
        
        Returns:
            dict: 주가 데이터 + 기술적 지표
        """
        try:
            data = get_stock_data(symbol, period=period, interval=interval)
            
            if data is None or data.empty:
                return {
                    'success': False,
                    'error': f'{symbol} 데이터 없음'
                }
            
            # ✅ 기술적 지표 계산
            data = TechnicalAnalysis.calculate_all_indicators(data)
            ichimoku_span_a_dates, ichimoku_span_a = self._build_ichimoku_forward_series(data, 'Ichimoku_Span_A')
            ichimoku_span_b_dates, ichimoku_span_b = self._build_ichimoku_forward_series(data, 'Ichimoku_Span_B')
            
            # DataFrame을 JSON 형식으로 변환
            data_dict = {
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'open': data['Open'].tolist(),
                'high': data['High'].tolist(),
                'low': data['Low'].tolist(),
                'close': data['Close'].tolist(),
                'volume': data['Volume'].tolist(),
                # 기술적 지표
                'ma5': data['MA5'].fillna(0).tolist(),
                'ma10': data['MA10'].fillna(0).tolist(),
                'ma20': data['MA20'].fillna(0).tolist(),
                'ma60': data['MA60'].fillna(0).tolist(),
                'ma120': data['MA120'].fillna(0).tolist(),
                'ma240': data['MA240'].fillna(0).tolist(),
                'bb_upper': data['BB_Upper'].fillna(0).tolist(),
                'bb_middle': data['BB_Middle'].fillna(0).tolist(),
                'bb_lower': data['BB_Lower'].fillna(0).tolist(),
                'rsi': data['RSI'].fillna(0).tolist(),
                'ichimoku_tenkan': data['Ichimoku_Tenkan'].fillna(0).tolist(),
                'ichimoku_kijun': data['Ichimoku_Kijun'].fillna(0).tolist(),
                'ichimoku_chikou': data['Ichimoku_Chikou'].fillna(0).tolist(),
                'ichimoku_span_a_dates': ichimoku_span_a_dates,
                'ichimoku_span_a': ichimoku_span_a,
                'ichimoku_span_b_dates': ichimoku_span_b_dates,
                'ichimoku_span_b': ichimoku_span_b,
            }
            
            return {
                'success': True,
                'symbol': symbol,
                'period': period,
                'data': data_dict
            }
            
        except Exception as e:
            logger.error(f"데이터 조회 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _build_ichimoku_forward_series(self, data: pd.DataFrame, column: str, forward_periods: int = 26):
        """일목 선행스팬을 미래 날짜까지 확장한 차트용 배열로 변환."""
        if data.empty or column not in data:
            return [], []

        base_dates = list(data.index)
        if not base_dates:
            return [], []

        future_dates = pd.bdate_range(
            start=base_dates[-1] + pd.offsets.BDay(1),
            periods=forward_periods
        ).to_pydatetime().tolist()
        extended_dates = base_dates + future_dates
        values = [0.0] * len(extended_dates)

        raw_values = data[column].shift(-forward_periods)
        for idx, value in enumerate(raw_values.tolist()):
            target_idx = idx + forward_periods
            if target_idx >= len(values) or pd.isna(value):
                continue
            values[target_idx] = float(value)

        date_strings = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in extended_dates]
        return date_strings, values
    
    def search_stocks(self, query: str, limit: int = 10) -> Dict:
        """
        종목 검색
        
        Args:
            query: 검색어
            limit: 최대 결과 수
        
        Returns:
            dict: 검색 결과
        """
        try:
            results = []
            
            for market in self.get_markets():
                if market not in self.master_data:
                    continue
                    
                stocks = self.master_data[market]
                
                if stocks is None or stocks.empty:
                    continue
                
                # ✅ 컬럼명: ticker, name 사용
                try:
                    mask = (
                        stocks['ticker'].astype(str).str.contains(query, case=False, na=False) |
                        stocks['name'].astype(str).str.contains(query, case=False, na=False)
                    )
                    
                    matched = stocks[mask].head(limit)
                    
                    for _, row in matched.iterrows():
                        results.append({
                            'symbol': str(row['ticker']),
                            'name': str(row['name']),
                            'market': market
                        })
                except KeyError as ke:
                    logger.error(f"{market} 시장 컬럼 오류: {ke}")
                    continue
            
            return {
                'success': True,
                'query': query,
                'count': len(results),
                'results': results[:limit]
            }
            
        except Exception as e:
            logger.error(f"검색 오류: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }


# 전역 인스턴스
data_service = DataService()
