"""
data_service.py
데이터 관리 서비스 - 웹 API용

종목 리스트, 시장 데이터 등을 제공
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

from csv_manager import CSVDataManager
from cache_manager import get_stock_data, get_ticker_info
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
            # 마스터 데이터에서 해당 시장 가져오기
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
                'count': len(stocks_list),
                'stocks': stocks_list
            }
            
        except Exception as e:
            logger.error(f"종목 리스트 조회 오류: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
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
            # 이동평균선
            data['MA5'] = data['Close'].rolling(window=5).mean()
            data['MA10'] = data['Close'].rolling(window=10).mean()
            data['MA20'] = data['Close'].rolling(window=20).mean()
            data['MA60'] = data['Close'].rolling(window=60).mean()
            data['MA120'] = data['Close'].rolling(window=120).mean()
            data['MA240'] = data['Close'].rolling(window=240).mean()
            
            # 볼린저 밴드 (20일 기준)
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

            # RSI (14일 기준)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
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
