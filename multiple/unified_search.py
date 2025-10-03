"""
Unified search module - consolidates duplicate search functions
Replaces search logic from prediction_window.py, enhanced_search.py, and screener.py
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


class UnifiedStockSearch:
    """Centralized stock search functionality"""

    def __init__(self):
        self._master_files_cache = None
        self._master_data_cache = {}

    def search_stocks(self, search_term: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        통합 주식 검색 함수

        Args:
            search_term: 검색어 (티커, 회사명, 섹터 등)
            use_cache: 캐시된 마스터 CSV 사용 여부

        Returns:
            검색 결과 리스트 [{'ticker': ..., 'name': ..., 'sector': ..., ...}, ...]
        """
        found_stocks = []
        seen_tickers = set()
        search_term_upper = search_term.strip().upper()

        if not search_term_upper:
            return []

        # 마스터 파일 목록 가져오기
        master_files = self._get_master_files()

        if not master_files:
            print("⚠️ 마스터 CSV 파일을 찾을 수 없습니다")
            return []

        # 각 마스터 CSV 파일에서 검색
        for file_path in master_files:
            if not os.path.exists(file_path):
                continue

            try:
                # 캐시 사용
                if use_cache and file_path in self._master_data_cache:
                    df = self._master_data_cache[file_path]
                else:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    if use_cache:
                        self._master_data_cache[file_path] = df

                market_name = self._get_market_name(file_path)

                # 벡터화 검색 (기존 iterrows() 대신)
                matches = self._vectorized_search(df, search_term_upper, market_name, seen_tickers)
                found_stocks.extend(matches)

            except Exception as e:
                print(f"검색 오류 ({file_path}): {e}")
                continue

        return found_stocks

    def _vectorized_search(self, df: pd.DataFrame, search_term: str, market_name: str,
                          seen_tickers: set) -> List[Dict[str, Any]]:
        """벡터화된 검색 (iterrows() 대신 사용)"""
        results = []

        # 검색 조건: 티커, 이름, 섹터에서 검색어 포함 여부
        ticker_match = df['ticker'].astype(str).str.upper().str.contains(search_term, na=False)
        name_match = df['name'].astype(str).str.upper().str.contains(search_term, na=False)

        # 섹터 컬럼이 있는 경우에만 검색
        if 'sector' in df.columns:
            sector_match = df['sector'].astype(str).str.upper().str.contains(search_term, na=False)
            mask = ticker_match | name_match | sector_match
        else:
            mask = ticker_match | name_match

        matched_df = df[mask]

        # 매칭된 결과를 딕셔너리 리스트로 변환
        for _, row in matched_df.iterrows():
            ticker = str(row.get('ticker', '')).strip()

            # 중복 제거
            if ticker in seen_tickers:
                continue
            seen_tickers.add(ticker)

            # 결과 딕셔너리 생성
            stock_info = {
                'ticker': ticker,
                'name': str(row.get('name', '')).strip(),
                'market': market_name,
                'sector': str(row.get('sector', '')).strip() if 'sector' in row else '',
                'market_cap': row.get('market_cap', 0),
            }

            # 추가 필드가 있으면 포함
            for col in ['industry', 'country', 'currency']:
                if col in row:
                    stock_info[col] = str(row.get(col, '')).strip()

            results.append(stock_info)

        return results

    def _get_master_files(self) -> List[str]:
        """마스터 CSV 파일 경로 목록 가져오기 (캐시 사용)"""
        if self._master_files_cache is not None:
            return self._master_files_cache

        # 두 가지 가능한 위치 확인
        possible_locations = [
            # 첫 번째 우선순위: master_csv 폴더
            [
                'master_csv/korea_stocks_master.csv',
                'master_csv/usa_stocks_master.csv',
                'master_csv/sweden_stocks_master.csv'
            ],
            # 두 번째 우선순위: stock_data 폴더
            [
                'stock_data/korea_stocks_master.csv',
                'stock_data/usa_stocks_master.csv',
                'stock_data/sweden_stocks_master.csv'
            ]
        ]

        # 첫 번째로 찾은 위치 사용
        master_files = []
        for location_set in possible_locations:
            if any(os.path.exists(f) for f in location_set):
                master_files = location_set
                break

        self._master_files_cache = master_files
        return master_files

    def _get_market_name(self, file_path: str) -> str:
        """파일 경로에서 시장 이름 추출"""
        file_name = Path(file_path).stem.lower()

        if 'korea' in file_name:
            return 'Korea'
        elif 'usa' in file_name or 'us' in file_name:
            return 'USA'
        elif 'sweden' in file_name:
            return 'Sweden'
        else:
            return 'Unknown'

    def clear_cache(self):
        """캐시 초기화"""
        self._master_data_cache.clear()
        self._master_files_cache = None

    def reload_master_files(self):
        """마스터 파일 재로드"""
        self.clear_cache()
        self._get_master_files()


# 전역 검색 인스턴스 (싱글톤 패턴)
_search_instance = UnifiedStockSearch()


def search_stocks(search_term: str, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    편의 함수: 주식 검색

    Usage:
        from unified_search import search_stocks
        results = search_stocks('AAPL')
        # [{'ticker': 'AAPL', 'name': 'Apple Inc.', 'market': 'USA', ...}, ...]
    """
    return _search_instance.search_stocks(search_term, use_cache)


def clear_search_cache():
    """
    편의 함수: 검색 캐시 초기화

    Usage:
        from unified_search import clear_search_cache
        clear_search_cache()
    """
    _search_instance.clear_cache()


def get_search_instance() -> UnifiedStockSearch:
    """검색 인스턴스 가져오기"""
    return _search_instance
