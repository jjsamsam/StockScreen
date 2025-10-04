"""
CSV Data Manager - Singleton pattern for managing CSV file operations
Reduces redundant file I/O by caching CSV data in memory
"""

import pandas as pd
import os
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime, timedelta


class CSVDataManager:
    """싱글톤 CSV 데이터 매니저"""

    _instance = None
    _cache: Dict[str, tuple] = {}  # {file_path: (dataframe, timestamp)}
    _cache_duration_minutes = 30  # 캐시 유효 시간

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CSVDataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cache = {}

    def read_csv(self, file_path: str, force_reload: bool = False,
                 encoding: str = 'utf-8-sig', **kwargs) -> Optional[pd.DataFrame]:
        """
        CSV 파일 읽기 (캐싱 지원)

        Args:
            file_path: CSV 파일 경로
            force_reload: 강제 재로드 여부
            encoding: 파일 인코딩
            **kwargs: pandas.read_csv에 전달할 추가 인자

        Returns:
            DataFrame 또는 None
        """
        # 절대 경로로 변환
        abs_path = os.path.abspath(file_path)

        # 파일 존재 확인
        if not os.path.exists(abs_path):
            return None

        # 캐시 확인
        if not force_reload and abs_path in self._cache:
            cached_df, cached_time = self._cache[abs_path]

            # 캐시 유효성 확인
            if datetime.now() - cached_time < timedelta(minutes=self._cache_duration_minutes):
                # 파일 수정 시간 확인
                file_mtime = datetime.fromtimestamp(os.path.getmtime(abs_path))
                if file_mtime <= cached_time:
                    return cached_df.copy()  # 복사본 반환 (원본 보호)

        # 파일 읽기
        try:
            df = pd.read_csv(abs_path, encoding=encoding, **kwargs)
            # 캐시에 저장
            self._cache[abs_path] = (df.copy(), datetime.now())
            return df
        except Exception as e:
            print(f"CSV 읽기 오류 ({abs_path}): {e}")
            return None

    def write_csv(self, file_path: str, dataframe: pd.DataFrame,
                  encoding: str = 'utf-8-sig', index: bool = False, **kwargs) -> bool:
        """
        CSV 파일 쓰기 (캐시 무효화)

        Args:
            file_path: CSV 파일 경로
            dataframe: 저장할 DataFrame
            encoding: 파일 인코딩
            index: 인덱스 포함 여부
            **kwargs: pandas.to_csv에 전달할 추가 인자

        Returns:
            성공 여부
        """
        abs_path = os.path.abspath(file_path)

        try:
            # 디렉토리 생성 (없을 경우)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            # CSV 저장
            dataframe.to_csv(abs_path, encoding=encoding, index=index, **kwargs)

            # 캐시 업데이트
            self._cache[abs_path] = (dataframe.copy(), datetime.now())

            return True
        except Exception as e:
            print(f"CSV 쓰기 오류 ({abs_path}): {e}")
            return False

    def get_master_csv_files(self) -> List[str]:
        """
        마스터 CSV 파일 목록 가져오기

        Returns:
            마스터 CSV 파일 경로 리스트
        """
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
        for location_set in possible_locations:
            if any(os.path.exists(f) for f in location_set):
                return location_set

        return []

    def load_all_master_csvs(self, force_reload: bool = False) -> Dict[str, pd.DataFrame]:
        """
        모든 마스터 CSV 파일 로드

        Args:
            force_reload: 강제 재로드 여부

        Returns:
            {market_name: dataframe} 딕셔너리 (예: {'korea': df, 'usa': df, 'sweden': df})
        """
        # 파일 매핑: 여러 가능한 파일명 지원
        # 우선순위: _master.csv (전체 종목) > .csv (필터링된 종목)
        file_mappings = {
            'korea': [
                'stock_data/korea_stocks_master.csv',
                'master_csv/korea_stocks_master.csv',
                'stock_data/korea_stocks.csv'
            ],
            'usa': [
                'stock_data/usa_stocks_master.csv',
                'master_csv/usa_stocks_master.csv',
                'stock_data/usa_stocks.csv'
            ],
            'sweden': [
                'stock_data/sweden_stocks_master.csv',
                'master_csv/sweden_stocks_master.csv',
                'stock_data/sweden_stocks.csv'
            ]
        }

        result = {}

        for market, possible_files in file_mappings.items():
            # 존재하는 첫 번째 파일 사용
            for file_path in possible_files:
                if os.path.exists(file_path):
                    df = self.read_csv(file_path, force_reload=force_reload)
                    if df is not None and not df.empty:
                        result[market] = df
                        print(f"✅ {market}: {file_path} 로드됨 ({len(df)}개 종목)")
                        break
            else:
                print(f"⚠️ {market}: CSV 파일을 찾을 수 없음")

        return result

    def clear_cache(self, file_path: Optional[str] = None):
        """
        캐시 삭제

        Args:
            file_path: 특정 파일만 삭제 (None이면 전체 삭제)
        """
        if file_path:
            abs_path = os.path.abspath(file_path)
            if abs_path in self._cache:
                del self._cache[abs_path]
        else:
            self._cache.clear()

    def get_cache_info(self) -> Dict:
        """
        캐시 정보 가져오기

        Returns:
            캐시 정보 딕셔너리
        """
        return {
            'cached_files': len(self._cache),
            'files': list(self._cache.keys()),
            'total_memory_mb': sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024
                for df, _ in self._cache.values()
            )
        }

    def cleanup_expired_cache(self):
        """만료된 캐시 항목 제거"""
        current_time = datetime.now()
        expired_keys = []

        for file_path, (df, cached_time) in self._cache.items():
            if current_time - cached_time >= timedelta(minutes=self._cache_duration_minutes):
                expired_keys.append(file_path)

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)


# 전역 인스턴스
_csv_manager = CSVDataManager()


def read_csv(file_path: str, force_reload: bool = False, **kwargs) -> Optional[pd.DataFrame]:
    """
    편의 함수: CSV 파일 읽기

    Usage:
        from csv_manager import read_csv
        df = read_csv('data.csv')
    """
    return _csv_manager.read_csv(file_path, force_reload, **kwargs)


def write_csv(file_path: str, dataframe: pd.DataFrame, **kwargs) -> bool:
    """
    편의 함수: CSV 파일 쓰기

    Usage:
        from csv_manager import write_csv
        write_csv('data.csv', df)
    """
    return _csv_manager.write_csv(file_path, dataframe, **kwargs)


def get_master_csv_files() -> List[str]:
    """
    편의 함수: 마스터 CSV 파일 목록 가져오기

    Usage:
        from csv_manager import get_master_csv_files
        files = get_master_csv_files()
    """
    return _csv_manager.get_master_csv_files()


def load_all_master_csvs(force_reload: bool = False) -> Dict[str, pd.DataFrame]:
    """
    편의 함수: 모든 마스터 CSV 로드

    Usage:
        from csv_manager import load_all_master_csvs
        master_data = load_all_master_csvs()
    """
    return _csv_manager.load_all_master_csvs(force_reload)


def clear_csv_cache(file_path: Optional[str] = None):
    """
    편의 함수: 캐시 삭제

    Usage:
        from csv_manager import clear_csv_cache
        clear_csv_cache('data.csv')  # 특정 파일
        clear_csv_cache()  # 전체
    """
    _csv_manager.clear_cache(file_path)


def get_csv_manager() -> CSVDataManager:
    """CSV 매니저 인스턴스 가져오기"""
    return _csv_manager
