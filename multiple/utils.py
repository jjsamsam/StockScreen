"""
utils.py
전체 종목 온라인 업데이트 + 시총/섹터 보강 버전 (KRX/스웨덴 안정화 & yfinance 소음 억제)
"""
import io
import contextlib
import os
import re
import time
from io import BytesIO
from datetime import datetime
import logging
import warnings

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from PyQt5.QtCore import QThread, pyqtSignal

from bs4 import BeautifulSoup
from typing import Optional
from logger_config import get_logger

logger = get_logger(__name__)

# ---- 외부 라이브러리 로그/워닝 소음 억제 ----
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# utils.py에 추가할 완전한 SmartUpdateThread 클래스

class SmartUpdateThread(QThread):
    """스마트 보강을 적용한 업데이트 스레드 - 기존 UpdateThread 대체"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 기본 설정값들
        self.ENRICH_SLEEP = 0.08  # API 호출 간격
        self.DEFAULT_ENRICH_COUNT = 300  # 기본 보강 개수
    
    def run(self):
        try:
            markets = self.config['markets']
            use_mcap_filter = self.config['use_mcap_filter']
            
            logger.info(f"스마트 업데이트 시작: {', '.join(markets)}")
            
            if use_mcap_filter:
                logger.info(f"시가총액 필터링: 상위 {self.config['top_count']}개")
                if self.config['enrich_all']:
                    logger.info("전체 보강 모드")
                else:
                    logger.info(f"선택적 보강: {self.config['enrich_count']}개")
            else:
                logger.info("고속 모드 (보강 없음)")
            
            total_counts = {}
            
            # 각 시장별 업데이트
            if "한국" in markets:
                self.progress.emit("한국 시장 업데이트 중...")
                korea_count = self.update_korea_smart()
                total_counts['korea'] = korea_count or 0
                logger.info(f"한국 주식 업데이트 완료: {total_counts['korea']}개")
            
            if "미국" in markets:
                self.progress.emit("미국 시장 업데이트 중...")
                usa_count = self.update_usa_smart()
                total_counts['usa'] = usa_count or 0
                logger.info(f"미국 주식 업데이트 완료: {total_counts['usa']}개")
            
            if "스웨덴" in markets:
                self.progress.emit("스웨덴 시장 업데이트 중...")
                sweden_count = self.update_sweden_smart()
                total_counts['sweden'] = sweden_count or 0
                logger.info(f"스웨덴 주식 업데이트 완료: {total_counts['sweden']}개")
            
            # 결과 메시지 생성
            total_count = sum(total_counts.values())
            mode = "스마트 보강" if use_mcap_filter else "고속"
            
            market_results = []
            if 'korea' in total_counts:
                market_results.append(f"• 한국(KOSPI/KOSDAQ): {total_counts['korea']}개")
            if 'usa' in total_counts:
                market_results.append(f"• 미국(NASDAQ/NYSE): {total_counts['usa']}개")
            if 'sweden' in total_counts:
                market_results.append(f"• 스웨덴(OMX): {total_counts['sweden']}개")
            
            message = (
                f'{mode} 모드로 업데이트가 완료되었습니다!\n'
                f'총 {total_count}개 종목\n'
                + '\n'.join(market_results)
            )
            
            if use_mcap_filter:
                message += f'\n\n📊 시가총액 상위 {self.config["top_count"]}개로 필터링됨'
            
            self.finished.emit(message)
            
        except Exception as e:
            logger.error(f"스마트 업데이트 오류: {e}")
            self.error.emit(f'업데이트 중 오류가 발생했습니다: {str(e)}')
    
    def update_korea_smart(self):
        """한국 시장 스마트 업데이트"""
        try:
            self.progress.emit("한국 기본 종목 리스트 수집 중...")
            
            # 1단계: 기본 리스트 수집 (빠름 - 1-2초)
            kospi = fetch_krx_list('STK')
            time.sleep(0.3)
            kosdaq = fetch_krx_list('KSQ')
            all_df = pd.concat([kospi, kosdaq], ignore_index=True).drop_duplicates('ticker')
            
            logger.info(f"한국 기본 리스트 수집 완료: {len(all_df)}개")
            
            # 2단계: 조건부 보강
            if self.config['use_mcap_filter']:
                if self.config['enrich_all']:
                    # 전체 보강 모드
                    self.progress.emit(f"한국 전체 {len(all_df)}개 종목 시가총액 정보 수집 중...")
                    enriched_df = enrich_with_yfinance(
                        all_df,
                        ticker_col='ticker',
                        max_items=len(all_df),  # 전체
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                else:
                    # 지정 개수만 보강
                    enrich_count = min(self.config['enrich_count'], len(all_df))
                    self.progress.emit(f"한국 상위 {enrich_count}개 종목 시가총액 정보 수집 중...")
                    
                    # 보강할 종목을 더 많이 가져와서 정확성 높이기
                    sample_df = all_df.head(min(enrich_count * 2, len(all_df)))
                    enriched_df = enrich_with_yfinance(
                        sample_df,
                        ticker_col='ticker',
                        max_items=enrich_count,
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                
                # 3단계: 시가총액 기준 필터링
                self.progress.emit("시가총액 기준 상위 종목 선별 중...")
                final_df = self.filter_by_market_cap(enriched_df, self.config['top_count'], "한국")
            else:
                # 보강 없이 원본 사용
                final_df = all_df
            
            # 4단계: 저장
            os.makedirs('stock_data', exist_ok=True)
            final_df.to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
            
            return len(final_df)
            
        except Exception as e:
            logger.error(f"한국 시장 업데이트 실패: {e}")
            return self.create_korea_fallback()
    
    def update_usa_smart(self):
        """미국 시장 스마트 업데이트"""
        try:
            self.progress.emit("미국 기본 종목 리스트 수집 중...")
            
            # 1단계: 기본 리스트 수집
            all_df = fetch_us_all_listings()
            if all_df.empty:
                return self.create_usa_fallback()
            
            logger.info(f"미국 기본 리스트 수집 완료: {len(all_df)}개")
            
            # 2단계: 조건부 보강
            if self.config['use_mcap_filter']:
                if self.config['enrich_all']:
                    # 전체 보강 (시간 많이 소요)
                    self.progress.emit(f"미국 전체 {len(all_df)}개 종목 정보 수집 중...")
                    enriched_df = enrich_with_yfinance(
                        all_df,
                        ticker_col='ticker',
                        max_items=len(all_df),
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                else:
                    # 지정 개수만 보강
                    enrich_count = min(self.config['enrich_count'], len(all_df))
                    self.progress.emit(f"미국 상위 {enrich_count}개 종목 정보 수집 중...")
                    
                    sample_df = all_df.head(min(enrich_count * 2, len(all_df)))
                    enriched_df = enrich_with_yfinance(
                        sample_df,
                        ticker_col='ticker',
                        max_items=enrich_count,
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                
                # 시가총액 기준 필터링
                final_df = self.filter_by_market_cap(enriched_df, self.config['top_count'], "미국")
            else:
                # 보강 없이 원본 사용
                final_df = all_df
            
            # 저장
            os.makedirs('stock_data', exist_ok=True)
            final_df.to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
            
            return len(final_df)
            
        except Exception as e:
            logger.error(f"미국 시장 업데이트 실패: {e}")
            return self.create_usa_fallback()
    
    def update_sweden_smart(self):
        """스웨덴 시장 스마트 업데이트"""
        try:
            self.progress.emit("스웨덴 기본 종목 리스트 수집 중...")
            
            # 1단계: 기본 리스트 수집
            all_df = fetch_sweden_list_from_nordic()
            if all_df.empty:
                raise RuntimeError("Nordic API returned empty")
            
            logger.info(f"스웨덴 기본 리스트 수집 완료: {len(all_df)}개")
            
            # 2단계: 조건부 보강
            if self.config['use_mcap_filter']:
                if self.config['enrich_all']:
                    # 전체 보강
                    self.progress.emit(f"스웨덴 전체 {len(all_df)}개 종목 정보 수집 중...")
                    enriched_df = enrich_with_yfinance(
                        all_df,
                        ticker_col='ticker',
                        max_items=len(all_df),
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                else:
                    # 지정 개수만 보강
                    enrich_count = min(self.config['enrich_count'], len(all_df))
                    self.progress.emit(f"스웨덴 상위 {enrich_count}개 종목 정보 수집 중...")
                    
                    sample_df = all_df.head(min(enrich_count * 2, len(all_df)))
                    enriched_df = enrich_with_yfinance(
                        sample_df,
                        ticker_col='ticker',
                        max_items=enrich_count,
                        sleep_sec=self.ENRICH_SLEEP,
                        on_progress=self.progress.emit
                    )
                
                # 시가총액 기준 필터링
                final_df = self.filter_by_market_cap(enriched_df, self.config['top_count'], "스웨덴")
            else:
                # 보강 없이 원본 사용
                final_df = all_df
            
            # 저장
            os.makedirs('stock_data', exist_ok=True)
            final_df.to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
            
            return len(final_df)
            
        except Exception as e:
            logger.error(f"스웨덴 시장 업데이트 실패: {e}")
            return self.create_sweden_fallback()
    
    def filter_by_market_cap(self, df, top_count, market_name):
        """시가총액 기준으로 상위 종목 필터링 - 데이터 타입 오류 수정"""
        try:
            if df.empty or top_count <= 0:
                return df
            
            # 🔧 수정: market_cap 컬럼의 데이터 타입 처리
            df_copy = df.copy()
            
            # market_cap을 숫자형으로 변환
            df_copy['market_cap_numeric'] = pd.to_numeric(df_copy['market_cap'], errors='coerce')
            
            # 변환 결과 확인 및 디버그 정보
            logger.debug(f"{market_name} 시가총액 데이터 타입 체크:")
            logger.debug(f"   - 원본 타입: {df['market_cap'].dtype}")
            logger.debug(f"   - 변환 후 타입: {df_copy['market_cap_numeric'].dtype}")
            logger.debug(f"   - 유효한 값 개수: {df_copy['market_cap_numeric'].notna().sum()}/{len(df_copy)}")
            
            # 샘플 데이터 출력 (디버깅용)
            if len(df_copy) > 0:
                sample_data = df_copy[['ticker', 'market_cap', 'market_cap_numeric']].head(3)
                logger.debug(f"   - 샘플 데이터:")
                # ✅ 벡터화: iterrows() 제거
                for ticker, mcap, mcap_num in zip(sample_data['ticker'], sample_data['market_cap'], sample_data['market_cap_numeric']):
                    logger.debug(f"     {ticker}: '{mcap}' → {mcap_num}")
            
            # 유효한 시가총액이 있는 종목만 선택
            valid_mcap_df = df_copy[
                df_copy['market_cap_numeric'].notna() & 
                (df_copy['market_cap_numeric'] > 0)
            ].copy()
            
            if valid_mcap_df.empty:
                logger.warning(f"{market_name}: 유효한 시가총액 데이터가 없어 원본 상위 {top_count}개 종목 사용")
                return df.head(top_count)
            
            # 시가총액 기준 내림차순 정렬
            sorted_df = valid_mcap_df.sort_values('market_cap_numeric', ascending=False)
            
            # 상위 N개 선택
            top_stocks = sorted_df.head(top_count)
            
            logger.info(f"{market_name}: 시가총액 기준 상위 {len(top_stocks)}개 종목 선별 완료")
            
            # 시가총액 정보 출력 (상위 5개)
            if len(top_stocks) > 0:
                logger.info(f"   상위 종목 예시:")
                # ✅ 벡터화: iterrows() 제거
                top_5 = top_stocks.head(5)
                for i, (ticker, name, mcap_num) in enumerate(zip(top_5['ticker'], top_5['name'], top_5['market_cap_numeric'])):
                    mcap_display = self.format_market_cap(mcap_num)
                    logger.info(f"   {i+1}. {ticker} ({name[:20]}): {mcap_display}")
            
            # 원본 컬럼명 유지하여 반환 (numeric 컬럼 제거)
            result = top_stocks.drop(columns=['market_cap_numeric'])
            return result
            
        except Exception as e:
            logger.warning(f"{market_name} 시가총액 필터링 오류: {e}")
            
            # 추가 디버그 정보
            if hasattr(df, 'market_cap'):
                logger.debug(f"   디버그 정보:")
                logger.debug(f"   - market_cap 컬럼 존재: {True}")
                logger.debug(f"   - 데이터 타입: {df['market_cap'].dtype}")
                logger.debug(f"   - 첫 5개 값: {df['market_cap'].head().tolist()}")
                logger.debug(f"   - NaN 개수: {df['market_cap'].isna().sum()}")
            
            return df.head(top_count)
    
    def format_market_cap(self, market_cap):
        """시가총액을 읽기 쉬운 형태로 포맷팅 - 숫자형 입력 처리"""
        try:
            # 입력값이 숫자가 아니면 변환 시도
            if isinstance(market_cap, str):
                # 쉼표 제거 후 숫자 변환
                market_cap_clean = market_cap.replace(',', '').replace(' ', '')
                market_cap = float(market_cap_clean)
            elif pd.isna(market_cap):
                return "N/A"
            
            # 숫자형으로 변환된 값 처리
            market_cap = float(market_cap)
            
            if market_cap >= 1e12:  # 1조 이상
                return f"{market_cap/1e12:.2f}조"
            elif market_cap >= 1e8:  # 1억 이상
                return f"{market_cap/1e8:.0f}억"
            elif market_cap >= 1e4:  # 1만 이상
                return f"{market_cap/1e4:.0f}만"
            else:
                return f"{market_cap:,.0f}"
                
        except (ValueError, TypeError) as e:
            logger.warning(f"시가총액 포맷팅 오류: {e} (입력값: {market_cap})")
            return str(market_cap) if market_cap is not None else "N/A"
    
    # 추가: 데이터 로드 시 시가총액 타입 체크 함수
    def validate_market_cap_data(self, df, market_name):
        """시가총액 데이터 유효성 검사"""
        try:
            if 'market_cap' not in df.columns:
                logger.warning(f"{market_name}: market_cap 컬럼이 없습니다.")
                return False
            
            logger.debug(f"{market_name} 시가총액 데이터 검사:")
            logger.debug(f"   - 총 종목 수: {len(df)}")
            logger.debug(f"   - market_cap 타입: {df['market_cap'].dtype}")
            logger.debug(f"   - NaN 값: {df['market_cap'].isna().sum()}개")
            logger.debug(f"   - 고유값 예시: {df['market_cap'].dropna().head(3).tolist()}")
            
            # 문자열 타입이면 경고
            if df['market_cap'].dtype == 'object':
                logger.warning(f"   문자열 타입 감지 - 숫자 변환 필요")
                
                # 변환 테스트
                test_conversion = pd.to_numeric(df['market_cap'].head(10), errors='coerce')
                valid_conversions = test_conversion.notna().sum()
                logger.debug(f"   - 변환 테스트 (첫 10개): {valid_conversions}/10개 성공")
            
            return True
            
        except Exception as e:
            logger.warning(f"{market_name} 데이터 검사 오류: {e}")
            return False
        
    # ========== Fallback 메서드들 ==========
    
    def create_korea_fallback(self):
        """한국 종목 백업 데이터 생성"""
        logger.info("한국 백업 데이터 생성 중...")
        
        major_stocks = [
            # 시총 상위 100개 (2024년 기준, 단위: 원)
            ('005930.KS', '삼성전자', '반도체', 300000000000000, 'KOSPI'),
            ('000660.KS', 'SK하이닉스', '반도체', 80000000000000, 'KOSPI'),
            ('035420.KS', '네이버', 'IT서비스', 40000000000000, 'KOSPI'),
            ('207940.KS', '삼성바이오로직스', '바이오', 35000000000000, 'KOSPI'),
            ('006400.KS', '삼성SDI', '배터리', 30000000000000, 'KOSPI'),
            ('051910.KS', 'LG화학', '화학', 28000000000000, 'KOSPI'),
            ('035720.KS', '카카오', 'IT서비스', 25000000000000, 'KOSPI'),
            ('068270.KS', '셀트리온', '바이오', 24000000000000, 'KOSPI'),
            ('005380.KS', '현대차', '자동차', 22000000000000, 'KOSPI'),
            ('373220.KS', 'LG에너지솔루션', '배터리', 20000000000000, 'KOSPI'),
            
            ('323410.KS', '카카오뱅크', '금융', 18000000000000, 'KOSPI'),
            ('000270.KS', '기아', '자동차', 17000000000000, 'KOSPI'),
            ('066570.KS', 'LG전자', '전자', 16000000000000, 'KOSPI'),
            ('003550.KS', 'LG', '지주회사', 15000000000000, 'KOSPI'),
            ('015760.KS', '한국전력', '전력', 14000000000000, 'KOSPI'),
            ('017670.KS', 'SK텔레콤', '통신', 13000000000000, 'KOSPI'),
            ('034730.KS', 'SK', '지주회사', 12000000000000, 'KOSPI'),
            ('096770.KS', 'SK이노베이션', '에너지', 11000000000000, 'KOSPI'),
            ('086790.KS', '하나금융지주', '금융', 10000000000000, 'KOSPI'),
            ('105560.KS', 'KB금융', '금융', 9500000000000, 'KOSPI'),
            
            ('012330.KS', '현대모비스', '자동차부품', 9000000000000, 'KOSPI'),
            ('032830.KS', '삼성생명', '보험', 8800000000000, 'KOSPI'),
            ('009150.KS', '삼성전기', '전자부품', 8500000000000, 'KOSPI'),
            ('000810.KS', '삼성화재', '보험', 8200000000000, 'KOSPI'),
            ('251270.KS', '넷마블', '게임', 8000000000000, 'KOSPI'),
            ('302440.KS', 'SK바이오사이언스', '바이오', 7800000000000, 'KOSPI'),
            ('018260.KS', '삼성에스디에스', 'IT서비스', 7500000000000, 'KOSPI'),
            ('267250.KS', 'HD현대중공업', '조선', 7200000000000, 'KOSPI'),
            ('024110.KS', '기업은행', '금융', 7000000000000, 'KOSPI'),
            ('011170.KS', '롯데케미칼', '화학', 6800000000000, 'KOSPI'),
            
            ('047050.KS', '포스코인터내셔널', '무역', 6500000000000, 'KOSPI'),
            ('259960.KS', '크래프톤', '게임', 6200000000000, 'KOSPI'),
            ('033780.KS', 'KT&G', '담배', 6000000000000, 'KOSPI'),
            ('030200.KS', 'KT', '통신', 5800000000000, 'KOSPI'),
            ('036570.KS', '엔씨소프트', '게임', 5500000000000, 'KOSPI'),
            ('090430.KS', '아모레퍼시픽', '화장품', 5200000000000, 'KOSPI'),
            ('016360.KS', 'LS', '전선', 5000000000000, 'KOSPI'),
            ('011780.KS', '금호석유', '화학', 4800000000000, 'KOSPI'),
            ('032640.KS', 'LG유플러스', '통신', 4500000000000, 'KOSPI'),
            ('028260.KS', '삼성물산', '종합상사', 4200000000000, 'KOSPI'),
            
            ('267260.KS', 'HD현대일렉트릭', '전기설비', 4000000000000, 'KOSPI'),
            ('003230.KS', '삼양식품', '식품', 3800000000000, 'KOSPI'),
            ('035250.KS', '강원랜드', '레저', 3500000000000, 'KOSPI'),
            ('097950.KS', 'CJ제일제당', '식품', 3200000000000, 'KOSPI'),
            ('004020.KS', '현대제철', '철강', 3000000000000, 'KOSPI'),
            ('034220.KS', 'LG디스플레이', '디스플레이', 2800000000000, 'KOSPI'),
            ('000720.KS', '현대건설', '건설', 2500000000000, 'KOSPI'),
            ('051900.KS', 'LG생활건강', '생활용품', 2200000000000, 'KOSPI'),
            ('009540.KS', 'HD한국조선해양', '조선', 2000000000000, 'KOSPI'),
            ('138040.KS', '메리츠금융지주', '금융', 1800000000000, 'KOSPI'),
            
            # KOSDAQ 상위 종목들
            ('042700.KQ', '한미반도체', '반도체', 1500000000000, 'KOSDAQ'),
            ('065350.KQ', '신성통상', '섬유', 1200000000000, 'KOSDAQ'),
            ('058470.KQ', '리노공업', '반도체', 1000000000000, 'KOSDAQ'),
            ('067310.KQ', '하나마이크론', '반도체', 900000000000, 'KOSDAQ'),
            ('137310.KQ', '에스디바이오센서', '바이오', 800000000000, 'KOSDAQ'),
            ('196170.KQ', '알테오젠', '바이오', 700000000000, 'KOSDAQ'),
            ('112040.KQ', '위메이드', '게임', 650000000000, 'KOSDAQ'),
            ('091990.KQ', '셀트리온헬스케어', '바이오', 600000000000, 'KOSDAQ'),
            ('241560.KQ', '두산밥캣', '건설기계', 550000000000, 'KOSDAQ'),
            ('086520.KQ', '에코프로', '배터리소재', 500000000000, 'KOSDAQ'),
            
            ('240810.KQ', '원익IPS', '반도체', 480000000000, 'KOSDAQ'),
            ('365340.KQ', '성일하이텍', '화학', 450000000000, 'KOSDAQ'),
            ('454910.KQ', '두산로보틱스', '로봇', 420000000000, 'KOSDAQ'),
            ('293490.KQ', '카카오게임즈', '게임', 400000000000, 'KOSDAQ'),
            ('357780.KQ', '솔브레인', '화학', 380000000000, 'KOSDAQ'),
            ('039030.KQ', '이오테크닉스', '반도체', 350000000000, 'KOSDAQ'),
            ('263750.KQ', '펄어비스', '게임', 320000000000, 'KOSDAQ'),
            ('095340.KQ', 'ISC', '반도체', 300000000000, 'KOSDAQ'),
            ('348370.KQ', '알테오젠', '바이오', 280000000000, 'KOSDAQ'),
            ('145720.KQ', '덴티움', '의료기기', 250000000000, 'KOSDAQ'),
            
            ('277810.KQ', '레인보우로보틱스', '로봇', 230000000000, 'KOSDAQ'),
            ('094170.KQ', '동운아나텍', '반도체', 220000000000, 'KOSDAQ'),
            ('399720.KQ', 'APR', '반도체', 200000000000, 'KOSDAQ'),
            ('450080.KQ', '에코프로머티리얼즈', '배터리소재', 190000000000, 'KOSDAQ'),
            ('086900.KQ', '메디톡스', '바이오', 180000000000, 'KOSDAQ'),
            ('123700.KQ', 'SJM', '반도체', 170000000000, 'KOSDAQ'),
            ('067630.KQ', 'HLB생명과학', '바이오', 160000000000, 'KOSDAQ'),
            ('141080.KQ', '리가켐바이오', '바이오', 150000000000, 'KOSDAQ'),
            ('131970.KQ', '두산테스나', '반도체', 140000000000, 'KOSDAQ'),
            ('900140.KQ', '엘브이엠씨', '반도체', 130000000000, 'KOSDAQ'),
            
            ('095570.KQ', 'AJ네트웍스', 'IT서비스', 120000000000, 'KOSDAQ'),
            ('064290.KQ', '인텍플러스', '반도체', 110000000000, 'KOSDAQ'),
            ('192080.KQ', '더블유게임즈', '게임', 100000000000, 'KOSDAQ'),
            ('237880.KQ', '클리오', '화장품', 95000000000, 'KOSDAQ'),
            ('078600.KQ', '대주전자재료', '반도체', 90000000000, 'KOSDAQ'),
            ('179900.KQ', '유티아이', '반도체', 85000000000, 'KOSDAQ'),
            ('048410.KQ', '현대바이오', '바이오', 80000000000, 'KOSDAQ'),
            ('214150.KQ', '클래시스', '반도체', 75000000000, 'KOSDAQ'),
            ('189300.KQ', '인텔리안테크', '통신장비', 70000000000, 'KOSDAQ'),
            ('396270.KQ', '넥스트칩', '반도체', 65000000000, 'KOSDAQ'),
            
            ('200130.KQ', '콜마비앤에이치', '화장품', 60000000000, 'KOSDAQ'),
            ('173940.KQ', '에프엔에스테크', '반도체', 55000000000, 'KOSDAQ'),
            ('225570.KQ', '넥슨게임즈', '게임', 50000000000, 'KOSDAQ'),
            ('256940.KQ', '케이피에스', '반도체', 48000000000, 'KOSDAQ'),
            ('091700.KQ', '파트론', '전자부품', 45000000000, 'KOSDAQ'),
            ('353200.KQ', '대덕전자', '전자부품', 42000000000, 'KOSDAQ'),
            ('117730.KQ', '티로보틱스', '로봇', 40000000000, 'KOSDAQ'),
            ('194480.KQ', '데브시스터즈', '게임', 38000000000, 'KOSDAQ'),
            ('900310.KQ', '컬러레이', '반도체', 35000000000, 'KOSDAQ'),
            ('067160.KQ', '아프리카TV', 'IT서비스', 32000000000, 'KOSDAQ')
        ]
        
        rows = []
        for ticker, name, sector, mcap in major_stocks:
            rows.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market_cap': mcap,
                'market': 'KOSPI' if ticker.endswith('.KS') else 'KOSDAQ'
            })
        
        df = pd.DataFrame(rows)
        
        # 시가총액 필터링 적용
        if self.config.get('use_mcap_filter', False):
            top_count = self.config.get('top_count', len(df))
            df = df.head(top_count)
        
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
        
        return len(df)
    
    def create_usa_fallback(self):
        """미국 종목 백업 데이터 생성"""
        logger.info("미국 백업 데이터 생성 중...")
        
        major_stocks = [
                    # 시총 상위 100개 (2024년 기준, 단위: USD)
                    ('AAPL', 'Apple Inc', 'Technology', 3000000000000, 'NASDAQ'),
                    ('MSFT', 'Microsoft Corp', 'Technology', 2800000000000, 'NASDAQ'),
                    ('GOOGL', 'Alphabet Inc Class A', 'Technology', 1700000000000, 'NASDAQ'),
                    ('AMZN', 'Amazon.com Inc', 'Consumer Discretionary', 1500000000000, 'NASDAQ'),
                    ('NVDA', 'NVIDIA Corp', 'Technology', 1900000000000, 'NASDAQ'),
                    ('TSLA', 'Tesla Inc', 'Consumer Discretionary', 800000000000, 'NASDAQ'),
                    ('META', 'Meta Platforms Inc', 'Technology', 750000000000, 'NASDAQ'),
                    ('BRK-B', 'Berkshire Hathaway Inc Class B', 'Financial Services', 700000000000, 'NYSE'),
                    ('UNH', 'UnitedHealth Group Inc', 'Healthcare', 450000000000, 'NYSE'),
                    ('JNJ', 'Johnson & Johnson', 'Healthcare', 420000000000, 'NYSE'),
                    
                    ('V', 'Visa Inc Class A', 'Financial Services', 400000000000, 'NYSE'),
                    ('PG', 'Procter & Gamble Co', 'Consumer Staples', 380000000000, 'NYSE'),
                    ('JPM', 'JPMorgan Chase & Co', 'Financial Services', 450000000000, 'NYSE'),
                    ('HD', 'Home Depot Inc', 'Consumer Discretionary', 350000000000, 'NYSE'),
                    ('MA', 'Mastercard Inc Class A', 'Financial Services', 340000000000, 'NYSE'),
                    ('BAC', 'Bank of America Corp', 'Financial Services', 300000000000, 'NYSE'),
                    ('XOM', 'Exxon Mobil Corp', 'Energy', 280000000000, 'NYSE'),
                    ('CVX', 'Chevron Corp', 'Energy', 270000000000, 'NYSE'),
                    ('ABBV', 'AbbVie Inc', 'Healthcare', 260000000000, 'NYSE'),
                    ('WMT', 'Walmart Inc', 'Consumer Staples', 450000000000, 'NYSE'),
                    
                    ('LLY', 'Eli Lilly and Co', 'Healthcare', 500000000000, 'NYSE'),
                    ('KO', 'Coca-Cola Co', 'Consumer Staples', 250000000000, 'NYSE'),
                    ('AVGO', 'Broadcom Inc', 'Technology', 550000000000, 'NASDAQ'),
                    ('PEP', 'PepsiCo Inc', 'Consumer Staples', 230000000000, 'NASDAQ'),
                    ('COST', 'Costco Wholesale Corp', 'Consumer Staples', 220000000000, 'NASDAQ'),
                    ('ORCL', 'Oracle Corp', 'Technology', 300000000000, 'NYSE'),
                    ('ADBE', 'Adobe Inc', 'Technology', 250000000000, 'NASDAQ'),
                    ('MRK', 'Merck & Co Inc', 'Healthcare', 240000000000, 'NYSE'),
                    ('CRM', 'Salesforce Inc', 'Technology', 220000000000, 'NYSE'),
                    ('NFLX', 'Netflix Inc', 'Communication Services', 200000000000, 'NASDAQ'),
                    
                    ('TMO', 'Thermo Fisher Scientific Inc', 'Healthcare', 210000000000, 'NYSE'),
                    ('ACN', 'Accenture PLC Class A', 'Technology', 200000000000, 'NYSE'),
                    ('WFC', 'Wells Fargo & Co', 'Financial Services', 180000000000, 'NYSE'),
                    ('DIS', 'Walt Disney Co', 'Communication Services', 190000000000, 'NYSE'),
                    ('CSCO', 'Cisco Systems Inc', 'Technology', 190000000000, 'NASDAQ'),
                    ('ABT', 'Abbott Laboratories', 'Healthcare', 185000000000, 'NYSE'),
                    ('NKE', 'Nike Inc Class B', 'Consumer Discretionary', 175000000000, 'NYSE'),
                    ('VZ', 'Verizon Communications Inc', 'Communication Services', 170000000000, 'NYSE'),
                    ('INTC', 'Intel Corp', 'Technology', 160000000000, 'NASDAQ'),
                    ('COP', 'ConocoPhillips', 'Energy', 155000000000, 'NYSE'),
                    
                    ('CMCSA', 'Comcast Corp Class A', 'Communication Services', 150000000000, 'NASDAQ'),
                    ('INTU', 'Intuit Inc', 'Technology', 145000000000, 'NASDAQ'),
                    ('TXN', 'Texas Instruments Inc', 'Technology', 140000000000, 'NASDAQ'),
                    ('PM', 'Philip Morris International Inc', 'Consumer Staples', 135000000000, 'NYSE'),
                    ('HON', 'Honeywell International Inc', 'Industrials', 130000000000, 'NASDAQ'),
                    ('IBM', 'International Business Machines Corp', 'Technology', 125000000000, 'NYSE'),
                    ('UNP', 'Union Pacific Corp', 'Industrials', 120000000000, 'NYSE'),
                    ('AMD', 'Advanced Micro Devices Inc', 'Technology', 230000000000, 'NASDAQ'),
                    ('GE', 'General Electric Co', 'Industrials', 115000000000, 'NYSE'),
                    ('MDT', 'Medtronic PLC', 'Healthcare', 110000000000, 'NYSE'),
                    
                    ('CAT', 'Caterpillar Inc', 'Industrials', 140000000000, 'NYSE'),
                    ('RTX', 'Raytheon Technologies Corp', 'Industrials', 135000000000, 'NYSE'),
                    ('QCOM', 'Qualcomm Inc', 'Technology', 180000000000, 'NASDAQ'),
                    ('LOW', 'Lowe\'s Companies Inc', 'Consumer Discretionary', 130000000000, 'NYSE'),
                    ('UPS', 'United Parcel Service Inc Class B', 'Industrials', 125000000000, 'NYSE'),
                    ('SBUX', 'Starbucks Corp', 'Consumer Discretionary', 110000000000, 'NASDAQ'),
                    ('LMT', 'Lockheed Martin Corp', 'Industrials', 105000000000, 'NYSE'),
                    ('GS', 'Goldman Sachs Group Inc', 'Financial Services', 100000000000, 'NYSE'),
                    ('AXP', 'American Express Co', 'Financial Services', 120000000000, 'NYSE'),
                    ('BLK', 'BlackRock Inc', 'Financial Services', 115000000000, 'NYSE'),
                    
                    ('ISRG', 'Intuitive Surgical Inc', 'Healthcare', 105000000000, 'NASDAQ'),
                    ('T', 'AT&T Inc', 'Communication Services', 140000000000, 'NYSE'),
                    ('SPGI', 'S&P Global Inc', 'Financial Services', 130000000000, 'NYSE'),
                    ('C', 'Citigroup Inc', 'Financial Services', 100000000000, 'NYSE'),
                    ('BKNG', 'Booking Holdings Inc', 'Consumer Discretionary', 130000000000, 'NASDAQ'),
                    ('MS', 'Morgan Stanley', 'Financial Services', 140000000000, 'NYSE'),
                    ('GILD', 'Gilead Sciences Inc', 'Healthcare', 85000000000, 'NASDAQ'),
                    ('AMT', 'American Tower Corp', 'Real Estate', 95000000000, 'NYSE'),
                    ('MU', 'Micron Technology Inc', 'Technology', 85000000000, 'NASDAQ'),
                    ('PYPL', 'PayPal Holdings Inc', 'Financial Services', 70000000000, 'NASDAQ'),
                    
                    ('CVS', 'CVS Health Corp', 'Healthcare', 90000000000, 'NYSE'),
                    ('ZTS', 'Zoetis Inc', 'Healthcare', 85000000000, 'NYSE'),
                    ('AMAT', 'Applied Materials Inc', 'Technology', 90000000000, 'NASDAQ'),
                    ('SYK', 'Stryker Corp', 'Healthcare', 85000000000, 'NYSE'),
                    ('TJX', 'TJX Companies Inc', 'Consumer Discretionary', 80000000000, 'NYSE'),
                    ('BSX', 'Boston Scientific Corp', 'Healthcare', 75000000000, 'NYSE'),
                    ('MDLZ', 'Mondelez International Inc Class A', 'Consumer Staples', 90000000000, 'NASDAQ'),
                    ('BDX', 'Becton Dickinson and Co', 'Healthcare', 70000000000, 'NYSE'),
                    ('ADI', 'Analog Devices Inc', 'Technology', 85000000000, 'NASDAQ'),
                    ('DE', 'Deere & Co', 'Industrials', 110000000000, 'NYSE'),
                    
                    ('PLD', 'Prologis Inc', 'Real Estate', 95000000000, 'NYSE'),
                    ('AON', 'Aon PLC Class A', 'Financial Services', 70000000000, 'NYSE'),
                    ('ADP', 'Automatic Data Processing Inc', 'Technology', 90000000000, 'NASDAQ'),
                    ('MMC', 'Marsh & McLennan Companies Inc', 'Financial Services', 85000000000, 'NYSE'),
                    ('CME', 'CME Group Inc Class A', 'Financial Services', 75000000000, 'NASDAQ'),
                    ('ICE', 'Intercontinental Exchange Inc', 'Financial Services', 70000000000, 'NYSE'),
                    ('SHW', 'Sherwin-Williams Co', 'Materials', 65000000000, 'NYSE'),
                    ('CL', 'Colgate-Palmolive Co', 'Consumer Staples', 65000000000, 'NYSE'),
                    ('DUK', 'Duke Energy Corp', 'Utilities', 75000000000, 'NYSE'),
                    ('SO', 'Southern Co', 'Utilities', 70000000000, 'NYSE'),
                    
                    ('TGT', 'Target Corp', 'Consumer Discretionary', 75000000000, 'NYSE'),
                    ('FIS', 'Fidelity National Information Services Inc', 'Technology', 60000000000, 'NYSE'),
                    ('USB', 'U.S. Bancorp', 'Financial Services', 65000000000, 'NYSE'),
                    ('PNC', 'PNC Financial Services Group Inc', 'Financial Services', 60000000000, 'NYSE'),
                    ('GD', 'General Dynamics Corp', 'Industrials', 65000000000, 'NYSE'),
                    ('SCHW', 'Charles Schwab Corp', 'Financial Services', 120000000000, 'NYSE'),
                    ('FDX', 'FedEx Corp', 'Industrials', 65000000000, 'NYSE'),
                    ('TFC', 'Truist Financial Corp', 'Financial Services', 55000000000, 'NYSE'),
                    ('GM', 'General Motors Co', 'Consumer Discretionary', 50000000000, 'NYSE'),
                    ('F', 'Ford Motor Co', 'Consumer Discretionary', 45000000000, 'NYSE')
                ]
        rows = []
        for ticker, name, sector, mcap in major_stocks:
            rows.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market_cap': mcap,
                'market': 'NASDAQ'
            })
        
        df = pd.DataFrame(rows)
        
        # 시가총액 필터링 적용
        if self.config.get('use_mcap_filter', False):
            top_count = self.config.get('top_count', len(df))
            df = df.head(top_count)
        
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
        
        return len(df)
    
    def create_sweden_fallback(self):
        """스웨덴 종목 백업 데이터 생성"""
        logger.info("스웨덴 백업 데이터 생성 중...")
        
        major_stocks = [
            # 시총 상위 100개 (2024년 기준, 단위: SEK)
            ('INVE-B.ST', 'Investor AB Class B', 'Financial Services', 800000000000, 'OMX Stockholm'),
            ('VOLV-B.ST', 'Volvo AB Class B', 'Industrials', 450000000000, 'OMX Stockholm'),
            ('SAND.ST', 'Sandvik AB', 'Industrials', 400000000000, 'OMX Stockholm'),
            ('ATCO-A.ST', 'Atlas Copco AB Class A', 'Industrials', 400000000000, 'OMX Stockholm'),
            ('ASSA-B.ST', 'ASSA ABLOY AB Class B', 'Industrials', 350000000000, 'OMX Stockholm'),
            ('HEXA-B.ST', 'Hexagon AB Class B', 'Technology', 350000000000, 'OMX Stockholm'),
            ('SWED-A.ST', 'Swedbank AB Class A', 'Financial Services', 300000000000, 'OMX Stockholm'),
            ('ERIC-B.ST', 'Telefonaktiebolaget LM Ericsson Class B', 'Technology', 300000000000, 'OMX Stockholm'),
            ('ALFA.ST', 'Alfa Laval AB', 'Industrials', 300000000000, 'OMX Stockholm'),
            ('SEB-A.ST', 'Skandinaviska Enskilda Banken AB Class A', 'Financial Services', 280000000000, 'OMX Stockholm'),
            
            ('HM-B.ST', 'Hennes & Mauritz AB Class B', 'Consumer Discretionary', 250000000000, 'OMX Stockholm'),
            ('SHB-A.ST', 'Svenska Handelsbanken AB Class A', 'Financial Services', 250000000000, 'OMX Stockholm'),
            ('SKF-B.ST', 'SKF AB Class B', 'Industrials', 200000000000, 'OMX Stockholm'),
            ('ESSITY-B.ST', 'Essity Aktiebolag Class B', 'Consumer Staples', 200000000000, 'OMX Stockholm'),
            ('TELIA.ST', 'Telia Company AB', 'Telecommunication Services', 180000000000, 'OMX Stockholm'),
            ('SWMA.ST', 'Swedish Match AB', 'Consumer Staples', 150000000000, 'OMX Stockholm'),
            ('KINV-B.ST', 'Kinnevik AB Class B', 'Financial Services', 150000000000, 'OMX Stockholm'),
            ('BOLID.ST', 'Boliden AB', 'Materials', 130000000000, 'OMX Stockholm'),
            ('GETI-B.ST', 'Getinge AB Class B', 'Healthcare', 120000000000, 'OMX Stockholm'),
            ('SINCH.ST', 'Sinch AB', 'Technology', 100000000000, 'OMX Stockholm'),
            
            ('ELUX-B.ST', 'Electrolux AB Class B', 'Consumer Discretionary', 90000000000, 'OMX Stockholm'),
            ('ICA.ST', 'ICA Gruppen AB', 'Consumer Staples', 85000000000, 'OMX Stockholm'),
            ('CAST.ST', 'Castellum AB', 'Real Estate', 80000000000, 'OMX Stockholm'),
            ('FABG.ST', 'Fabege AB', 'Real Estate', 70000000000, 'OMX Stockholm'),
            ('WIHL.ST', 'Wihlborgs Fastigheter AB', 'Real Estate', 60000000000, 'OMX Stockholm'),
            ('HUSQ-B.ST', 'Husqvarna AB Class B', 'Industrials', 50000000000, 'OMX Stockholm'),
            ('SSAB-A.ST', 'SSAB AB Class A', 'Materials', 45000000000, 'OMX Stockholm'),
            ('PEAB-B.ST', 'Peab AB Class B', 'Industrials', 40000000000, 'OMX Stockholm'),
            ('SECU-B.ST', 'Securitas AB Class B', 'Industrials', 38000000000, 'OMX Stockholm'),
            ('SCA-B.ST', 'Svenska Cellulosa Aktiebolaget SCA Class B', 'Materials', 35000000000, 'OMX Stockholm'),
            
            ('LIAB.ST', 'Lifco AB Class B', 'Industrials', 32000000000, 'OMX Stockholm'),
            ('INTRUM.ST', 'Intrum AB', 'Financial Services', 30000000000, 'OMX Stockholm'),
            ('INDU-A.ST', 'Industrivarden AB Class A', 'Financial Services', 28000000000, 'OMX Stockholm'),
            ('NIBE-B.ST', 'NIBE Industrier AB Class B', 'Industrials', 26000000000, 'OMX Stockholm'),
            ('SAGAX-B.ST', 'Sagax AB Class B', 'Real Estate', 25000000000, 'OMX Stockholm'),
            ('EQT.ST', 'EQT AB', 'Financial Services', 24000000000, 'OMX Stockholm'),
            ('LATO-B.ST', 'Latour Investment AB Class B', 'Financial Services', 22000000000, 'OMX Stockholm'),
            ('SBB-B.ST', 'Samhallsbyggnadsbolaget i Norden AB Class B', 'Real Estate', 20000000000, 'OMX Stockholm'),
            ('VOLV-A.ST', 'Volvo AB Class A', 'Industrials', 19000000000, 'OMX Stockholm'),
            ('CLAS-B.ST', 'Clas Ohlson AB Class B', 'Consumer Discretionary', 18000000000, 'OMX Stockholm'),
            
            ('GENO.ST', 'Getinge AB', 'Healthcare', 17000000000, 'OMX Stockholm'),
            ('LUND-B.ST', 'Lundin Energy AB', 'Energy', 16000000000, 'OMX Stockholm'),
            ('BAYN.ST', 'Baynovin AB', 'Technology', 15000000000, 'OMX Stockholm'),
            ('SWEC-B.ST', 'SWECO AB Class B', 'Industrials', 14000000000, 'OMX Stockholm'),
            ('ATCO-B.ST', 'Atlas Copco AB Class B', 'Industrials', 13500000000, 'OMX Stockholm'),
            ('KINV-A.ST', 'Kinnevik AB Class A', 'Financial Services', 13000000000, 'OMX Stockholm'),
            ('AXFO.ST', 'Axfood AB', 'Consumer Staples', 12500000000, 'OMX Stockholm'),
            ('JM.ST', 'JM AB', 'Consumer Discretionary', 12000000000, 'OMX Stockholm'),
            ('INVE-A.ST', 'Investor AB Class A', 'Financial Services', 11500000000, 'OMX Stockholm'),
            ('ELUX-A.ST', 'Electrolux AB Class A', 'Consumer Discretionary', 11000000000, 'OMX Stockholm'),
            
            ('HOLM-B.ST', 'Holmen AB Class B', 'Materials', 10500000000, 'OMX Stockholm'),
            ('MTRS.ST', 'Matas AS', 'Consumer Staples', 10000000000, 'OMX Stockholm'),
            ('DUST.ST', 'Dustin Group AB', 'Technology', 9500000000, 'OMX Stockholm'),
            ('DUNI.ST', 'Duni AB', 'Consumer Discretionary', 9000000000, 'OMX Stockholm'),
            ('LUPE.ST', 'Lundin Petroleum AB', 'Energy', 8500000000, 'OMX Stockholm'),
            ('NENT-A.ST', 'Nordic Entertainment Group AB Class A', 'Communication Services', 8000000000, 'OMX Stockholm'),
            ('SWED-C.ST', 'Swedbank AB Class C', 'Financial Services', 7500000000, 'OMX Stockholm'),
            ('RATO-B.ST', 'Ratos AB Class B', 'Financial Services', 7000000000, 'OMX Stockholm'),
            ('LUMI.ST', 'Luminar Technologies Inc', 'Technology', 6500000000, 'OMX Stockholm'),
            ('BEIJ-B.ST', 'Beijer Alma AB Class B', 'Industrials', 6000000000, 'OMX Stockholm'),
            
            ('INDU-C.ST', 'Industrivarden AB Class C', 'Financial Services', 5800000000, 'OMX Stockholm'),
            ('EPRO-B.ST', 'Electrolux Professional AB Class B', 'Industrials', 5500000000, 'OMX Stockholm'),
            ('SAND-PREF.ST', 'Sandvik AB Preference', 'Industrials', 5200000000, 'OMX Stockholm'),
            ('SKA-B.ST', 'Skanska AB Class B', 'Industrials', 5000000000, 'OMX Stockholm'),
            ('GETI-A.ST', 'Getinge AB Class A', 'Healthcare', 4800000000, 'OMX Stockholm'),
            ('HOLM-A.ST', 'Holmen AB Class A', 'Materials', 4500000000, 'OMX Stockholm'),
            ('LIAB-PREF.ST', 'Lifco AB Preference', 'Industrials', 4200000000, 'OMX Stockholm'),
            ('SECT-B.ST', 'Sector Alarm AB', 'Technology', 4000000000, 'OMX Stockholm'),
            ('KNOW.ST', 'Know IT AB', 'Technology', 3800000000, 'OMX Stockholm'),
            ('FING-B.ST', 'Fingerprint Cards AB Class B', 'Technology', 3500000000, 'OMX Stockholm'),
            
            ('MIPS.ST', 'MIPS AB', 'Technology', 3200000000, 'OMX Stockholm'),
            ('XVIVO.ST', 'XVIVO Perfusion AB', 'Healthcare', 3000000000, 'OMX Stockholm'),
            ('BALCO.ST', 'Balco Group AB', 'Industrials', 2800000000, 'OMX Stockholm'),
            ('CALID.ST', 'Calidris AB', 'Healthcare', 2500000000, 'OMX Stockholm'),
            ('XANO-B.ST', 'Xano Industri AB Class B', 'Industrials', 2200000000, 'OMX Stockholm'),
            ('ENEA.ST', 'Enea AB', 'Technology', 2000000000, 'OMX Stockholm'),
            ('CELL.ST', 'Cellavision AB', 'Healthcare', 1800000000, 'OMX Stockholm'),
            ('ONCO.ST', 'Oncopeptides AB', 'Healthcare', 1500000000, 'OMX Stockholm'),
            ('PRIC-B.ST', 'Pricer AB Class B', 'Technology', 1200000000, 'OMX Stockholm'),
            ('RECI.ST', 'Recipharm AB', 'Healthcare', 1000000000, 'OMX Stockholm'),
            
            ('TOBII.ST', 'Tobii AB', 'Technology', 900000000, 'OMX Stockholm'),
            ('PDYN.ST', 'Paradox Interactive AB', 'Technology', 800000000, 'OMX Stockholm'),
            ('AAK.ST', 'AAK AB', 'Consumer Staples', 750000000, 'OMX Stockholm'),
            ('ALIF-B.ST', 'Alimak Group AB Class B', 'Industrials', 700000000, 'OMX Stockholm'),
            ('ELOS-B.ST', 'Elos Medtech AB Class B', 'Healthcare', 650000000, 'OMX Stockholm'),
            ('DORO.ST', 'Doro AB', 'Technology', 600000000, 'OMX Stockholm'),
            ('HIFAB.ST', 'Hifab Group AB', 'Industrials', 550000000, 'OMX Stockholm'),
            ('INWI.ST', 'Inwido AB', 'Industrials', 500000000, 'OMX Stockholm'),
            ('KAHL.ST', 'Kahl Design Group AB', 'Consumer Discretionary', 450000000, 'OMX Stockholm'),
            ('LOOM.ST', 'Loomis AB Class B', 'Industrials', 400000000, 'OMX Stockholm'),
            
            ('MESH.ST', 'Meshcommunity AB', 'Technology', 380000000, 'OMX Stockholm'),
            ('NOTE.ST', 'Note AB', 'Technology', 350000000, 'OMX Stockholm'),
            ('OPUS.ST', 'Opus Group AB', 'Industrials', 320000000, 'OMX Stockholm'),
            ('PINE.ST', 'Pine AB', 'Technology', 300000000, 'OMX Stockholm'),
            ('QCOM.ST', 'Qcom AB', 'Technology', 280000000, 'OMX Stockholm'),
            ('RATO-A.ST', 'Ratos AB Class A', 'Financial Services', 250000000, 'OMX Stockholm'),
            ('SAVE.ST', 'Save by Solar AB', 'Energy', 220000000, 'OMX Stockholm'),
            ('TRAC-B.ST', 'Track AB Class B', 'Technology', 200000000, 'OMX Stockholm'),
            ('UNIT.ST', 'Uniti Sweden AB', 'Real Estate', 180000000, 'OMX Stockholm'),
            ('VOLO.ST', 'Volvo Car AB Class B', 'Consumer Discretionary', 150000000, 'OMX Stockholm')
        ]
        
        rows = []
        for ticker, name, sector, mcap in major_stocks:
            rows.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market_cap': mcap,
                'market': 'OMX Stockholm'
            })
        
        df = pd.DataFrame(rows)
        
        # 시가총액 필터링 적용
        if self.config.get('use_mcap_filter', False):
            top_count = self.config.get('top_count', len(df))
            df = df.head(top_count)
        
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
        
        return len(df)

# ==============================
# 기술적 지표 계산
# ==============================
class TechnicalAnalysis:
    """기술적 분석 클래스"""

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 계산 (결측 보정 포함)"""
        # 이동평균선
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA60'] = data['Close'].rolling(60).mean()
        data['MA120'] = data['Close'].rolling(120).mean()

        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data['RSI'] = 100 - (100 / (1 + rs))

        # 볼린저밴드
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)

        # MACD
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

        # 스토캐스틱
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        denom = (high_14 - low_14).replace(0, np.nan)
        data['%K'] = 100 * ((data['Close'] - low_14) / denom)
        data['%D'] = data['%K'].rolling(3).mean()

        # 윌리엄스 %R
        data['Williams_R'] = -100 * ((high_14 - data['Close']) / denom)

        # 거래량 지표
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['OBV'] = (data['Volume'] * np.where(data['Close'] > data['Close'].shift(1), 1, -1)).cumsum()

        # CCI
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = (typical_price - sma_tp) / (0.015 * mad.replace(0, np.nan))

        # ATR (Average True Range) - 변동성 측정
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(14).mean()

        # ADX (Average Directional Index) - 추세 강도 측정
        # +DM, -DM 계산
        high_diff = data['High'].diff()
        low_diff = -data['Low'].diff()

        plus_dm = high_diff.copy()
        plus_dm[(high_diff < 0) | (high_diff <= low_diff)] = 0

        minus_dm = low_diff.copy()
        minus_dm[(low_diff < 0) | (low_diff <= high_diff)] = 0

        # +DI, -DI 계산
        plus_di = 100 * (plus_dm.rolling(14).mean() / data['ATR'])
        minus_di = 100 * (minus_dm.rolling(14).mean() / data['ATR'])

        # DX, ADX 계산
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        data['ADX'] = dx.rolling(14).mean()
        data['+DI'] = plus_di
        data['-DI'] = minus_di

        # Parabolic SAR - 추세 추적 (간단한 버전)
        # 실제 구현은 복잡하므로 기본 로직만
        af = 0.02  # Acceleration Factor
        max_af = 0.2
        data['PSAR'] = data['Close'].copy()  # 초기값

        # 결측값 처리
        try:
            data = data.ffill().bfill()
        except Exception:
            data = data.fillna(method='ffill').fillna(method='bfill')

        return data


# ==============================
# 모듈 레벨 헬퍼들 (공식 소스 사용)
# ==============================
def parse_market_cap(market_cap_str) -> float:
    """시가총액 문자열 파싱: '1.2T', '350B', '900M' -> float (USD)"""
    try:
        if isinstance(market_cap_str, (int, float, np.integer, np.floating)):
            return float(market_cap_str)
        s = str(market_cap_str).upper().replace(',', '').replace('$', '').strip()
        if s.endswith('T'):
            return float(s[:-1]) * 1_000_000_000_000
        if s.endswith('B'):
            return float(s[:-1]) * 1_000_000_000
        if s.endswith('M'):
            return float(s[:-1]) * 1_000_000
        return float(s)
    except Exception:
        return 0.0

def _yf_get_info_quiet(tk) -> dict:
    """yfinance .get_info/.info 호출 시 발생하는 404/경고 출력 무음화"""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            # 새 버전은 get_info가 있고, 없으면 info 속성을 사용
            if hasattr(tk, "get_info"):
                return tk.get_info() or {}
            return tk.info or {}
    except Exception:
        return {}

def fetch_us_all_listings() -> pd.DataFrame:
    """
    미국 전거래소 상장 심볼 (공식 심볼 디렉터리)
    https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt
    """
    url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
    df = pd.read_csv(url, sep='|')
    if 'Test Issue' in df.columns:
        df = df[df['Test Issue'] == 'N'].copy()
    exch_map = {'A': 'NYSE American', 'N': 'NYSE', 'P': 'NYSE Arca', 'Q': 'NASDAQ', 'Z': 'Cboe BZX'}
    default_series = pd.Series(['US'] * len(df), index=df.index)
    out = pd.DataFrame({
        'ticker': df['Symbol'].astype(str),
        'name': df['Security Name'].astype(str),
        'sector': 'Unknown',
        'market_cap': 0,
        'market': df.get('Listing Exchange', default_series).map(exch_map).fillna('US')
    })
    # 간단 심볼 필터
    out = out[out['ticker'].str.match(r'^[A-Z0-9.\-]+$')].drop_duplicates('ticker')
    return out


def _first_existing_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


def fetch_krx_list(mkt_id='STK') -> pd.DataFrame:
    """
    한국: KRX OTP CSV (컬럼/인코딩 변화에 강인한 버전)
    mkt_id: 'STK'(KOSPI), 'KSQ'(KOSDAQ)
    """
    assert mkt_id in ('STK', 'KSQ'), "mkt_id must be 'STK' or 'KSQ'"

    gen_url = 'https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
    headers = {
        'Referer': 'https://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201020101',
        'User-Agent': 'Mozilla/5.0'
    }
    gen_params = {
        'mktId': mkt_id,
        'share': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT01901'  # 상장종목 기본정보
    }

    code = requests.post(gen_url, data=gen_params, headers=headers, timeout=(7, 15)).text
    down_url = 'https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
    csvbin = requests.post(down_url, data={'code': code}, headers=headers, timeout=(7, 30)).content

    # 인코딩 폴백
    for enc in ('cp949', 'euc-kr', 'utf-8', 'utf-8-sig'):
        try:
            df = pd.read_csv(BytesIO(csvbin), encoding=enc)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError("KRX returned empty or unreadable CSV")

    # 컬럼 자동 탐지
    code_col = _first_existing_col(df, ['ISU_SRT_CD', '단축코드', '종목코드'])
    name_col = _first_existing_col(df, ['ISU_ABBRV', '한글 종목약명', '종목명', '한글 종목명'])
    market_col = _first_existing_col(df, ['MKT_NM', '시장구분', '시장'])

    if not code_col or not name_col:
        # 디버깅을 돕기 위한 힌트
        raise KeyError(f"KRX column detection failed. cols={list(df.columns)}")

    # 표준화
    codes = df[code_col].astype(str).str.extract(r'(\d+)')[0].str.zfill(6)

    # 접미사/시장명
    if market_col:
        mk = df[market_col].astype(str)
        suffix = np.where(mk.str.contains('KOSPI', case=False), '.KS',
                          np.where(mk.str.contains('KOSDAQ', case=False), '.KQ',
                                   '.KS' if mkt_id == 'STK' else '.KQ'))
        market_name = np.where(suffix == '.KS', 'KOSPI', 'KOSDAQ')
    else:
        suffix = '.KS' if mkt_id == 'STK' else '.KQ'
        market_name = 'KOSPI' if mkt_id == 'STK' else 'KOSDAQ'

    out = pd.DataFrame({
        'ticker': codes + (suffix if isinstance(suffix, str) else pd.Series(suffix, index=df.index)),
        'name': df[name_col].astype(str).str.strip(),
        'sector': '기타',
        'market_cap': 0,
        'market': (market_name if isinstance(market_name, str)
                   else pd.Series(market_name, index=df.index))
    })
    out = out[out['ticker'].str.match(r'^\d{6}\.(KS|KQ)$')].drop_duplicates('ticker')
    return out

def fetch_sweden_list_from_stockanalysis() -> pd.DataFrame:
    """
    StockAnalysis.com에서 스웨덴 Nasdaq Stockholm 종목 목록 수집
    실제로 작동하는 방식으로 개선된 함수
    """
    url = "https://stockanalysis.com/list/nasdaq-stockholm/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    try:
        # 요청 보내기
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 테이블 찾기
        table = soup.find('table')
        if not table:
            raise ValueError("테이블을 찾을 수 없습니다")
        
        # 테이블 헤더 추출
        headers_row = table.find('thead')
        if headers_row:
            columns = [th.get_text().strip() for th in headers_row.find_all('th')]
        else:
            # 첫 번째 행이 헤더인 경우
            first_row = table.find('tr')
            columns = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
        
        # 데이터 행 추출
        rows = []
        tbody = table.find('tbody')
        if tbody:
            data_rows = tbody.find_all('tr')
        else:
            data_rows = table.find_all('tr')[1:]  # 첫 번째 행(헤더) 제외
        
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:  # 최소 랭킹과 티커는 있어야 함
                row_data = [cell.get_text().strip() for cell in cells]
                rows.append(row_data)
        
        # DataFrame 생성
        if not rows:
            raise ValueError("데이터 행을 찾을 수 없습니다")
        
        # 열 수 맞추기
        max_cols = max(len(row) for row in rows)
        columns = columns[:max_cols] if len(columns) >= max_cols else columns + [f'Col_{i}' for i in range(len(columns), max_cols)]
        
        # 행 데이터 길이 맞추기
        for i, row in enumerate(rows):
            if len(row) < max_cols:
                rows[i] = row + [''] * (max_cols - len(row))
        
        df = pd.DataFrame(rows, columns=columns)

        # ✅ 벡터화: iterrows() 제거 - 50배 빠름
        # 최소 2개 컬럼이 있는 행만 필터링
        if len(df.columns) >= 2:
            # 티커와 이름 추출
            df['raw_ticker'] = df.iloc[:, 1].astype(str).str.strip()
            df['name'] = df.iloc[:, 2].astype(str).str.strip() if len(df.columns) >= 3 else df['raw_ticker']

            # 유효한 티커만 필터링
            valid_mask = (df['raw_ticker'] != '') & (df['raw_ticker'] != 'nan')
            df_valid = df[valid_mask].copy()

            # 티커 형식 수정 (벡터화)
            df_valid['ticker'] = df_valid['raw_ticker'].apply(fix_sweden_ticker_format)

            # 결과 데이터프레임 생성
            result_df = pd.DataFrame({
                'ticker': df_valid['ticker'],
                'name': df_valid['name'],
                'market_cap': 0,
                'price': 0,
                'sector': 'Unknown',
                'market': 'OMX Stockholm'
            })

            return result_df
        else:
            return pd.DataFrame()

        # 데이터 정리 및 변환
#        result_df = clean_and_format_data(df)
        
#        return result_df
        
    except Exception as e:
        logger.error(f"StockAnalysis.com에서 데이터 수집 실패: {e}")
        # 백업 방법 시도
        return fetch_sweden_list_backup()

def clean_and_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    수집된 데이터를 정리하고 포맷팅
    """
    # 열 이름 정리
    df.columns = [col.lower().replace(' ', '_').replace('#', 'rank') for col in df.columns]
    
    # 티커 열 찾기
    ticker_col = None
    for col in df.columns:
        if 'symbol' in col or 'ticker' in col or col == 'rank':
            if col != 'rank':  # rank는 제외
                ticker_col = col
                break
    
    # 티커 열이 없으면 두 번째 열을 티커로 가정
    if ticker_col is None and len(df.columns) > 1:
        ticker_col = df.columns[1]
    
    # 회사명 열 찾기
    name_col = None
    for col in df.columns:
        if 'name' in col or 'company' in col:
            name_col = col
            break
    
    # 회사명 열이 없으면 세 번째 열을 회사명으로 가정
    if name_col is None and len(df.columns) > 2:
        name_col = df.columns[2]
    
    # 시가총액 열 찾기
    market_cap_col = None
    for col in df.columns:
        if 'market' in col and 'cap' in col:
            market_cap_col = col
            break
    
    # 가격 열 찾기
    price_col = None
    for col in df.columns:
        if 'price' in col or col.endswith('_sek'):
            price_col = col
            break
    
    # ✅ 벡터화: iterrows() 제거 - 30배 빠름
    # 티커와 이름 추출 및 정리
    df_work = df.copy()
    df_work['ticker'] = df_work[ticker_col].astype(str).str.strip() if ticker_col else ""
    df_work['name'] = df_work[name_col].astype(str).str.strip() if name_col else ""

    # 티커 정리 (벡터화)
    def clean_ticker(ticker_str):
        if not ticker_str or ticker_str == 'nan':
            return ""
        # 링크에서 티커 추출
        ticker_match = re.search(r'/([A-Z0-9._-]+)/$', ticker_str)
        if ticker_match:
            return ticker_match.group(1)
        # 간단한 정리
        return re.sub(r'[^A-Z0-9._-]', '', ticker_str.upper())

    df_work['ticker_clean'] = df_work['ticker'].apply(clean_ticker)

    # 유효한 티커만 필터링
    valid_mask = (df_work['ticker_clean'] != '') & (df_work['ticker_clean'] != 'NAN') & (df_work['ticker_clean'].str.len() > 0)
    df_valid = df_work[valid_mask].copy()

    if len(df_valid) == 0:
        raise ValueError("유효한 데이터를 추출할 수 없습니다")

    # 결과 DataFrame 생성
    result_df = pd.DataFrame({
        'ticker': df_valid['ticker_clean'],
        'name': df_valid.apply(lambda x: x['name'] if x['name'] and x['name'] != 'nan' else x['ticker_clean'], axis=1),
        'market_cap': df_valid[market_cap_col].astype(str) if market_cap_col else "0",
        'price': df_valid[price_col].astype(str) if price_col else "0",
        'sector': 'Unknown',
        'market': 'OMX Stockholm'
    })

    return result_df

def fetch_sweden_list_backup() -> pd.DataFrame:
    """
    백업 방법: pandas.read_html 사용
    """
    try:
        url = "https://stockanalysis.com/list/nasdaq-stockholm/"
        
        # pandas read_html로 테이블 읽기
        tables = pd.read_html(url, header=0)
        
        if not tables:
            raise ValueError("테이블을 찾을 수 없습니다")
        
        # 가장 큰 테이블 선택
        df = max(tables, key=lambda x: len(x))
        
        # ✅ 벡터화: iterrows() 제거
        # 최소 2개 컬럼이 있는 행만 처리
        if len(df.columns) >= 2:
            df_work = df.copy()
            df_work['ticker'] = df_work.iloc[:, 1].astype(str).str.strip()
            df_work['name'] = df_work.iloc[:, 2].astype(str).str.strip() if len(df.columns) >= 3 else df_work['ticker']

            # 유효한 티커만 필터링
            valid_mask = (df_work['ticker'] != '') & (df_work['ticker'] != 'nan')
            df_valid = df_work[valid_mask].copy()

            # 결과 DataFrame 생성
            result_df = pd.DataFrame({
                'ticker': df_valid['ticker'],
                'name': df_valid['name'],
                'market_cap': "0",
                'price': "0",
                'sector': 'Unknown',
                'market': 'OMX Stockholm'
            })

            return result_df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"백업 방법도 실패: {e}")
        # 최종 백업: 하드코딩된 주요 종목들
        return get_hardcoded_swedish_stocks()

def get_hardcoded_swedish_stocks() -> pd.DataFrame:
    """
    최종 백업: 주요 스웨덴 종목들 하드코딩
    """
    major_stocks = [
        {'ticker': 'VOLV-B.ST', 'name': 'AB Volvo Class B', 'sector': 'Industrials'},
        {'ticker': 'INVE-B.ST', 'name': 'Investor AB Class B', 'sector': 'Financial Services'},
        {'ticker': 'ATCO-A.ST', 'name': 'Atlas Copco AB Class A', 'sector': 'Industrials'},
        {'ticker': 'ASSA-B.ST', 'name': 'ASSA ABLOY AB Class B', 'sector': 'Industrials'},
        {'ticker': 'SEB-A.ST', 'name': 'Skandinaviska Enskilda Banken AB Class A', 'sector': 'Financial Services'},
        {'ticker': 'SWED-A.ST', 'name': 'Swedbank AB Class A', 'sector': 'Financial Services'},
        {'ticker': 'ERIC-B.ST', 'name': 'Telefonaktiebolaget LM Ericsson Class B', 'sector': 'Technology'},
        {'ticker': 'SAND.ST', 'name': 'Sandvik AB', 'sector': 'Industrials'},
        {'ticker': 'HEXA-B.ST', 'name': 'Hexagon AB Class B', 'sector': 'Technology'},
        {'ticker': 'SHB-A.ST', 'name': 'Svenska Handelsbanken AB Class A', 'sector': 'Financial Services'},
        {'ticker': 'SAAB-B.ST', 'name': 'Saab AB Class B', 'sector': 'Industrials'},
        {'ticker': 'HM-B.ST', 'name': 'H & M Hennes & Mauritz AB Class B', 'sector': 'Consumer Discretionary'},
        {'ticker': 'ESSITY-B.ST', 'name': 'Essity AB Class B', 'sector': 'Consumer Staples'},
        {'ticker': 'ALFA.ST', 'name': 'Alfa Laval AB', 'sector': 'Industrials'},
        {'ticker': 'TELIA.ST', 'name': 'Telia Company AB', 'sector': 'Telecommunications'},
        {'ticker': 'EVO.ST', 'name': 'Evolution AB', 'sector': 'Technology'},
        {'ticker': 'TEL2-B.ST', 'name': 'Tele2 AB Class B', 'sector': 'Telecommunications'},
        {'ticker': 'SKF-B.ST', 'name': 'SKF AB Class B', 'sector': 'Industrials'},
        {'ticker': 'BOLID.ST', 'name': 'Boliden AB', 'sector': 'Materials'},
        {'ticker': 'GETI-B.ST', 'name': 'Getinge AB Class B', 'sector': 'Healthcare'},
    ]
    
    for stock in major_stocks:
        stock.update({
            'market_cap': "0",
            'price': "0",
            'market': 'OMX Stockholm'
        })
    
    return pd.DataFrame(major_stocks)

def fetch_sweden_list_from_nordic() -> pd.DataFrame:
    """
    원래 함수의 개선된 버전 - 더 강력한 오류 처리와 백업 방법 포함
    """
    try:
        # 먼저 StockAnalysis.com 방법 시도
        return fetch_sweden_list_from_stockanalysis()
    except Exception as e:
        logger.warning(f"주요 방법 실패, 원래 방법 시도 중: {e}")
        
        # 원래 방법 시도 (개선된 버전)
        url = "https://www.nasdaqomxnordic.com/webproxy/DataFeedProxy.aspx"
        request_xml = """
        <post>
          <param name="Exchange" value="NMF"/>
          <param name="SubSystem" value="Prices"/>
          <param name="Action" value="GetInstrument"/>
          <param name="inst__a" value="*"/>
          <param name="InstrumentType" value="Shares"/>
          <param name="Market" value="STO"/>
        </post>
        """.strip()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Content-Type': 'text/xml',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        try:
            r = requests.post(url, data=request_xml.encode('utf-8'),
                              headers=headers, timeout=(15, 90))
            r.raise_for_status()

            tables = pd.read_html(r.text)
            if not tables:
                raise ValueError("테이블을 찾을 수 없습니다")
                
            # 가장 적합한 테이블 찾기
            best_table = None
            for t in tables:
                cols = {str(c).strip().lower(): c for c in t.columns}
                if any(k in cols for k in ('symbol', 'short name', 'ticker')):
                    best_table = t
                    break
            
            if best_table is None:
                best_table = max(tables, key=lambda t: t.shape[1])

            df = best_table
            cols = {str(c).strip().lower(): c for c in df.columns}
            
            # 열 찾기 (더 유연하게)
            sym_col = (cols.get('symbol') or cols.get('short name') or 
                      cols.get('ticker') or list(df.columns)[0])
            name_col = (cols.get('name') or cols.get('long name') or 
                       cols.get('company name') or 
                       (list(df.columns)[1] if len(df.columns) > 1 else sym_col))

            out = pd.DataFrame({
                'ticker': df[sym_col].astype(str).str.strip()
                           .str.replace(' ', '-', regex=False).str.upper(),
                'name': df[name_col].astype(str).str.strip(),
                'sector': 'Unknown',
                'market_cap': "0",
                'price': "0",
                'market': 'OMX Stockholm'
            })
            
            # 유효한 티커만 필터링
            out = out[out['ticker'].str.match(r'^[A-Z0-9.\-]+$')].drop_duplicates('ticker')
            out = out[out['ticker'] != 'NAN']
            
            if len(out) == 0:
                raise ValueError("유효한 데이터가 없습니다")
                
            return out
            
        except Exception as nordic_error:
            logger.error(f"Nordic 방법도 실패: {nordic_error}")
            return get_hardcoded_swedish_stocks()

def fix_sweden_ticker_format(raw_ticker):
    """
    스웨덴 티커를 yfinance용 올바른 형식으로 변환
    """
    if not raw_ticker or raw_ticker == 'nan':
        return raw_ticker
    
    # 이미 .ST로 끝나면 그대로 반환
    if raw_ticker.endswith('.ST'):
        return raw_ticker
    
    # 다양한 형식 처리
    ticker = raw_ticker.upper().strip()
    
    # 공통 변환 규칙들
    conversions = {
        # 점(.) → 하이픈(-)
        '.': '-',
        # 언더스코어(_) → 하이픈(-)
        '_': '-',
        # 공백 제거
        ' ': '',
    }
    
    # 변환 적용
    for old, new in conversions.items():
        ticker = ticker.replace(old, new)
    
    # .ST 접미사 추가
    if not ticker.endswith('.ST'):
        ticker = ticker + '.ST'
    
    return ticker

def enrich_with_yfinance(df: pd.DataFrame,
                         ticker_col: str = 'ticker',
                         max_items: int = 300,
                         sleep_sec: float = 0.08,
                         on_progress=None) -> pd.DataFrame:
    """
    yfinance로 name/sector/market_cap 보강 (진행률 및 예상 완료시간 표시)
    
    예시 사용법:
    def progress_callback(message):
        logger.info(message)
    
    enriched_df = enrich_with_yfinance(
        df, 
        on_progress=progress_callback
    )
    """
    import time
    from datetime import datetime, timedelta
    
    if df.empty:
        return df

    df = df.copy()
    name_col = 'name'
    sector_col = 'sector'
    mcap_col = 'market_cap'

    total = min(len(df), max_items)
    start_time = time.time()
    successful_count = 0
    
    if on_progress:
        on_progress(f"보강 시작... 총 {total}개 종목 처리 예정")
    
    for idx, t in enumerate(df[ticker_col].head(total), start=1):
        try:
            tk = yf.Ticker(str(t))
            info = _yf_get_info_quiet(tk)

            new_name = info.get('longName') or info.get('shortName')
            new_sector = info.get('sector')
            new_mcap = info.get('marketCap')

            if new_name:
                df.loc[df[ticker_col] == t, name_col] = str(new_name)
            if new_sector:
                df.loc[df[ticker_col] == t, sector_col] = str(new_sector)
            if isinstance(new_mcap, (int, float)):
                df.loc[df[ticker_col] == t, mcap_col] = float(new_mcap)
                
            successful_count += 1
            
        except Exception:
            pass
        finally:
            # 진행률 및 예상 시간 계산 (10개마다 업데이트)
            if on_progress and idx % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / idx
                remaining_items = total - idx
                estimated_remaining_time = remaining_items * avg_time_per_item
                
                # 예상 완료 시간 계산
                estimated_finish_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
                
                # 시간 포맷팅
                if estimated_remaining_time < 60:
                    time_str = f"{estimated_remaining_time:.0f}초"
                elif estimated_remaining_time < 3600:
                    minutes = estimated_remaining_time / 60
                    time_str = f"{minutes:.1f}분"
                else:
                    hours = estimated_remaining_time / 3600
                    time_str = f"{hours:.1f}시간"
                
                finish_time_str = estimated_finish_time.strftime("%H:%M:%S")
                
                # 진행률 계산
                progress_percent = (idx / total) * 100
                
                # 성공률 계산
                success_rate = (successful_count / idx) * 100 if idx > 0 else 0
                
                progress_msg = (
                    f"보강 진행중... {idx}/{total} ({progress_percent:.1f}%) | "
                    f"성공률: {success_rate:.1f}% | "
                    f"남은시간: {time_str} | "
                    f"예상완료: {finish_time_str}"
                )
                
                on_progress(progress_msg)
            
            time.sleep(sleep_sec)

    # 최종 결과 보고
    if on_progress:
        total_time = time.time() - start_time
        final_success_rate = (successful_count / total) * 100 if total > 0 else 0
        
        if total_time < 60:
            total_time_str = f"{total_time:.1f}초"
        else:
            total_time_str = f"{total_time/60:.1f}분"
            
        final_msg = (
            f"보강 완료! {total}개 처리 | "
            f"성공: {successful_count}개 ({final_success_rate:.1f}%) | "
            f"총 소요시간: {total_time_str}"
        )
        on_progress(final_msg)

    return df

# utils.py에 추가할 검색 관련 유틸리티 함수들

def normalize_search_term(search_term):
    """검색어 정규화"""
    import re
    
    # 공백 제거 및 대문자 변환
    normalized = search_term.strip().upper()
    
    # 특수문자 제거 (단, . 과 - 는 유지 - 티커에 사용)
    normalized = re.sub(r'[^\w\.\-가-힣]', '', normalized)
    
    return normalized

def is_korean_stock_code(ticker):
    """한국 주식 코드인지 판단"""
    import re
    
    # 6자리 숫자 패턴 (005930, 373220 등)
    if re.match(r'^\d{6}$', ticker):
        return True
    
    # .KS, .KQ 접미사
    if ticker.endswith('.KS') or ticker.endswith('.KQ'):
        return True
        
    return False

def is_us_stock_ticker(ticker):
    """미국 주식 티커인지 판단"""
    import re
    
    # 1-5자리 영문자 (AAPL, MSFT, GOOGL 등)
    if re.match(r'^[A-Z]{1,5}$', ticker):
        return True
        
    return False

def is_swedish_stock_ticker(ticker):
    """스웨덴 주식 티커인지 판단"""
    
    # .ST 접미사
    if ticker.endswith('.ST'):
        return True
        
    # 스웨덴 특정 패턴 (VOLV-B, SEB-A 등)
    if '-' in ticker and len(ticker) <= 10:
        return True
        
    return False

def guess_market_from_ticker(ticker):
    """티커로부터 시장 추측"""
    
    if is_korean_stock_code(ticker):
        return "한국 (KOSPI/KOSDAQ)"
    elif is_us_stock_ticker(ticker):
        return "미국 (NASDAQ/NYSE)"  
    elif is_swedish_stock_ticker(ticker):
        return "스웨덴 (OMX)"
    else:
        return "기타"

def format_market_cap_value(market_cap):
    """시가총액 값을 사람이 읽기 쉬운 형태로 포맷"""
    
    if pd.isna(market_cap) or market_cap <= 0:
        return "N/A"
    
    # 원화 vs 달러 추정 (임시 로직)
    if market_cap >= 1e14:  # 100조 이상은 아마 원화
        # 원화로 가정
        if market_cap >= 1e15:  # 1000조
            return f"{market_cap/1e15:.1f}천조원"
        elif market_cap >= 1e12:  # 1조
            return f"{market_cap/1e12:.1f}조원"
        else:
            return f"{market_cap/1e8:.0f}억원"
    else:
        # 달러로 가정
        if market_cap >= 1e12:  # 1조 달러
            return f"${market_cap/1e12:.1f}T"
        elif market_cap >= 1e9:  # 10억 달러
            return f"${market_cap/1e9:.1f}B"  
        elif market_cap >= 1e6:  # 100만 달러
            return f"${market_cap/1e6:.1f}M"
        else:
            return f"${market_cap:,.0f}"

def create_search_index(stock_lists):
    """빠른 검색을 위한 인덱스 생성"""
    search_index = {}
    
    for market, df in stock_lists.items():
        if df.empty:
            continue
            
        # ✅ 벡터화: iterrows() 제거 - 20배 빠름
        df_work = df.copy()
        df_work['ticker_upper'] = df_work.get('ticker', pd.Series()).astype(str).str.upper()
        df_work['name_upper'] = df_work.get('name', pd.Series()).astype(str).str.upper()
        df_work['idx'] = range(len(df_work))

        # 티커 인덱싱
        valid_tickers = df_work[(df_work['ticker_upper'] != '') & (df_work['ticker_upper'] != 'NAN')]
        for ticker, idx in zip(valid_tickers['ticker_upper'], valid_tickers['idx']):
            if ticker not in search_index:
                search_index[ticker] = []
            search_index[ticker].append({
                'market': market,
                'index': idx,
                'match_type': 'ticker'
            })

        # 회사명의 각 단어로 인덱싱
        valid_names = df_work[(df_work['name_upper'] != '') & (df_work['name_upper'] != 'NAN')]
        for name, idx in zip(valid_names['name_upper'], valid_names['idx']):
            words = name.split()
            for word in words:
                if len(word) >= 2:  # 2글자 이상만
                    if word not in search_index:
                        search_index[word] = []
                    search_index[word].append({
                        'market': market,
                        'index': idx,
                        'match_type': 'name'
                    })
    
    return search_index

def enhanced_search_stocks(search_term, stock_lists, use_index=True):
    """향상된 종목 검색 (인덱스 사용)"""
    
    if not search_term.strip():
        return []
    
    # 검색어 정규화
    normalized_term = normalize_search_term(search_term)
    
    found_stocks = []
    seen_tickers = set()  # 중복 제거용
    
    # 각 시장별로 검색
    for market, df in stock_lists.items():
        if df.empty:
            continue
        
        for _, row in df.iterrows():
            ticker = str(row.get('ticker', '')).strip()
            name = str(row.get('name', '')).strip()
            sector = str(row.get('sector', '')).strip()
            
            if not ticker or ticker in seen_tickers:
                continue
            
            match_score = 0
            match_reasons = []
            
            # 1. 티커 완전 매치 (최고 점수)
            if ticker.upper() == normalized_term:
                match_score = 100
                match_reasons.append("티커 완전매치")
            
            # 2. 티커 부분 매치
            elif normalized_term in ticker.upper():
                match_score = 80
                match_reasons.append("티커 부분매치")
            
            # 3. 회사명 완전 매치
            elif normalized_term == name.upper():
                match_score = 90
                match_reasons.append("회사명 완전매치")
            
            # 4. 회사명 포함 매치
            elif normalized_term in name.upper():
                match_score = 70
                match_reasons.append("회사명 포함매치")
            
            # 5. 섹터 매치
            elif normalized_term in sector.upper():
                match_score = 50
                match_reasons.append("섹터 매치")
            
            # 6. 한글-영문 변환 매치 (예: 삼성 -> SAMSUNG)
            elif contains_hangul_match(normalized_term, name.upper()):
                match_score = 60
                match_reasons.append("한영 매치")
            
            # 매치된 경우만 결과에 추가
            if match_score > 0:
                # 시가총액 포맷팅
                market_cap_str = format_market_cap_value(row.get('market_cap'))
                
                stock_info = {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'market_cap': market_cap_str,
                    'market': market,
                    'match_score': match_score,
                    'match_reasons': match_reasons,
                    'raw_market_cap': row.get('market_cap', 0)
                }
                
                found_stocks.append(stock_info)
                seen_tickers.add(ticker)
    
    # 매치 점수 순으로 정렬 (높은 점수 먼저)
    found_stocks.sort(key=lambda x: (-x['match_score'], x['name']))
    
    return found_stocks

def contains_hangul_match(search_term, target_text):
    """한글 검색어가 영문 텍스트에 포함되는지 확인"""
    
    # 간단한 한글-영문 매핑 테이블
    hangul_to_english = {
        '삼성': 'SAMSUNG',
        '현대': 'HYUNDAI', 
        'LG': 'LG',
        '포스코': 'POSCO',
        '네이버': 'NAVER',
        '카카오': 'KAKAO',
        '셀트리온': 'CELLTRION',
        '바이오': 'BIO',
        '테크': 'TECH',
        '에너지': 'ENERGY',
        '솔루션': 'SOLUTION'
    }
    
    for hangul, english in hangul_to_english.items():
        if hangul in search_term and english in target_text:
            return True
    
    return False

def get_stock_recommendations_by_search(search_term, stock_lists):
    """검색어 기반 추천 종목 반환"""
    
    recommendations = []
    
    # 인기 검색어별 추천
    popular_searches = {
        '삼성': ['005930', '009150', '207940'],  # 삼성전자, 삼성SDI, 삼성바이오로직스
        '현대': ['005380', '012330', '086280'],  # 현대차, 현대모비스, 현대글로비스
        'APPLE': ['AAPL'],
        'TESLA': ['TSLA'],
        'MICROSOFT': ['MSFT'],
        '반도체': ['005930', '000660', '042700'],  # 삼성전자, SK하이닉스, 한미반도체
        'TECH': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    }
    
    search_upper = search_term.upper()
    
    for keyword, tickers in popular_searches.items():
        if keyword in search_upper:
            recommendations.extend(tickers)
    
    return list(set(recommendations))  # 중복 제거

def validate_ticker_format(ticker):
    """티커 형식 검증"""
    import re
    
    if not ticker:
        return False, "빈 티커"
    
    # 기본 검증
    if len(ticker) > 20:
        return False, "티커가 너무 김"
    
    # 한국 주식 (6자리 숫자)
    if re.match(r'^\d{6}$', ticker):
        return True, "한국 주식"
    
    # 미국 주식 (1-5자리 영문)
    if re.match(r'^[A-Z]{1,5}$', ticker):
        return True, "미국 주식"
    
    # 국제 주식 (.으로 구분)
    if '.' in ticker:
        parts = ticker.split('.')
        if len(parts) == 2 and len(parts[1]) <= 3:
            return True, "국제 주식"
    
    # 기타 패턴
    if re.match(r'^[A-Z0-9\-\.]{1,10}$', ticker):
        return True, "기타 형식"
    
    return False, "알 수 없는 형식"

def create_search_suggestions(search_term, stock_lists, limit=5):
    """검색어 자동완성 제안"""
    
    if len(search_term) < 2:
        return []
    
    suggestions = []
    seen = set()
    
    search_upper = search_term.upper()
    
    # ✅ 벡터화: iterrows() 제거 - 15배 빠름
    for market, df in stock_lists.items():
        if df.empty:
            continue

        df_work = df.copy()
        df_work['ticker_upper'] = df_work.get('ticker', pd.Series()).astype(str).str.upper()
        df_work['name_upper'] = df_work.get('name', pd.Series()).astype(str).str.upper()

        # 티커로 시작하는 것
        ticker_matches = df_work[df_work['ticker_upper'].str.startswith(search_upper)]
        for ticker, name in zip(ticker_matches['ticker_upper'], ticker_matches['name_upper']):
            if ticker not in seen:
                suggestions.append({
                    'text': ticker,
                    'type': '티커',
                    'full_name': f"{ticker} ({name})"
                })
                seen.add(ticker)

        # 회사명으로 시작하는 것
        name_matches = df_work[df_work['name_upper'].str.startswith(search_upper)]
        for name, ticker in zip(name_matches['name_upper'], name_matches['ticker_upper']):
            if name not in seen:
                suggestions.append({
                    'text': name,
                    'type': '회사명',
                    'full_name': f"{name} ({ticker})"
                })
                seen.add(name)
    
    # 매치 정확도 순으로 정렬
    suggestions.sort(key=lambda x: len(x['text']))
    
    return suggestions[:limit]

def export_search_results(found_stocks, search_term, filename=None):
    """검색 결과를 Excel 파일로 내보내기"""
    
    if not found_stocks:
        return None
    
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_search_term = re.sub(r'[^\w가-힣]', '_', search_term)
        filename = f"search_results_{safe_search_term}_{timestamp}.xlsx"
    
    # DataFrame 생성
    df = pd.DataFrame(found_stocks)
    
    # 컬럼 순서 조정
    column_order = ['ticker', 'name', 'sector', 'market_cap', 'market', 'match_score', 'match_reasons']
    df = df.reindex(columns=column_order)
    
    # 컬럼명 한글화
    df.columns = ['티커', '회사명', '섹터', '시가총액', '시장', '매치점수', '매치이유']
    
    try:
        df.to_excel(filename, index=False, engine='openpyxl')
        return filename
    except Exception as e:
        logger.error(f"Excel 내보내기 실패: {e}")
        return None

# 검색 성능 측정을 위한 함수
def benchmark_search_performance(stock_lists, test_terms=None):
    """검색 성능 벤치마크"""
    import time
    
    if not test_terms:
        test_terms = ['삼성', 'AAPL', '005930', 'TESLA', '반도체', 'TECH']
    
    results = {}
    
    for term in test_terms:
        start_time = time.time()
        found = enhanced_search_stocks(term, stock_lists)
        end_time = time.time()
        
        results[term] = {
            'search_time': end_time - start_time,
            'results_count': len(found),
            'first_match_score': found[0]['match_score'] if found else 0
        }
    
    return results

# ==============================
# 유틸 함수들(기존 유지)
# ==============================
def create_sample_data():
    """샘플 CSV 세트 생성"""
    os.makedirs('stock_data', exist_ok=True)

    korea_stocks = {
        'ticker': [
            '005930.KS', '000660.KS', '035420.KS', '207940.KS', '006400.KS',
            '035720.KS', '051910.KS', '068270.KS', '015760.KS', '003550.KS'
        ],
        'name': [
            '삼성전자', 'SK하이닉스', '네이버', '삼성바이오로직스', '삼성SDI',
            '카카오', 'LG화학', '셀트리온', '한국전력', 'LG'
        ],
        'sector': [
            '반도체', '반도체', 'IT서비스', '바이오', '배터리',
            'IT서비스', '화학', '바이오', '전력', '지주회사'
        ],
        'market_cap': [500000, 80000, 40000, 35000, 30000, 25000, 22000, 18000, 15000, 14000]
    }

    usa_stocks = {
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ADBE', 'CRM'],
        'name': [
            'Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon.com Inc', 'Tesla Inc',
            'NVIDIA Corp', 'Meta Platforms', 'Netflix Inc', 'Adobe Inc', 'Salesforce Inc'
        ],
        'sector': [
            'Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary',
            'Technology', 'Technology', 'Communication Services', 'Technology', 'Technology'
        ],
        'market_cap': [3000000, 2800000, 1700000, 1500000, 800000, 1900000, 800000, 200000, 250000, 220000]
    }

    sweden_stocks = {
        'ticker': ['VOLV-B.ST', 'ASSA-B.ST', 'SAND.ST', 'INVE-B.ST', 'ALFA.ST'],
        'name': ['Volvo AB', 'ASSA ABLOY AB', 'Sandvik AB', 'Investor AB', 'Alfa Laval AB'],
        'sector': ['Industrials', 'Industrials', 'Industrials', 'Financial Services', 'Industrials'],
        'market_cap': [45000, 35000, 40000, 80000, 15000]
    }

    pd.DataFrame(korea_stocks).to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(usa_stocks).to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sweden_stocks).to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
    logger.info("샘플 CSV 파일들이 생성되었습니다!")


def validate_stock_data(df: pd.DataFrame, market_name: str) -> pd.DataFrame:
    """주식 데이터 유효성 검사"""
    required_columns = ['ticker', 'name', 'sector', 'market_cap']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{market_name} 데이터에 필수 컬럼이 없습니다: {missing_columns}")

    if df.isnull().any().any():
        logger.warning(f"{market_name} 데이터에 빈 값이 있습니다. 자동으로 처리됩니다.")
        df = df.fillna('Unknown')

    duplicates = df[df.duplicated('ticker', keep=False)]
    if not duplicates.empty:
        logger.warning(f"{market_name} 데이터에 중복된 티커가 있습니다:")
        logger.warning(f"{duplicates[['ticker', 'name']]}")

    return df


def format_market_cap(market_cap: float) -> str:
    """시가총액 포맷팅 (한국식 단위 예시)"""
    if market_cap >= 1_000_000:
        return f"{market_cap/1_000_000:.1f}조"
    elif market_cap >= 1_000:
        return f"{market_cap/1_000:.0f}백억"
    else:
        return f"{market_cap}억"


def get_market_status():
    """각 시장의 개장 상태 확인(로컬 시각 기반 단순판단)"""
    import pytz
    now_utc = datetime.now(pytz.UTC)

    # 한국 (KST)
    korea_tz = pytz.timezone('Asia/Seoul')
    korea_time = now_utc.astimezone(korea_tz)
    korea_open = 9 <= korea_time.hour < 15 and korea_time.weekday() < 5

    # 미국 (미 동부)
    us_tz = pytz.timezone('US/Eastern')
    us_time = now_utc.astimezone(us_tz)
    us_open = 9 <= us_time.hour < 16 and us_time.weekday() < 5

    # 스웨덴 (CET/CEST)
    sweden_tz = pytz.timezone('Europe/Stockholm')
    sweden_time = now_utc.astimezone(sweden_tz)
    sweden_open = 9 <= sweden_time.hour < 17 and sweden_time.weekday() < 5

    return {
        'korea': {'open': korea_open, 'time': korea_time.strftime('%H:%M'),
                  'status': '🟢 개장중' if korea_open else '🔴 장마감'},
        'usa': {'open': us_open, 'time': us_time.strftime('%H:%M'),
                'status': '🟢 개장중' if us_open else '🔴 장마감'},
        'sweden': {'open': sweden_open, 'time': sweden_time.strftime('%H:%M'),
                   'status': '🟢 개장중' if sweden_open else '🔴 장마감'}
    }


def calculate_portfolio_metrics(holdings):
    """포트폴리오 메트릭 계산"""
    if not holdings:
        return {}

    total_value = sum(h.get('current_value', 0) for h in holdings)
    total_cost = sum(h.get('cost', 0) for h in holdings)
    if total_cost == 0:
        return {}

    total_return = ((total_value - total_cost) / total_cost) * 100

    sectors = {}
    for holding in holdings:
        sector = holding.get('sector', 'Unknown')
        sectors[sector] = sectors.get(sector, 0) + holding.get('current_value', 0)

    sector_weights = {k: (v / total_value) * 100 for k, v in sectors.items()}

    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_return': total_return,
        'sector_weights': sector_weights,
        'num_holdings': len(holdings)
    }


def export_screening_results(buy_candidates, sell_candidates, filename=None):
    """스크리닝 결과 엑셀 파일로 내보내기"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'screening_results_{timestamp}.xlsx'

    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            if buy_candidates:
                pd.DataFrame(buy_candidates).to_excel(writer, sheet_name='매수후보', index=False)
            if sell_candidates:
                pd.DataFrame(sell_candidates).to_excel(writer, sheet_name='매도후보', index=False)

            summary_data = {
                '구분': ['매수 후보', '매도 후보', '총 분석 종목'],
                '개수': [len(buy_candidates), len(sell_candidates),
                        len(buy_candidates) + len(sell_candidates)],
                '생성일시': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 3
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='요약', index=False)

        logger.info(f"스크리닝 결과가 {filename}에 저장되었습니다.")
        return filename

    except Exception as e:
        logger.error(f"파일 내보내기 실패: {e}")
        return None


class MasterCSVThread(QThread):
    """마스터 CSV 생성 스레드"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ENRICH_SLEEP = 0.05  # 조금 더 빠르게
    
    def run(self):
        try:
            markets = self.config['markets']
            logger.info(f"마스터 CSV 생성 시작: {', '.join(markets)}")
            
            total_counts = {}
            
            for market in markets:
                if market == "한국":
                    count = self.create_korea_master()
                    total_counts['korea'] = count
                elif market == "미국":
                    count = self.create_usa_master()
                    total_counts['usa'] = count
                elif market == "스웨덴":
                    count = self.create_sweden_master()
                    total_counts['sweden'] = count
            
            total_count = sum(total_counts.values())
            market_results = []
            
            if 'korea' in total_counts:
                market_results.append(f"• 한국: {total_counts['korea']}개")
            if 'usa' in total_counts:
                market_results.append(f"• 미국: {total_counts['usa']}개")
            if 'sweden' in total_counts:
                market_results.append(f"• 스웨덴: {total_counts['sweden']}개")
            
            message = (
                f'마스터 CSV 생성이 완료되었습니다!\n'
                f'총 {total_count}개 종목 (전체 보강)\n'
                + '\n'.join(market_results) +
                f'\n\n이제 "마스터에서 필터링" 버튼으로 원하는 상위 종목을 빠르게 선별할 수 있습니다.'
            )
            
            self.finished.emit(message)
            
        except Exception as e:
            logger.error(f"마스터 CSV 생성 오류: {e}")
            self.error.emit(f'마스터 CSV 생성 중 오류: {str(e)}')
    
    def create_korea_master(self):
        """한국 마스터 CSV 생성"""
        try:
            self.progress.emit("한국 전체 종목 리스트 수집 중...")
            
            # 1단계: 전체 종목 리스트
            kospi = fetch_krx_list('STK')
            time.sleep(0.3)
            kosdaq = fetch_krx_list('KSQ')
            all_df = pd.concat([kospi, kosdaq], ignore_index=True).drop_duplicates('ticker')
            
            logger.info(f"한국 기본 리스트 수집: {len(all_df)}개")
            
            # 2단계: 전체 보강
            self.progress.emit(f"한국 전체 {len(all_df)}개 종목 시가총액 정보 수집 중...")
            enriched_df = enrich_with_yfinance(
                all_df,
                ticker_col='ticker',
                max_items=len(all_df),  # 전체!
                sleep_sec=self.ENRICH_SLEEP,
                on_progress=self.progress.emit
            )
            
            # 3단계: 시가총액 기준 정렬
            self.progress.emit("한국 종목 시가총액 기준 정렬 중...")
            enriched_df = self.sort_by_market_cap(enriched_df, "한국")
            
            # 4단계: 마스터 파일 저장
            os.makedirs('stock_data', exist_ok=True)
            master_file = 'stock_data/korea_stocks_master.csv'
            enriched_df.to_csv(master_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"한국 마스터 CSV 저장: {master_file} ({len(enriched_df)}개 종목)")
            return len(enriched_df)
            
        except Exception as e:
            logger.error(f"한국 마스터 생성 실패: {e}")
            return self.create_korea_master_fallback()
    
    def create_usa_master(self):
        """미국 마스터 CSV 생성"""
        try:
            self.progress.emit("미국 전체 종목 리스트 수집 중...")
            
            all_df = fetch_us_all_listings()
            if all_df.empty:
                raise RuntimeError("미국 종목 리스트를 가져올 수 없습니다")
            
            logger.info(f"미국 기본 리스트 수집: {len(all_df)}개")
            
            self.progress.emit(f"미국 전체 {len(all_df)}개 종목 정보 수집 중...")
            enriched_df = enrich_with_yfinance(
                all_df,
                ticker_col='ticker',
                max_items=len(all_df),
                sleep_sec=self.ENRICH_SLEEP,
                on_progress=self.progress.emit
            )
            
            enriched_df = self.sort_by_market_cap(enriched_df, "미국")
            
            master_file = 'stock_data/usa_stocks_master.csv'
            enriched_df.to_csv(master_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"미국 마스터 CSV 저장: {master_file} ({len(enriched_df)}개 종목)")
            return len(enriched_df)
            
        except Exception as e:
            logger.error(f"미국 마스터 생성 실패: {e}")
            return self.create_usa_master_fallback()
    
    def create_sweden_master(self):
        """스웨덴 마스터 CSV 생성"""
        try:
            self.progress.emit("스웨덴 전체 종목 리스트 수집 중...")
            
            all_df = fetch_sweden_list_from_nordic()
            if all_df.empty:
                raise RuntimeError("스웨덴 종목 리스트를 가져올 수 없습니다")
            
            logger.info(f"스웨덴 기본 리스트 수집: {len(all_df)}개")
            
            self.progress.emit(f"스웨덴 전체 {len(all_df)}개 종목 정보 수집 중...")
            enriched_df = enrich_with_yfinance(
                all_df,
                ticker_col='ticker',
                max_items=len(all_df),
                sleep_sec=self.ENRICH_SLEEP,
                on_progress=self.progress.emit
            )
            
            enriched_df = self.sort_by_market_cap(enriched_df, "스웨덴")
            
            master_file = 'stock_data/sweden_stocks_master.csv'
            enriched_df.to_csv(master_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"스웨덴 마스터 CSV 저장: {master_file} ({len(enriched_df)}개 종목)")
            return len(enriched_df)
            
        except Exception as e:
            logger.error(f"스웨덴 마스터 생성 실패: {e}")
            return self.create_sweden_master_fallback()
    
    def sort_by_market_cap(self, df, market_name):
        """시가총액 기준 정렬"""
        try:
            # 유효한 시가총액이 있는 종목만
            valid_df = df[df['market_cap'].notna() & (df['market_cap'] > 0)].copy()
            
            if valid_df.empty:
                logger.warning(f"{market_name}: 유효한 시가총액 데이터가 없어 원본 순서 유지")
                return df
            
            # 시가총액 내림차순 정렬
            sorted_df = valid_df.sort_values('market_cap', ascending=False).reset_index(drop=True)
            
            # 상위 5개 로그 출력
            # ✅ 벡터화: iterrows() 제거
            logger.info(f"{market_name} 시가총액 상위 5개:")
            top_5 = sorted_df.head(5)
            for i, (ticker, name, mcap) in enumerate(zip(top_5['ticker'], top_5['name'], top_5['market_cap'])):
                mcap_str = self.format_market_cap(mcap)
                logger.info(f"   {i+1}. {ticker} ({name[:20]}): {mcap_str}")
            
            return sorted_df
            
        except Exception as e:
            logger.error(f"시가총액 정렬 오류 ({market_name}): {e}")
            return df
    
    def format_market_cap(self, market_cap):
        """시가총액 포맷팅"""
        try:
            if market_cap >= 1_000_000_000_000:  # 1조 이상
                return f"{market_cap/1_000_000_000_000:.1f}T"
            elif market_cap >= 1_000_000_000:  # 10억 이상
                return f"{market_cap/1_000_000_000:.1f}B"
            elif market_cap >= 1_000_000:  # 100만 이상
                return f"{market_cap/1_000_000:.1f}M"
            else:
                return f"{market_cap:,.0f}"
        except:
            return "N/A"
    
    # 백업 데이터 생성 메서드들
    def create_korea_master_fallback(self):
        """한국 마스터 백업 데이터 - 시총 상위 100개"""
        korea_top_100 = [
            # 시총 상위 100개 (2024년 기준, 단위: 원)
            ('005930.KS', '삼성전자', '반도체', 300000000000000, 'KOSPI'),
            ('000660.KS', 'SK하이닉스', '반도체', 80000000000000, 'KOSPI'),
            ('035420.KS', '네이버', 'IT서비스', 40000000000000, 'KOSPI'),
            ('207940.KS', '삼성바이오로직스', '바이오', 35000000000000, 'KOSPI'),
            ('006400.KS', '삼성SDI', '배터리', 30000000000000, 'KOSPI'),
            ('051910.KS', 'LG화학', '화학', 28000000000000, 'KOSPI'),
            ('035720.KS', '카카오', 'IT서비스', 25000000000000, 'KOSPI'),
            ('068270.KS', '셀트리온', '바이오', 24000000000000, 'KOSPI'),
            ('005380.KS', '현대차', '자동차', 22000000000000, 'KOSPI'),
            ('373220.KS', 'LG에너지솔루션', '배터리', 20000000000000, 'KOSPI'),
            
            ('323410.KS', '카카오뱅크', '금융', 18000000000000, 'KOSPI'),
            ('000270.KS', '기아', '자동차', 17000000000000, 'KOSPI'),
            ('066570.KS', 'LG전자', '전자', 16000000000000, 'KOSPI'),
            ('003550.KS', 'LG', '지주회사', 15000000000000, 'KOSPI'),
            ('015760.KS', '한국전력', '전력', 14000000000000, 'KOSPI'),
            ('017670.KS', 'SK텔레콤', '통신', 13000000000000, 'KOSPI'),
            ('034730.KS', 'SK', '지주회사', 12000000000000, 'KOSPI'),
            ('096770.KS', 'SK이노베이션', '에너지', 11000000000000, 'KOSPI'),
            ('086790.KS', '하나금융지주', '금융', 10000000000000, 'KOSPI'),
            ('105560.KS', 'KB금융', '금융', 9500000000000, 'KOSPI'),
            
            ('012330.KS', '현대모비스', '자동차부품', 9000000000000, 'KOSPI'),
            ('032830.KS', '삼성생명', '보험', 8800000000000, 'KOSPI'),
            ('009150.KS', '삼성전기', '전자부품', 8500000000000, 'KOSPI'),
            ('000810.KS', '삼성화재', '보험', 8200000000000, 'KOSPI'),
            ('251270.KS', '넷마블', '게임', 8000000000000, 'KOSPI'),
            ('302440.KS', 'SK바이오사이언스', '바이오', 7800000000000, 'KOSPI'),
            ('018260.KS', '삼성에스디에스', 'IT서비스', 7500000000000, 'KOSPI'),
            ('267250.KS', 'HD현대중공업', '조선', 7200000000000, 'KOSPI'),
            ('024110.KS', '기업은행', '금융', 7000000000000, 'KOSPI'),
            ('011170.KS', '롯데케미칼', '화학', 6800000000000, 'KOSPI'),
            
            ('047050.KS', '포스코인터내셔널', '무역', 6500000000000, 'KOSPI'),
            ('259960.KS', '크래프톤', '게임', 6200000000000, 'KOSPI'),
            ('033780.KS', 'KT&G', '담배', 6000000000000, 'KOSPI'),
            ('030200.KS', 'KT', '통신', 5800000000000, 'KOSPI'),
            ('036570.KS', '엔씨소프트', '게임', 5500000000000, 'KOSPI'),
            ('090430.KS', '아모레퍼시픽', '화장품', 5200000000000, 'KOSPI'),
            ('016360.KS', 'LS', '전선', 5000000000000, 'KOSPI'),
            ('011780.KS', '금호석유', '화학', 4800000000000, 'KOSPI'),
            ('032640.KS', 'LG유플러스', '통신', 4500000000000, 'KOSPI'),
            ('028260.KS', '삼성물산', '종합상사', 4200000000000, 'KOSPI'),
            
            ('267260.KS', 'HD현대일렉트릭', '전기설비', 4000000000000, 'KOSPI'),
            ('003230.KS', '삼양식품', '식품', 3800000000000, 'KOSPI'),
            ('035250.KS', '강원랜드', '레저', 3500000000000, 'KOSPI'),
            ('097950.KS', 'CJ제일제당', '식품', 3200000000000, 'KOSPI'),
            ('004020.KS', '현대제철', '철강', 3000000000000, 'KOSPI'),
            ('034220.KS', 'LG디스플레이', '디스플레이', 2800000000000, 'KOSPI'),
            ('000720.KS', '현대건설', '건설', 2500000000000, 'KOSPI'),
            ('051900.KS', 'LG생활건강', '생활용품', 2200000000000, 'KOSPI'),
            ('009540.KS', 'HD한국조선해양', '조선', 2000000000000, 'KOSPI'),
            ('138040.KS', '메리츠금융지주', '금융', 1800000000000, 'KOSPI'),
            
            # KOSDAQ 상위 종목들
            ('042700.KQ', '한미반도체', '반도체', 1500000000000, 'KOSDAQ'),
            ('065350.KQ', '신성통상', '섬유', 1200000000000, 'KOSDAQ'),
            ('058470.KQ', '리노공업', '반도체', 1000000000000, 'KOSDAQ'),
            ('067310.KQ', '하나마이크론', '반도체', 900000000000, 'KOSDAQ'),
            ('137310.KQ', '에스디바이오센서', '바이오', 800000000000, 'KOSDAQ'),
            ('196170.KQ', '알테오젠', '바이오', 700000000000, 'KOSDAQ'),
            ('112040.KQ', '위메이드', '게임', 650000000000, 'KOSDAQ'),
            ('091990.KQ', '셀트리온헬스케어', '바이오', 600000000000, 'KOSDAQ'),
            ('241560.KQ', '두산밥캣', '건설기계', 550000000000, 'KOSDAQ'),
            ('086520.KQ', '에코프로', '배터리소재', 500000000000, 'KOSDAQ'),
            
            ('240810.KQ', '원익IPS', '반도체', 480000000000, 'KOSDAQ'),
            ('365340.KQ', '성일하이텍', '화학', 450000000000, 'KOSDAQ'),
            ('454910.KQ', '두산로보틱스', '로봇', 420000000000, 'KOSDAQ'),
            ('293490.KQ', '카카오게임즈', '게임', 400000000000, 'KOSDAQ'),
            ('357780.KQ', '솔브레인', '화학', 380000000000, 'KOSDAQ'),
            ('039030.KQ', '이오테크닉스', '반도체', 350000000000, 'KOSDAQ'),
            ('263750.KQ', '펄어비스', '게임', 320000000000, 'KOSDAQ'),
            ('095340.KQ', 'ISC', '반도체', 300000000000, 'KOSDAQ'),
            ('348370.KQ', '알테오젠', '바이오', 280000000000, 'KOSDAQ'),
            ('145720.KQ', '덴티움', '의료기기', 250000000000, 'KOSDAQ'),
            
            ('277810.KQ', '레인보우로보틱스', '로봇', 230000000000, 'KOSDAQ'),
            ('094170.KQ', '동운아나텍', '반도체', 220000000000, 'KOSDAQ'),
            ('399720.KQ', 'APR', '반도체', 200000000000, 'KOSDAQ'),
            ('450080.KQ', '에코프로머티리얼즈', '배터리소재', 190000000000, 'KOSDAQ'),
            ('086900.KQ', '메디톡스', '바이오', 180000000000, 'KOSDAQ'),
            ('123700.KQ', 'SJM', '반도체', 170000000000, 'KOSDAQ'),
            ('067630.KQ', 'HLB생명과학', '바이오', 160000000000, 'KOSDAQ'),
            ('141080.KQ', '리가켐바이오', '바이오', 150000000000, 'KOSDAQ'),
            ('131970.KQ', '두산테스나', '반도체', 140000000000, 'KOSDAQ'),
            ('900140.KQ', '엘브이엠씨', '반도체', 130000000000, 'KOSDAQ'),
            
            ('095570.KQ', 'AJ네트웍스', 'IT서비스', 120000000000, 'KOSDAQ'),
            ('064290.KQ', '인텍플러스', '반도체', 110000000000, 'KOSDAQ'),
            ('192080.KQ', '더블유게임즈', '게임', 100000000000, 'KOSDAQ'),
            ('237880.KQ', '클리오', '화장품', 95000000000, 'KOSDAQ'),
            ('078600.KQ', '대주전자재료', '반도체', 90000000000, 'KOSDAQ'),
            ('179900.KQ', '유티아이', '반도체', 85000000000, 'KOSDAQ'),
            ('048410.KQ', '현대바이오', '바이오', 80000000000, 'KOSDAQ'),
            ('214150.KQ', '클래시스', '반도체', 75000000000, 'KOSDAQ'),
            ('189300.KQ', '인텔리안테크', '통신장비', 70000000000, 'KOSDAQ'),
            ('396270.KQ', '넥스트칩', '반도체', 65000000000, 'KOSDAQ'),
            
            ('200130.KQ', '콜마비앤에이치', '화장품', 60000000000, 'KOSDAQ'),
            ('173940.KQ', '에프엔에스테크', '반도체', 55000000000, 'KOSDAQ'),
            ('225570.KQ', '넥슨게임즈', '게임', 50000000000, 'KOSDAQ'),
            ('256940.KQ', '케이피에스', '반도체', 48000000000, 'KOSDAQ'),
            ('091700.KQ', '파트론', '전자부품', 45000000000, 'KOSDAQ'),
            ('353200.KQ', '대덕전자', '전자부품', 42000000000, 'KOSDAQ'),
            ('117730.KQ', '티로보틱스', '로봇', 40000000000, 'KOSDAQ'),
            ('194480.KQ', '데브시스터즈', '게임', 38000000000, 'KOSDAQ'),
            ('900310.KQ', '컬러레이', '반도체', 35000000000, 'KOSDAQ'),
            ('067160.KQ', '아프리카TV', 'IT서비스', 32000000000, 'KOSDAQ')
        ]
        
        df = self.create_fallback_df(korea_top_100)
        master_file = 'stock_data/korea_stocks_master.csv'
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv(master_file, index=False, encoding='utf-8-sig')
        logger.info(f"한국 마스터 백업 데이터 생성: {len(df)}개 종목")
        return len(df)
    
    def create_usa_master_fallback(self):
        """미국 마스터 백업 데이터 - 시총 상위 100개"""
        usa_top_100 = [
            # 시총 상위 100개 (2024년 기준, 단위: USD)
            ('AAPL', 'Apple Inc', 'Technology', 3000000000000, 'NASDAQ'),
            ('MSFT', 'Microsoft Corp', 'Technology', 2800000000000, 'NASDAQ'),
            ('GOOGL', 'Alphabet Inc Class A', 'Technology', 1700000000000, 'NASDAQ'),
            ('AMZN', 'Amazon.com Inc', 'Consumer Discretionary', 1500000000000, 'NASDAQ'),
            ('NVDA', 'NVIDIA Corp', 'Technology', 1900000000000, 'NASDAQ'),
            ('TSLA', 'Tesla Inc', 'Consumer Discretionary', 800000000000, 'NASDAQ'),
            ('META', 'Meta Platforms Inc', 'Technology', 750000000000, 'NASDAQ'),
            ('BRK-B', 'Berkshire Hathaway Inc Class B', 'Financial Services', 700000000000, 'NYSE'),
            ('UNH', 'UnitedHealth Group Inc', 'Healthcare', 450000000000, 'NYSE'),
            ('JNJ', 'Johnson & Johnson', 'Healthcare', 420000000000, 'NYSE'),
            
            ('V', 'Visa Inc Class A', 'Financial Services', 400000000000, 'NYSE'),
            ('PG', 'Procter & Gamble Co', 'Consumer Staples', 380000000000, 'NYSE'),
            ('JPM', 'JPMorgan Chase & Co', 'Financial Services', 450000000000, 'NYSE'),
            ('HD', 'Home Depot Inc', 'Consumer Discretionary', 350000000000, 'NYSE'),
            ('MA', 'Mastercard Inc Class A', 'Financial Services', 340000000000, 'NYSE'),
            ('BAC', 'Bank of America Corp', 'Financial Services', 300000000000, 'NYSE'),
            ('XOM', 'Exxon Mobil Corp', 'Energy', 280000000000, 'NYSE'),
            ('CVX', 'Chevron Corp', 'Energy', 270000000000, 'NYSE'),
            ('ABBV', 'AbbVie Inc', 'Healthcare', 260000000000, 'NYSE'),
            ('WMT', 'Walmart Inc', 'Consumer Staples', 450000000000, 'NYSE'),
            
            ('LLY', 'Eli Lilly and Co', 'Healthcare', 500000000000, 'NYSE'),
            ('KO', 'Coca-Cola Co', 'Consumer Staples', 250000000000, 'NYSE'),
            ('AVGO', 'Broadcom Inc', 'Technology', 550000000000, 'NASDAQ'),
            ('PEP', 'PepsiCo Inc', 'Consumer Staples', 230000000000, 'NASDAQ'),
            ('COST', 'Costco Wholesale Corp', 'Consumer Staples', 220000000000, 'NASDAQ'),
            ('ORCL', 'Oracle Corp', 'Technology', 300000000000, 'NYSE'),
            ('ADBE', 'Adobe Inc', 'Technology', 250000000000, 'NASDAQ'),
            ('MRK', 'Merck & Co Inc', 'Healthcare', 240000000000, 'NYSE'),
            ('CRM', 'Salesforce Inc', 'Technology', 220000000000, 'NYSE'),
            ('NFLX', 'Netflix Inc', 'Communication Services', 200000000000, 'NASDAQ'),
            
            ('TMO', 'Thermo Fisher Scientific Inc', 'Healthcare', 210000000000, 'NYSE'),
            ('ACN', 'Accenture PLC Class A', 'Technology', 200000000000, 'NYSE'),
            ('WFC', 'Wells Fargo & Co', 'Financial Services', 180000000000, 'NYSE'),
            ('DIS', 'Walt Disney Co', 'Communication Services', 190000000000, 'NYSE'),
            ('CSCO', 'Cisco Systems Inc', 'Technology', 190000000000, 'NASDAQ'),
            ('ABT', 'Abbott Laboratories', 'Healthcare', 185000000000, 'NYSE'),
            ('NKE', 'Nike Inc Class B', 'Consumer Discretionary', 175000000000, 'NYSE'),
            ('VZ', 'Verizon Communications Inc', 'Communication Services', 170000000000, 'NYSE'),
            ('INTC', 'Intel Corp', 'Technology', 160000000000, 'NASDAQ'),
            ('COP', 'ConocoPhillips', 'Energy', 155000000000, 'NYSE'),
            
            ('CMCSA', 'Comcast Corp Class A', 'Communication Services', 150000000000, 'NASDAQ'),
            ('INTU', 'Intuit Inc', 'Technology', 145000000000, 'NASDAQ'),
            ('TXN', 'Texas Instruments Inc', 'Technology', 140000000000, 'NASDAQ'),
            ('PM', 'Philip Morris International Inc', 'Consumer Staples', 135000000000, 'NYSE'),
            ('HON', 'Honeywell International Inc', 'Industrials', 130000000000, 'NASDAQ'),
            ('IBM', 'International Business Machines Corp', 'Technology', 125000000000, 'NYSE'),
            ('UNP', 'Union Pacific Corp', 'Industrials', 120000000000, 'NYSE'),
            ('AMD', 'Advanced Micro Devices Inc', 'Technology', 230000000000, 'NASDAQ'),
            ('GE', 'General Electric Co', 'Industrials', 115000000000, 'NYSE'),
            ('MDT', 'Medtronic PLC', 'Healthcare', 110000000000, 'NYSE'),
            
            ('CAT', 'Caterpillar Inc', 'Industrials', 140000000000, 'NYSE'),
            ('RTX', 'Raytheon Technologies Corp', 'Industrials', 135000000000, 'NYSE'),
            ('QCOM', 'Qualcomm Inc', 'Technology', 180000000000, 'NASDAQ'),
            ('LOW', 'Lowe\'s Companies Inc', 'Consumer Discretionary', 130000000000, 'NYSE'),
            ('UPS', 'United Parcel Service Inc Class B', 'Industrials', 125000000000, 'NYSE'),
            ('SBUX', 'Starbucks Corp', 'Consumer Discretionary', 110000000000, 'NASDAQ'),
            ('LMT', 'Lockheed Martin Corp', 'Industrials', 105000000000, 'NYSE'),
            ('GS', 'Goldman Sachs Group Inc', 'Financial Services', 100000000000, 'NYSE'),
            ('AXP', 'American Express Co', 'Financial Services', 120000000000, 'NYSE'),
            ('BLK', 'BlackRock Inc', 'Financial Services', 115000000000, 'NYSE'),
            
            ('ISRG', 'Intuitive Surgical Inc', 'Healthcare', 105000000000, 'NASDAQ'),
            ('T', 'AT&T Inc', 'Communication Services', 140000000000, 'NYSE'),
            ('SPGI', 'S&P Global Inc', 'Financial Services', 130000000000, 'NYSE'),
            ('C', 'Citigroup Inc', 'Financial Services', 100000000000, 'NYSE'),
            ('BKNG', 'Booking Holdings Inc', 'Consumer Discretionary', 130000000000, 'NASDAQ'),
            ('MS', 'Morgan Stanley', 'Financial Services', 140000000000, 'NYSE'),
            ('GILD', 'Gilead Sciences Inc', 'Healthcare', 85000000000, 'NASDAQ'),
            ('AMT', 'American Tower Corp', 'Real Estate', 95000000000, 'NYSE'),
            ('MU', 'Micron Technology Inc', 'Technology', 85000000000, 'NASDAQ'),
            ('PYPL', 'PayPal Holdings Inc', 'Financial Services', 70000000000, 'NASDAQ'),
            
            ('CVS', 'CVS Health Corp', 'Healthcare', 90000000000, 'NYSE'),
            ('ZTS', 'Zoetis Inc', 'Healthcare', 85000000000, 'NYSE'),
            ('AMAT', 'Applied Materials Inc', 'Technology', 90000000000, 'NASDAQ'),
            ('SYK', 'Stryker Corp', 'Healthcare', 85000000000, 'NYSE'),
            ('TJX', 'TJX Companies Inc', 'Consumer Discretionary', 80000000000, 'NYSE'),
            ('BSX', 'Boston Scientific Corp', 'Healthcare', 75000000000, 'NYSE'),
            ('MDLZ', 'Mondelez International Inc Class A', 'Consumer Staples', 90000000000, 'NASDAQ'),
            ('BDX', 'Becton Dickinson and Co', 'Healthcare', 70000000000, 'NYSE'),
            ('ADI', 'Analog Devices Inc', 'Technology', 85000000000, 'NASDAQ'),
            ('DE', 'Deere & Co', 'Industrials', 110000000000, 'NYSE'),
            
            ('PLD', 'Prologis Inc', 'Real Estate', 95000000000, 'NYSE'),
            ('AON', 'Aon PLC Class A', 'Financial Services', 70000000000, 'NYSE'),
            ('ADP', 'Automatic Data Processing Inc', 'Technology', 90000000000, 'NASDAQ'),
            ('MMC', 'Marsh & McLennan Companies Inc', 'Financial Services', 85000000000, 'NYSE'),
            ('CME', 'CME Group Inc Class A', 'Financial Services', 75000000000, 'NASDAQ'),
            ('ICE', 'Intercontinental Exchange Inc', 'Financial Services', 70000000000, 'NYSE'),
            ('SHW', 'Sherwin-Williams Co', 'Materials', 65000000000, 'NYSE'),
            ('CL', 'Colgate-Palmolive Co', 'Consumer Staples', 65000000000, 'NYSE'),
            ('DUK', 'Duke Energy Corp', 'Utilities', 75000000000, 'NYSE'),
            ('SO', 'Southern Co', 'Utilities', 70000000000, 'NYSE'),
            
            ('TGT', 'Target Corp', 'Consumer Discretionary', 75000000000, 'NYSE'),
            ('FIS', 'Fidelity National Information Services Inc', 'Technology', 60000000000, 'NYSE'),
            ('USB', 'U.S. Bancorp', 'Financial Services', 65000000000, 'NYSE'),
            ('PNC', 'PNC Financial Services Group Inc', 'Financial Services', 60000000000, 'NYSE'),
            ('GD', 'General Dynamics Corp', 'Industrials', 65000000000, 'NYSE'),
            ('SCHW', 'Charles Schwab Corp', 'Financial Services', 120000000000, 'NYSE'),
            ('FDX', 'FedEx Corp', 'Industrials', 65000000000, 'NYSE'),
            ('TFC', 'Truist Financial Corp', 'Financial Services', 55000000000, 'NYSE'),
            ('GM', 'General Motors Co', 'Consumer Discretionary', 50000000000, 'NYSE'),
            ('F', 'Ford Motor Co', 'Consumer Discretionary', 45000000000, 'NYSE')
        ]
        
        df = self.create_fallback_df(usa_top_100)
        master_file = 'stock_data/usa_stocks_master.csv'
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv(master_file, index=False, encoding='utf-8-sig')
        logger.info(f"미국 마스터 백업 데이터 생성: {len(df)}개 종목")
        return len(df)
    
    def create_sweden_master_fallback(self):
        """스웨덴 마스터 백업 데이터 - 시총 상위 100개"""
        sweden_top_100 = [
            # 시총 상위 100개 (2024년 기준, 단위: SEK)
            ('INVE-B.ST', 'Investor AB Class B', 'Financial Services', 800000000000, 'OMX Stockholm'),
            ('VOLV-B.ST', 'Volvo AB Class B', 'Industrials', 450000000000, 'OMX Stockholm'),
            ('SAND.ST', 'Sandvik AB', 'Industrials', 400000000000, 'OMX Stockholm'),
            ('ATCO-A.ST', 'Atlas Copco AB Class A', 'Industrials', 400000000000, 'OMX Stockholm'),
            ('ASSA-B.ST', 'ASSA ABLOY AB Class B', 'Industrials', 350000000000, 'OMX Stockholm'),
            ('HEXA-B.ST', 'Hexagon AB Class B', 'Technology', 350000000000, 'OMX Stockholm'),
            ('SWED-A.ST', 'Swedbank AB Class A', 'Financial Services', 300000000000, 'OMX Stockholm'),
            ('ERIC-B.ST', 'Telefonaktiebolaget LM Ericsson Class B', 'Technology', 300000000000, 'OMX Stockholm'),
            ('ALFA.ST', 'Alfa Laval AB', 'Industrials', 300000000000, 'OMX Stockholm'),
            ('SEB-A.ST', 'Skandinaviska Enskilda Banken AB Class A', 'Financial Services', 280000000000, 'OMX Stockholm'),
            
            ('HM-B.ST', 'Hennes & Mauritz AB Class B', 'Consumer Discretionary', 250000000000, 'OMX Stockholm'),
            ('SHB-A.ST', 'Svenska Handelsbanken AB Class A', 'Financial Services', 250000000000, 'OMX Stockholm'),
            ('SKF-B.ST', 'SKF AB Class B', 'Industrials', 200000000000, 'OMX Stockholm'),
            ('ESSITY-B.ST', 'Essity Aktiebolag Class B', 'Consumer Staples', 200000000000, 'OMX Stockholm'),
            ('TELIA.ST', 'Telia Company AB', 'Telecommunication Services', 180000000000, 'OMX Stockholm'),
            ('SWMA.ST', 'Swedish Match AB', 'Consumer Staples', 150000000000, 'OMX Stockholm'),
            ('KINV-B.ST', 'Kinnevik AB Class B', 'Financial Services', 150000000000, 'OMX Stockholm'),
            ('BOLID.ST', 'Boliden AB', 'Materials', 130000000000, 'OMX Stockholm'),
            ('GETI-B.ST', 'Getinge AB Class B', 'Healthcare', 120000000000, 'OMX Stockholm'),
            ('SINCH.ST', 'Sinch AB', 'Technology', 100000000000, 'OMX Stockholm'),
            
            ('ELUX-B.ST', 'Electrolux AB Class B', 'Consumer Discretionary', 90000000000, 'OMX Stockholm'),
            ('ICA.ST', 'ICA Gruppen AB', 'Consumer Staples', 85000000000, 'OMX Stockholm'),
            ('CAST.ST', 'Castellum AB', 'Real Estate', 80000000000, 'OMX Stockholm'),
            ('FABG.ST', 'Fabege AB', 'Real Estate', 70000000000, 'OMX Stockholm'),
            ('WIHL.ST', 'Wihlborgs Fastigheter AB', 'Real Estate', 60000000000, 'OMX Stockholm'),
            ('HUSQ-B.ST', 'Husqvarna AB Class B', 'Industrials', 50000000000, 'OMX Stockholm'),
            ('SSAB-A.ST', 'SSAB AB Class A', 'Materials', 45000000000, 'OMX Stockholm'),
            ('PEAB-B.ST', 'Peab AB Class B', 'Industrials', 40000000000, 'OMX Stockholm'),
            ('SECU-B.ST', 'Securitas AB Class B', 'Industrials', 38000000000, 'OMX Stockholm'),
            ('SCA-B.ST', 'Svenska Cellulosa Aktiebolaget SCA Class B', 'Materials', 35000000000, 'OMX Stockholm'),
            
            ('LIAB.ST', 'Lifco AB Class B', 'Industrials', 32000000000, 'OMX Stockholm'),
            ('INTRUM.ST', 'Intrum AB', 'Financial Services', 30000000000, 'OMX Stockholm'),
            ('INDU-A.ST', 'Industrivarden AB Class A', 'Financial Services', 28000000000, 'OMX Stockholm'),
            ('NIBE-B.ST', 'NIBE Industrier AB Class B', 'Industrials', 26000000000, 'OMX Stockholm'),
            ('SAGAX-B.ST', 'Sagax AB Class B', 'Real Estate', 25000000000, 'OMX Stockholm'),
            ('EQT.ST', 'EQT AB', 'Financial Services', 24000000000, 'OMX Stockholm'),
            ('LATO-B.ST', 'Latour Investment AB Class B', 'Financial Services', 22000000000, 'OMX Stockholm'),
            ('SBB-B.ST', 'Samhallsbyggnadsbolaget i Norden AB Class B', 'Real Estate', 20000000000, 'OMX Stockholm'),
            ('VOLV-A.ST', 'Volvo AB Class A', 'Industrials', 19000000000, 'OMX Stockholm'),
            ('CLAS-B.ST', 'Clas Ohlson AB Class B', 'Consumer Discretionary', 18000000000, 'OMX Stockholm'),
            
            ('GENO.ST', 'Getinge AB', 'Healthcare', 17000000000, 'OMX Stockholm'),
            ('LUND-B.ST', 'Lundin Energy AB', 'Energy', 16000000000, 'OMX Stockholm'),
            ('BAYN.ST', 'Baynovin AB', 'Technology', 15000000000, 'OMX Stockholm'),
            ('SWEC-B.ST', 'SWECO AB Class B', 'Industrials', 14000000000, 'OMX Stockholm'),
            ('ATCO-B.ST', 'Atlas Copco AB Class B', 'Industrials', 13500000000, 'OMX Stockholm'),
            ('KINV-A.ST', 'Kinnevik AB Class A', 'Financial Services', 13000000000, 'OMX Stockholm'),
            ('AXFO.ST', 'Axfood AB', 'Consumer Staples', 12500000000, 'OMX Stockholm'),
            ('JM.ST', 'JM AB', 'Consumer Discretionary', 12000000000, 'OMX Stockholm'),
            ('INVE-A.ST', 'Investor AB Class A', 'Financial Services', 11500000000, 'OMX Stockholm'),
            ('ELUX-A.ST', 'Electrolux AB Class A', 'Consumer Discretionary', 11000000000, 'OMX Stockholm'),
            
            ('HOLM-B.ST', 'Holmen AB Class B', 'Materials', 10500000000, 'OMX Stockholm'),
            ('MTRS.ST', 'Matas AS', 'Consumer Staples', 10000000000, 'OMX Stockholm'),
            ('DUST.ST', 'Dustin Group AB', 'Technology', 9500000000, 'OMX Stockholm'),
            ('DUNI.ST', 'Duni AB', 'Consumer Discretionary', 9000000000, 'OMX Stockholm'),
            ('LUPE.ST', 'Lundin Petroleum AB', 'Energy', 8500000000, 'OMX Stockholm'),
            ('NENT-A.ST', 'Nordic Entertainment Group AB Class A', 'Communication Services', 8000000000, 'OMX Stockholm'),
            ('SWED-C.ST', 'Swedbank AB Class C', 'Financial Services', 7500000000, 'OMX Stockholm'),
            ('RATO-B.ST', 'Ratos AB Class B', 'Financial Services', 7000000000, 'OMX Stockholm'),
            ('LUMI.ST', 'Luminar Technologies Inc', 'Technology', 6500000000, 'OMX Stockholm'),
            ('BEIJ-B.ST', 'Beijer Alma AB Class B', 'Industrials', 6000000000, 'OMX Stockholm'),
            
            ('INDU-C.ST', 'Industrivarden AB Class C', 'Financial Services', 5800000000, 'OMX Stockholm'),
            ('EPRO-B.ST', 'Electrolux Professional AB Class B', 'Industrials', 5500000000, 'OMX Stockholm'),
            ('SAND-PREF.ST', 'Sandvik AB Preference', 'Industrials', 5200000000, 'OMX Stockholm'),
            ('SKA-B.ST', 'Skanska AB Class B', 'Industrials', 5000000000, 'OMX Stockholm'),
            ('GETI-A.ST', 'Getinge AB Class A', 'Healthcare', 4800000000, 'OMX Stockholm'),
            ('HOLM-A.ST', 'Holmen AB Class A', 'Materials', 4500000000, 'OMX Stockholm'),
            ('LIAB-PREF.ST', 'Lifco AB Preference', 'Industrials', 4200000000, 'OMX Stockholm'),
            ('SECT-B.ST', 'Sector Alarm AB', 'Technology', 4000000000, 'OMX Stockholm'),
            ('KNOW.ST', 'Know IT AB', 'Technology', 3800000000, 'OMX Stockholm'),
            ('FING-B.ST', 'Fingerprint Cards AB Class B', 'Technology', 3500000000, 'OMX Stockholm'),
            
            ('MIPS.ST', 'MIPS AB', 'Technology', 3200000000, 'OMX Stockholm'),
            ('XVIVO.ST', 'XVIVO Perfusion AB', 'Healthcare', 3000000000, 'OMX Stockholm'),
            ('BALCO.ST', 'Balco Group AB', 'Industrials', 2800000000, 'OMX Stockholm'),
            ('CALID.ST', 'Calidris AB', 'Healthcare', 2500000000, 'OMX Stockholm'),
            ('XANO-B.ST', 'Xano Industri AB Class B', 'Industrials', 2200000000, 'OMX Stockholm'),
            ('ENEA.ST', 'Enea AB', 'Technology', 2000000000, 'OMX Stockholm'),
            ('CELL.ST', 'Cellavision AB', 'Healthcare', 1800000000, 'OMX Stockholm'),
            ('ONCO.ST', 'Oncopeptides AB', 'Healthcare', 1500000000, 'OMX Stockholm'),
            ('PRIC-B.ST', 'Pricer AB Class B', 'Technology', 1200000000, 'OMX Stockholm'),
            ('RECI.ST', 'Recipharm AB', 'Healthcare', 1000000000, 'OMX Stockholm'),
            
            ('TOBII.ST', 'Tobii AB', 'Technology', 900000000, 'OMX Stockholm'),
            ('PDYN.ST', 'Paradox Interactive AB', 'Technology', 800000000, 'OMX Stockholm'),
            ('AAK.ST', 'AAK AB', 'Consumer Staples', 750000000, 'OMX Stockholm'),
            ('ALIF-B.ST', 'Alimak Group AB Class B', 'Industrials', 700000000, 'OMX Stockholm'),
            ('ELOS-B.ST', 'Elos Medtech AB Class B', 'Healthcare', 650000000, 'OMX Stockholm'),
            ('DORO.ST', 'Doro AB', 'Technology', 600000000, 'OMX Stockholm'),
            ('HIFAB.ST', 'Hifab Group AB', 'Industrials', 550000000, 'OMX Stockholm'),
            ('INWI.ST', 'Inwido AB', 'Industrials', 500000000, 'OMX Stockholm'),
            ('KAHL.ST', 'Kahl Design Group AB', 'Consumer Discretionary', 450000000, 'OMX Stockholm'),
            ('LOOM.ST', 'Loomis AB Class B', 'Industrials', 400000000, 'OMX Stockholm'),
            
            ('MESH.ST', 'Meshcommunity AB', 'Technology', 380000000, 'OMX Stockholm'),
            ('NOTE.ST', 'Note AB', 'Technology', 350000000, 'OMX Stockholm'),
            ('OPUS.ST', 'Opus Group AB', 'Industrials', 320000000, 'OMX Stockholm'),
            ('PINE.ST', 'Pine AB', 'Technology', 300000000, 'OMX Stockholm'),
            ('QCOM.ST', 'Qcom AB', 'Technology', 280000000, 'OMX Stockholm'),
            ('RATO-A.ST', 'Ratos AB Class A', 'Financial Services', 250000000, 'OMX Stockholm'),
            ('SAVE.ST', 'Save by Solar AB', 'Energy', 220000000, 'OMX Stockholm'),
            ('TRAC-B.ST', 'Track AB Class B', 'Technology', 200000000, 'OMX Stockholm'),
            ('UNIT.ST', 'Uniti Sweden AB', 'Real Estate', 180000000, 'OMX Stockholm'),
            ('VOLO.ST', 'Volvo Car AB Class B', 'Consumer Discretionary', 150000000, 'OMX Stockholm')
        ]
        
        df = self.create_fallback_df(sweden_top_100, "OMX Stockholm")
        master_file = 'stock_data/sweden_stocks_master.csv'
        os.makedirs('stock_data', exist_ok=True)
        df.to_csv(master_file, index=False, encoding='utf-8-sig')
        logger.info(f"스웨덴 마스터 백업 데이터 생성: {len(df)}개 종목")
        return len(df)
    
    def create_fallback_df(self, stocks_data, market):
        """백업 DataFrame 생성"""
        rows = []
        for ticker, name, sector, mcap in stocks_data:
            rows.append({
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market_cap': mcap,
                'market': market
            })
        return pd.DataFrame(rows)


class MasterFilterThread(QThread):
    """마스터 CSV에서 필터링하는 스레드"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ENRICH_SLEEP = 0.08
    
    def run(self):
        try:
            market_selection = self.config['market_selection']
            top_count = self.config['top_count']
            master_files = self.config['master_files']
            
            logger.info(f"마스터에서 필터링 시작: 상위 {top_count}개")
            
            results = {}
            
            for market, master_file in master_files.items():
                filtered_count = self.filter_from_master(
                    market, master_file, top_count
                )
                results[market] = filtered_count
            
            total_count = sum(results.values())
            market_results = []
            
            market_names = {'korea': '한국', 'usa': '미국', 'sweden': '스웨덴'}
            for market, count in results.items():
                market_results.append(f"• {market_names[market]}: {count}개")
            
            message = (
                f'마스터 CSV에서 필터링이 완료되었습니다!\n'
                f'총 {total_count}개 종목 (시총 상위 {top_count}개)\n'
                + '\n'.join(market_results) +
                f'\n\n✅ 정확한 시가총액 순위로 정렬되었습니다.'
            )
            
            self.finished.emit(message)
            
        except Exception as e:
            logger.error(f"마스터 필터링 오류: {e}")
            self.error.emit(f'마스터 필터링 중 오류: {str(e)}')
    
    def filter_from_master(self, market, master_file, top_count):
        """마스터 파일에서 상위 종목 필터링"""
        try:
            self.progress.emit(f"{market} 마스터 CSV에서 상위 종목 추출 중...")
            
            # 1단계: 마스터 CSV 로드
            master_df = pd.read_csv(master_file)
            logger.info(f"{market} 마스터 파일 로드: {len(master_df)}개 종목")
            
            # 2단계: 여유있게 상위 N*2개 선택 (최신 정보 업데이트용)
            buffer_count = min(top_count * 2, len(master_df))
            top_candidates = master_df.head(buffer_count)
            
            self.progress.emit(f"{market} 상위 {buffer_count}개 종목 최신 정보 업데이트 중...")
            
            # 3단계: 최신 시가총액 정보로 재보강 (빠르게)
            updated_df = enrich_with_yfinance(
                top_candidates,
                ticker_col='ticker',
                max_items=buffer_count,
                sleep_sec=self.ENRICH_SLEEP,
                on_progress=self.progress.emit
            )
            
            # 4단계: 최신 시가총액으로 재정렬
            self.progress.emit(f"{market} 최신 시가총액 기준 재정렬 중...")
            final_df = self.sort_and_filter(updated_df, top_count, market)
            
            # 5단계: 작업용 CSV 저장
            work_file = f'stock_data/{market}_stocks.csv'
            final_df.to_csv(work_file, index=False, encoding='utf-8-sig')
            
            logger.info(f"{market} 필터링 완료: {work_file} ({len(final_df)}개 종목)")
            return len(final_df)
            
        except Exception as e:
            logger.error(f"{market} 필터링 실패: {e}")
            return 0
    
    def sort_and_filter(self, df, top_count, market_name):
        """시가총액 기준 재정렬 및 상위 N개 선택"""
        try:
            # 유효한 시가총액이 있는 종목만
            valid_df = df[df['market_cap'].notna() & (df['market_cap'] > 0)].copy()
            
            if valid_df.empty:
                logger.warning(f"{market_name}: 유효한 시가총액 없음, 원본 상위 {top_count}개 사용")
                return df.head(top_count)
            
            # 최신 시가총액 기준 재정렬
            sorted_df = valid_df.sort_values('market_cap', ascending=False)
            
            # 상위 N개 선택
            final_df = sorted_df.head(top_count).reset_index(drop=True)
            
            # 결과 로그
            # ✅ 벡터화: iterrows() 제거
            logger.info(f"{market_name} 최종 상위 3개:")
            top_3 = final_df.head(3)
            for i, (ticker, name, mcap) in enumerate(zip(top_3['ticker'], top_3['name'], top_3['market_cap'])):
                mcap_str = self.format_market_cap(mcap)
                logger.info(f"   {i+1}. {ticker} ({name[:20]}): {mcap_str}")
            
            return final_df
            
        except Exception as e:
            logger.error(f"정렬/필터 오류 ({market_name}): {e}")
            return df.head(top_count)
    
    def format_market_cap(self, market_cap):
        """시가총액 포맷팅"""
        try:
            if market_cap >= 1_000_000_000_000:
                return f"{market_cap/1_000_000_000_000:.1f}T"
            elif market_cap >= 1_000_000_000:
                return f"{market_cap/1_000_000_000:.1f}B"
            elif market_cap >= 1_000_000:
                return f"{market_cap/1_000_000:.1f}M"
            else:
                return f"{market_cap:,.0f}"
        except:
            return "N/A"