"""
enhanced_screener.py
AI 예측 기능이 강화된 스크리너 - 예측 함수 통합 버전

✅ 변경 사항:
- predict_stock_consistent() 함수를 predict_stock()으로 통합
- 기존 predict_stock() 함수 제거 (중복 제거)
- 더 나은 일관성과 정확도 제공
"""

import os
import sys
import json
import random
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial

# GUI 라이브러리 (데스크톱 버전용, 웹 배포 시에는 무시됨)
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# 로깅 설정
from logger_config import get_logger
logger = get_logger(__name__)

# 최적화 모듈 import
from cache_manager import get_stock_data, get_ticker_info, get_cache_instance

# 기본 라이브러리 가용성 확인
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ML 라이브러리 전체 확인
ML_AVAILABLE = SKLEARN_AVAILABLE and XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE

if ML_AVAILABLE:
    logger.info("모든 ML 라이브러리 사용 가능")
else:
    logger.warning("일부 ML 라이브러리 누락:")
    logger.warning(f"   - scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    logger.warning(f"   - XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    logger.warning(f"   - LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    logger.warning(f"   - statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌'}")


def to_scalar(value):
    """pandas Series/numpy 값을 스칼라로 안전하게 변환 - 개선 버전"""
    # 이미 스칼라면 그대로 반환
    if isinstance(value, (int, float, bool, np.integer, np.floating)):
        return float(value)
    
    # pandas Series 처리
    if isinstance(value, pd.Series):
        if len(value) == 0:
            return None
        # 첫 번째 값 추출
        value = value.iloc[0] if len(value) > 0 else value.values[0]
    
    # numpy array 처리
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.item() if value.size == 1 else value.flatten()[0]
    
    # .item() 메서드가 있으면 사용
    if hasattr(value, 'item'):
        try:
            return float(value.item())
        except:
            pass
    
    # 최후의 수단: 형변환
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except:
        return None

class EnhancedCPUPredictor:
    """CPU 최적화 예측기 - 통합된 예측 함수 버전 (싱글톤)"""

    _instance = None
    _initialized = False

    def __new__(cls):
        """싱글톤 패턴: 인스턴스가 없으면 생성, 있으면 재사용"""
        if cls._instance is None:
            cls._instance = super(EnhancedCPUPredictor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """CPU 최적화 모델들 초기화 (1회만 실행)"""
        # 이미 초기화되었으면 스킵
        if EnhancedCPUPredictor._initialized:
            return

        if not ML_AVAILABLE:
            logger.warning("ML 라이브러리가 부족합니다")
            self.models = {}
            self.scalers = {}
            EnhancedCPUPredictor._initialized = True
            return

        logger.info("CPU 최적화 예측기 초기화 중...")
        
        # 고정된 시드로 초기화
        self.fix_all_random_seeds(42)

        # ✅ 새로 추가: 캐싱 시스템
        self._data_cache = {}  # {ticker: (data, timestamp)}
        self._cache_duration = 3600  # 1시간 캐시 유지 (초 단위)
        self._feature_cache = {}  # 특성 계산 결과 캐싱

        self.load_settings()
        
        # CPU 스레드 설정 (기본값: 1 - 안전 모드)
        cpu_cores = self.settings.get('cpu_cores', 1)
        logger.info(f"모델 학습 스레드 제한: {cpu_cores}개 코어 사용")

        # CPU 최적화 모델들
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,        # 150 → 200
                max_depth=12,            # 8 → 12 ⭐
                learning_rate=0.1,       # 0.08 → 0.1
                subsample=0.9,           # 0.85 → 0.9
                colsample_bytree=0.9,    # 0.85 → 0.9
                reg_alpha=0.01,          # 0.1 → 0.01 ⭐ (정규화 완화)
                reg_lambda=0.01,         # 0.1 → 0.01 ⭐
                random_state=42,
                n_jobs=cpu_cores,
                verbosity=0,
                nthread=cpu_cores  # CPU 코어 수 동적 할당
            ),
            
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=12,            # 8 → 12 ⭐
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.01,          # 0.1 → 0.01 ⭐
                reg_lambda=0.01,         # 0.1 → 0.01 ⭐
                random_state=42,
                n_jobs=cpu_cores,
                num_threads=cpu_cores,  # CPU 코어 수 동적 할당
                device='cpu',
                verbose=-1
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=200,        # 150 → 200
                max_depth=15,            # 12 → 15 ⭐
                min_samples_split=2,     # 5 → 2 ⭐
                min_samples_leaf=1,      # 2 → 1 ⭐
                max_features=0.8,        # 0.7 → 0.8
                n_jobs=cpu_cores,
                random_state=42
            ),
            
            'extra_trees': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,            # 10 → 12 ⭐
                min_samples_split=2,     # 8 → 2 ⭐
                min_samples_leaf=1,      # 4 → 1 ⭐
                max_features=0.8,        # 0.6 → 0.8
                bootstrap=False,
                n_jobs=cpu_cores,
                random_state=42
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,             # 5 → 6 ⭐
                learning_rate=0.1,       # 0.08 → 0.1
                subsample=0.9,           # 0.85 → 0.9
                min_samples_split=2,     # 5 → 2 ⭐
                min_samples_leaf=1,      # 2 → 1 ⭐
                random_state=42,
                validation_fraction=0.1
            )
        }
        
        # 고급 전처리기들
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()  # 이상치에 강함
        }
        
        self.current_scaler = self.scalers['robust']  # 주식 데이터는 이상치 많음


        self.accuracy_history_file = 'prediction_accuracy_history.json'
        self.accuracy_history = self.load_accuracy_history()

        # 성능 추적 설정
        self.max_history_records = 1000  # 최대 기록 수
        self.accuracy_window_days = 30   # 정확도 평가 기간

        logger.info(f"{len(self.models)}개 모델 초기화 완료")

        # 초기화 완료 플래그 설정
        EnhancedCPUPredictor._initialized = True


    def get_model_config_for_period(self, forecast_days):
        """예측 기간에 따른 모델 설정 반환"""
        
        if forecast_days <= 5:
            # 단기 (1-5일): 빠른 반응, 단기 패턴
            return {
                'sequence_length': 10,
                'min_data_days': 200,
                'ma_periods': [5, 10, 20],
                'models': {
                    'xgboost': {
                        'n_estimators': 150,
                        'max_depth': 10,
                        'learning_rate': 0.1,
                        'reg_alpha': 0.01,
                        'reg_lambda': 0.01
                    },
                    'lightgbm': {
                        'n_estimators': 150,
                        'max_depth': 10,
                        'learning_rate': 0.1,
                        'reg_alpha': 0.01,
                        'reg_lambda': 0.01
                    },
                    'random_forest': {
                        'n_estimators': 150,
                        'max_depth': 12,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1
                    }
                }
            }
        
        elif forecast_days <= 14:
            # 중기 (6-14일): 균형잡힌 설정
            return {
                'sequence_length': 20,
                'min_data_days': 300,
                'ma_periods': [5, 10, 20, 50],
                'models': {
                    'xgboost': {
                        'n_estimators': 200,
                        'max_depth': 12,
                        'learning_rate': 0.08,
                        'reg_alpha': 0.05,
                        'reg_lambda': 0.05
                    },
                    'lightgbm': {
                        'n_estimators': 200,
                        'max_depth': 12,
                        'learning_rate': 0.08,
                        'reg_alpha': 0.05,
                        'reg_lambda': 0.05
                    },
                    'random_forest': {
                        'n_estimators': 200,
                        'max_depth': 15,
                        'min_samples_split': 3,
                        'min_samples_leaf': 2
                    }
                }
            }
        
        else:
            # 장기 (15-30일): 추세 중심, 장기 패턴
            return {
                'sequence_length': 30,
                'min_data_days': 400,
                'ma_periods': [10, 20, 50, 120, 200],
                'models': {
                    'xgboost': {
                        'n_estimators': 250,
                        'max_depth': 8,  # 과적합 방지
                        'learning_rate': 0.05,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1
                    },
                    'lightgbm': {
                        'n_estimators': 250,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.1
                    },
                    'random_forest': {
                        'n_estimators': 250,
                        'max_depth': 18,
                        'min_samples_split': 5,
                        'min_samples_leaf': 3
                    }
                }
            }

    def reconfigure_models(self, forecast_days):
        """예측 기간에 따라 모델 재구성"""
        config = self.get_model_config_for_period(forecast_days)
        cpu_cores = self.settings.get('cpu_cores', 1)

        logger.debug(f"{forecast_days}일 예측을 위한 모델 재구성:")
        logger.debug(f"   • 시퀀스 길이: {config['sequence_length']}일")
        logger.debug(f"   • 최소 데이터: {config['min_data_days']}일")
        logger.debug(f"   • MA 기간: {config['ma_periods']}")
        
        # 모델 재생성
        for model_name, params in config['models'].items():
            if model_name == 'xgboost':
                self.models['xgboost'] = xgb.XGBRegressor(
                    random_state=42, n_jobs=cpu_cores, nthread=cpu_cores, verbosity=0, **params
                )
            elif model_name == 'lightgbm':
                self.models['lightgbm'] = lgb.LGBMRegressor(
                    random_state=42, n_jobs=cpu_cores, num_threads=cpu_cores, device='cpu', verbose=-1, **params
                )
            elif model_name == 'random_forest':
                self.models['random_forest'] = RandomForestRegressor(
                    random_state=42, n_jobs=cpu_cores, **params
                )
        
        return config

    def load_settings(self):
        """✅ 새로 추가: 설정 파일에서 예측 설정 로드"""
        default_settings = {
            'forecast_days': 7,
            'confidence_threshold': 0.6,
            'batch_delay': 1.0,
            'min_data_days': 300,
            'use_arima_validation': True,
            'cpu_cores': 1,  # 기본값 1 (안전제일)
            'models_enabled': {
                'xgboost': True,
                'lightgbm': True,
                'random_forest': True,
                'extra_trees': True,
                'gradient_boosting': True
            }
        }
        
        try:
            settings_candidates = [
                'prediction_settings.json',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_settings.json'),
                '/app/parent/prediction_settings.json',
            ]
            settings_path = next((path for path in settings_candidates if os.path.exists(path)), None)

            if settings_path:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                default_settings.update(saved_settings)
                logger.info(f"설정 로드 완료: {settings_path} (예측기간 {saved_settings.get('forecast_days', 7)}일)")
            else:
                logger.warning("설정 파일 없음, 기본값 사용")
        except Exception as e:
            logger.error(f"설정 로드 오류: {e}, 기본값 사용")
        
        self.settings = default_settings

    def fix_all_random_seeds(self, seed=42):
        """모든 랜덤 시드 고정 - 완전한 일관성 보장"""
        logger.debug(f"모든 랜덤 시드를 {seed}로 고정")
        
        # Python 기본 random
        random.seed(seed)
        
        # NumPy random
        np.random.seed(seed)
        
        # 환경변수로 추가 고정
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # pandas random 상태도 고정
        try:
            pd.core.common.random_state(seed)
        except:
            pass

    def get_cached_data(self, ticker):
        """캐시된 데이터 가져오기"""
        if ticker in self._data_cache:
            data, timestamp = self._data_cache[ticker]
            elapsed_seconds = (datetime.now() - timestamp).total_seconds()

            if elapsed_seconds < self._cache_duration:
                logger.debug(f"캐시 사용: {ticker} (저장된 지 {int(elapsed_seconds)}초)")

                # ✅ 추가: 데이터 유효성 확인
                if data is not None and not data.empty and len(data) > 0:
                    return data
                else:
                    # 잘못된 캐시 데이터 삭제
                    logger.warning(f"잘못된 캐시 데이터 삭제: {ticker}")
                    del self._data_cache[ticker]
                    return None
            else:
                logger.debug(f"캐시 만료: {ticker} (저장된 지 {int(elapsed_seconds)}초)")
                del self._data_cache[ticker]
        
        return None
    
    def cache_data(self, ticker, data):
        """데이터 캐싱"""
        # ✅ 추가: 유효한 데이터만 캐싱
        if data is None or data.empty or len(data) == 0:
            logger.warning(f"잘못된 데이터, 캐싱 안 함: {ticker}")
            return

        self._data_cache[ticker] = (data.copy(), datetime.now())
        logger.debug(f"캐시 저장: {ticker} ({len(data)}개 데이터)")

    def _get_download_period_for_training(self, min_data_days, forecast_days, sequence_length):
        """필요 거래일을 확보하기 위한 yfinance 기간 계산."""
        trading_days_needed = int(min_data_days + forecast_days + sequence_length + 80)
        calendar_days = int(math.ceil(trading_days_needed * 365 / 252))

        if calendar_days <= 730:
            return f"{max(calendar_days, 365)}d"
        if calendar_days <= 1095:
            return "3y"
        if calendar_days <= 1825:
            return "5y"
        return "10y"

    def _get_minimum_usable_days(self, forecast_days, sequence_length):
        """예측을 완전히 중단해야 하는 최소 데이터 기준."""
        if forecast_days <= 5:
            floor = 120
        elif forecast_days <= 14:
            floor = 180
        else:
            floor = 220

        return max(floor, sequence_length + forecast_days + 60)
    
    def clear_cache(self):
        """캐시 전체 삭제 (메모리 정리용)
        
        예시:
            predictor.clear_cache()  # 배치 예측 후 메모리 정리
        """
        cache_count = len(self._data_cache)
        self._data_cache.clear()
        self._feature_cache.clear()
        logger.info(f"캐시 정리 완료: {cache_count}개 항목 삭제")

    # ✅ 통합된 예측 함수 - predict_stock_consistent의 로직을 predict_stock으로 변경
    def predict_stock(self, ticker, forecast_days=None, min_data_days=None, mode='smart'):
        # 설정 파일 값 우선 사용
        if forecast_days is None:
            forecast_days = self.settings.get('forecast_days', 7)
        
        # 예측 기간에 따라 모델 재구성
        config = self.reconfigure_models(forecast_days)
        
        # 설정에서 가져온 값 업데이트
        if min_data_days is None:
            min_data_days = config['min_data_days']

        settings_min_data_days = self.settings.get('min_data_days')
        if settings_min_data_days:
            min_data_days = min(min_data_days, int(settings_min_data_days))
        
        sequence_length = config['sequence_length']
        minimum_usable_days = self._get_minimum_usable_days(forecast_days, sequence_length)
        min_data_days = max(min_data_days, minimum_usable_days)
        configured_min_data_days = min_data_days
        limited_history = False

        logger.info(f"{ticker} 예측 시작:")
        logger.info(f"   • 예측 기간: {forecast_days}일 ({'단기' if forecast_days <= 5 else '중기' if forecast_days <= 14 else '장기'})")
        logger.info(f"   • 시퀀스: {sequence_length}일")

        confidence_threshold = getattr(self, 'settings', {}).get('confidence_threshold', 0.6)

        # 매번 시드 재고정 (완전한 일관성 보장) - 기존 코드 그대로
        self.fix_all_random_seeds(42)
        
        try:
            logger.info(f"{ticker} 일관성 예측 시작...")
            
            # 1. 실제 현재가 조회 (캐싱 사용, 검증 우회)
            current_data = get_stock_data(ticker, period="2d", validate_cache=False)

            # ✅ 수정: current_data 확인
            if current_data is None or current_data.empty or len(current_data) == 0:
                return None, "현재가 데이터를 가져올 수 없습니다"

            actual_current_price = float(current_data['Close'].iloc[-1])
            actual_current_date = current_data.index[-1]
            
            # 2. 캐시 확인 (내부 캐시)
            data = self.get_cached_data(ticker)

            if data is not None and len(data) < min_data_days:
                logger.info(f"캐시 데이터 부족: {len(data)}개 < {min_data_days}개, 더 긴 기간으로 재다운로드")
                data = None

            if data is None:
                logger.info(f"{ticker} 데이터 다운로드 중...")

                period_param = self._get_download_period_for_training(
                    min_data_days, forecast_days, sequence_length
                )
                logger.info(f"학습 데이터 기간 요청: {period_param} (목표 {min_data_days}거래일)")

                # 캐싱 매니저 사용 (중복 API 호출 방지)
                data = get_stock_data(ticker, period=period_param)

                # ✅ 수정: 데이터 확인
                if data is None or data.empty or len(data) == 0:
                    return None, f"{ticker} 데이터를 가져올 수 없습니다"

                # ✅ 내부 캐시에도 저장
                self.cache_data(ticker, data)
            else:
                # 캐시된 데이터 사용
                logger.info(f"캐시 데이터 사용: {len(data)}개 행")
            
            # 3. 데이터 길이 확인
            if len(data) < min_data_days:
                if len(data) >= minimum_usable_days:
                    logger.warning(f"제한적 이력으로 예측 진행: 권장 {min_data_days}일, 현재 {len(data)}일")
                    min_data_days = len(data)
                    limited_history = True
                else:
                    return None, (
                        f"데이터 부족 (권장: {configured_min_data_days}거래일, "
                        f"최소: {minimum_usable_days}거래일, 현재: {len(data)}거래일)"
                    )
            
            # 데이터 정렬 및 정리 (일관성 보장) - 기존 코드 그대로
            data = data.sort_index().round(4)
            
            null_count = to_scalar(data['Close'].isnull().sum())
            threshold = to_scalar(len(data) * 0.1)

            if null_count > threshold:
                return None, f"결측치가 너무 많습니다 ({null_count}개 > {threshold}개)"
                        
            # 시드 재고정 - 기존 코드 그대로
            self.fix_all_random_seeds(42)
            
            # 고급 특성 생성 - 기존 코드 그대로
            features = self.create_advanced_features_deterministic(data)
            
            if features.empty or features.isnull().all().all():
                return None, "특성 생성 실패"

            logger.debug(f"미래 수익률 계산 전 데이터 길이: {len(data)}")

            # ✅ 설정에서 가져온 forecast_days 사용 (수정된 부분)
            future_returns = data['Close'].pct_change(forecast_days).shift(-forecast_days)

            logger.debug(f"미래 수익률 계산 후:")
            logger.debug(f"     전체 길이: {len(future_returns)}")
            logger.debug(f"     유효 값: {future_returns.notna().sum()}개")
            logger.debug(f"     NaN: {future_returns.isna().sum()}개")

            # ✅ DataFrame이 아니라 Series로 유지
            if isinstance(future_returns, pd.DataFrame):
                future_returns = future_returns.iloc[:, 0]

            null_count = to_scalar(future_returns.isnull().sum())
            threshold = to_scalar(len(future_returns) * 0.8)

            if null_count > threshold:
                return None, f"미래 수익률 계산 실패 (결측치 {null_count}/{len(future_returns)}개)"
            
            # 시드 재고정 - 기존 코드 그대로
            self.fix_all_random_seeds(42)
            
            # 시퀀스 데이터 준비 - 기존 코드 그대로
            X, y = self.prepare_sequences_deterministic(features, future_returns, 
                                                    sequence_length=sequence_length, 
                                                    forecast_horizon=forecast_days)

            logger.debug(f"\n===== 데이터 진단 =====")
            logger.debug(f"X shape: {X.shape}")
            logger.debug(f"y shape: {y.shape}")
            logger.debug(f"y 통계:")
            logger.debug(f"     최소값: {y.min():.6f}")
            logger.debug(f"     최대값: {y.max():.6f}")
            logger.debug(f"     평균: {y.mean():.6f}")
            logger.debug(f"     표준편차: {y.std():.6f}")
            logger.debug(f"     중앙값: {np.median(y):.6f}")
            logger.debug(f"y 분포 샘플 (처음 10개): {y[:10]}")
            logger.debug(f"y 분포 샘플 (마지막 10개): {y[-10:]}")
            logger.debug(f"===========================\n")

            if len(X) == 0 or len(y) == 0:
                return None, "시퀀스 데이터 생성 실패"

            logger.info(f"데이터 준비 완료: {len(X)}개 학습 샘플")
            
            # ✅ 전체 데이터 학습으로 변경
            X_train = X
            y_train = y
            X_test = np.array([])  # 빈 배열
            y_test = np.array([])

            logger.debug(f"전체 데이터로 학습: {len(X_train)}개 샘플")

            # 데이터 정규화
            try:
                X_train_scaled = X_train
                X_test_scaled = X_test
                
                # 최신 데이터 준비 (예측용)
                latest_X = X[-1]
                latest_X_scaled = latest_X.reshape(1, -1)

                logger.debug(f"스케일링 제거됨 (Tree 기반 모델은 불필요)")
            except Exception as e:
                return None, f"데이터 준비 실패: {str(e)}"
            
            # 시드 재고정 - 기존 코드 그대로
            self.fix_all_random_seeds(42)
            
            # ✅ 모델별 예측 실행 (설정 반영 - 수정된 부분)
            predictions = []
            model_results = {}
            successful_models = 0
            
            # 설정에서 활성화된 모델만 사용
            models_enabled = self.settings.get('models_enabled', {})
            
            for model_name, model in self.models.items():
                if not models_enabled.get(model_name, True):
                    logger.debug(f"{model_name} 모델 비활성화됨 (설정)")
                    continue
                
                result = self.safe_predict_with_model(
                    model, X_train_scaled, y_train, X_test_scaled, y_test, latest_X_scaled, model_name
                )
                
                if result is not None:
                    prediction = result['prediction']
                    predictions.append(prediction)
                    successful_models += 1
                    
                    model_results[model_name] = {
                        'prediction': prediction,
                        'r2_score': result.get('r2_score', 0),
                        'mae': result.get('mae', 0)
                    }

            if successful_models == 0:
                return None, "모든 모델이 실패했습니다"

            logger.info(f"{successful_models}개 모델 성공 (설정 적용됨)")
            
            # 결정적 앙상블 계산 - 기존 코드 그대로
            ensemble_prediction, confidence = self.calculate_deterministic_ensemble(
                predictions, model_results
            )
            
            # 핵심 수정: 현재가 vs 예측가 분리 - 기존 코드 그대로
            historical_price = data['Close'].iloc[-1].item() # 예측 기준 가격
            predicted_return = float(ensemble_prediction)
            
            # 실제 현재가 기준으로 예측가 계산 - 기존 코드 그대로
            predicted_price_actual = actual_current_price * (1 + predicted_return)
            
            # ✅ 신뢰도 임계값 적용
            is_high_confidence = confidence >= confidence_threshold
            
            # ✅ 신뢰도에 따른 추천 결정
            if is_high_confidence:
                if predicted_return > 0.02:  # 2% 이상
                    recommendation = "🚀 매수 추천"
                    confidence_note = "높은 신뢰도"
                elif predicted_return < -0.02:  # -2% 이하
                    recommendation = "📉 매도 고려"
                    confidence_note = "높은 신뢰도"
                else:
                    recommendation = "⏸️ 관망"
                    confidence_note = "높은 신뢰도"
            else:
                # ✅ 낮은 신뢰도일 때 보수적 추천
                if predicted_return > 0.05:  # 5% 이상일 때만 보수적 매수
                    recommendation = "⚠️ 보수적 매수 고려"
                    confidence_note = "낮은 신뢰도 - 신중 판단 필요"
                elif predicted_return < -0.05:  # -5% 이하일 때만 보수적 매도
                    recommendation = "⚠️ 보수적 매도 고려"
                    confidence_note = "낮은 신뢰도 - 신중 판단 필요"
                else:
                    recommendation = "⚠️ 관망 권장"
                    confidence_note = "낮은 신뢰도 - 불확실한 예측"

            # ✅ 결과 구성 (설정 정보 추가)
            result = {
                'ticker': ticker,
                
                # 실제 현재가 정보 (사용자가 보는 정보) - 기존과 동일
                'current_price': round(actual_current_price, 4),
                'predicted_price': round(predicted_price_actual, 4),
                'expected_return': round(predicted_return, 6),
                
                # 예측 기술 정보 - 기존과 동일
                'confidence': round(confidence, 4),
                'forecast_days': forecast_days,  # ✅ 설정에서 가져온 값
                'days': forecast_days,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

                # ✅ 신뢰도 관련 정보 추가
                'confidence_threshold': confidence_threshold,
                'is_high_confidence': is_high_confidence,
                'recommendation': recommendation,
                'confidence_note': confidence_note,

                # 상세 정보 - 기존과 동일
                'successful_models': successful_models,
                'model_results': model_results,
                'individual_predictions': predictions,
                'feature_count': features.shape[1],
                'training_samples': len(X_train),
                'data_points': len(data),

                # ✅ 설정 정보 추가 (새로 추가된 부분)
                'min_data_days': min_data_days,  # 실제 사용된 최소 데이터 일수
                'recommended_min_data_days': configured_min_data_days,
                'minimum_usable_days': minimum_usable_days,
                'limited_history': limited_history,
                'active_models': [name for name, enabled in models_enabled.items() if enabled],
                'settings_applied': True,  # 설정 적용 여부 표시
                'settings_source': 'prediction_settings.json'  # 설정 출처
            }
            
            confidence_status = "높은 신뢰도" if is_high_confidence else "낮은 신뢰도"
            logger.info(f"예측 완료: {predicted_return*100:+.2f}% (신뢰도: {confidence*100:.1f}% - {confidence_status})")
                      
            return result, None
            
        except Exception as e:
            import traceback
            error_msg = f"예측 중 오류: {str(e)}"
            logger.error(f"{error_msg}")
            logger.error(f"상세 에러:")
            traceback.print_exc()  # ✅ 전체 스택 트레이스 출력
            return None, error_msg

    def safe_predict_with_model(self, model, X_train, y_train, X_test, y_test, X_predict, model_name):
        """개별 모델 예측 - 성능 평가 포함"""
        try:
            logger.debug(f"{model_name} 훈련 중...")

            # 입력 데이터 검증
            if X_train.size == 0 or y_train.size == 0:
                logger.error(f"{model_name} 오류: 빈 훈련 데이터")
                return None

            # 모델 훈련
            model.fit(X_train, y_train)

            # 학습 데이터 성능 확인 (과적합 진단)
            y_pred_train_sample = model.predict(X_train[-5:])  # 마지막 5개
            logger.debug(f"학습 데이터 마지막 5개 예측 평균: {y_pred_train_sample.mean()*100:+.2f}%")

            # ✅ 테스트 없이 바로 예측
            prediction = model.predict(X_predict)[0]

            if pd.isnull(prediction) or np.isinf(prediction):
                logger.error(f"{model_name} 오류: 잘못된 예측값")
                return None

            logger.info(f"{model_name} 완료: {prediction:.6f} ({prediction*100:+.2f}%)")

            return {
                'prediction': float(prediction),
                'r2_score': 0.0,  # R² 계산 안 함
                'mae': 0.0
            }

        except Exception as e:
            logger.error(f"{model_name} 오류: {str(e)}")
            return None

    def calculate_deterministic_ensemble(self, predictions, model_results):
        """성능 기반 동적 앙상블"""
        if not predictions:
            return 0.0, 0.0
        
        # ✅ R²에 따라 가중치 계산
        weights = {}
        total_r2 = 0
        
        for model_name, result in model_results.items():
            r2 = max(0, result.get('r2_score', 0))  # 음수 R²는 0으로
            weights[model_name] = r2
            total_r2 += r2
        
        # 정규화
        if total_r2 > 0:
            for model_name in weights:
                weights[model_name] /= total_r2
        else:
            # R²가 모두 0이면 균등 가중치
            equal_weight = 1.0 / len(predictions)
            weights = {name: equal_weight for name in model_results.keys()}
        
        # 가중 평균 계산
        weighted_sum = 0.0
        for model_name, result in model_results.items():
            weight = weights[model_name]
            prediction = result['prediction']
            weighted_sum += prediction * weight
        
        # 신뢰도 계산
        confidence = self.calculate_advanced_confidence(predictions, model_results)

        logger.debug(f"동적 가중치: {weights}")

        return weighted_sum, confidence

    # def calculate_advanced_confidence(self, predictions, model_results, market_conditions=None):
    #     """고급 신뢰도 계산 - 개선 버전"""
        
    #     # 1. 통계적 신뢰도 (모델 간 일치도)
    #     base_confidence = self.calculate_statistical_confidence(predictions)
        
    #     # 2. ✅ 모델 성능 신뢰도 (R² 기반)
    #     r2_scores = [r.get('r2_score', 0) for r in model_results.values()]
    #     if r2_scores:
    #         avg_r2 = np.mean([max(0, r2) for r2 in r2_scores])
    #         # R² 0.5 이상을 좋은 성능으로 간주
    #         performance_confidence = min(1.0, avg_r2 / 0.5 + 0.5)
    #     else:
    #         performance_confidence = 0.5
        
    #     # 3. 시장 상황
    #     if market_conditions is None:
    #         market_conditions = self.analyze_market_conditions(ticker=None, data=None)
    #     market_adjustment = self.calculate_market_confidence_adjustment(market_conditions)
        
    #     # 4. 역사적 성능
    #     historical_adjustment = self.calculate_historical_accuracy_adjustment()
        
    #     # 5. ✅ 가중치 조정 (성능 중시)
    #     final_confidence = (
    #         base_confidence * 0.25 +
    #         performance_confidence * 0.40 +  # 40%로 증가
    #         market_adjustment * 0.20 +
    #         historical_adjustment * 0.15
    #     )
        
    #     return max(0.1, min(0.95, final_confidence))

    def calculate_advanced_confidence(self, predictions, model_results, market_conditions=None):
        """신뢰도 계산 - 단순 버전"""
        
        # 1. 모델 간 일치도만 사용
        if len(predictions) <= 1:
            return 0.5
        
        # 예측값들의 변동계수 (낮을수록 일치도 높음)
        std = np.std(predictions)
        mean_pred = np.mean(predictions)
        
        if abs(mean_pred) > 1e-6:
            cv = abs(std / mean_pred)
            # CV가 0.5 이하면 신뢰도 높음
            base_confidence = 1.0 / (1.0 + cv * 2)
        else:
            base_confidence = 0.5
        
        # 2. 시장 상황 조정 (약한 영향)
        if market_conditions is None:
            market_conditions = self.analyze_market_conditions(ticker=None, data=None)
        
        market_adjustment = self.calculate_market_confidence_adjustment(market_conditions)
        
        # 3. 종합 (모델 일치도 80%, 시장 상황 20%)
        final_confidence = base_confidence * 0.8 + market_adjustment * 0.2
        
        return max(0.3, min(0.9, final_confidence))

    def calculate_statistical_confidence(self, predictions):
        """통계적 신뢰도 계산"""
        if len(predictions) <= 1:
            return 0.5
        
        # 예측값들의 표준편차
        std = np.std(predictions)
        mean_pred = np.mean(predictions)
        
        # 변동계수 (CV: Coefficient of Variation)
        if abs(mean_pred) > 1e-6:
            cv = abs(std / mean_pred)
            # CV가 작을수록 높은 신뢰도
            confidence = 1.0 / (1.0 + cv * 5)
        else:
            confidence = 0.5
        
        return confidence

    def calculate_performance_confidence(self, model_results):
        """모델 성능 기반 신뢰도"""
        if not model_results:
            return 0.5
        
        # R² 점수들 수집
        r2_scores = []
        for result in model_results.values():
            r2 = result.get('r2_score', 0)
            # R² 정규화: -∞~1 → 0~1
            normalized_r2 = max(0, min(1, (r2 + 0.5) / 1.5))
            r2_scores.append(normalized_r2)
        
        # 평균 성능
        avg_performance = np.mean(r2_scores)
        
        # 성능 일관성 (모든 모델이 비슷한 성능인지)
        performance_consistency = 1.0 - np.std(r2_scores)
        
        return (avg_performance * 0.7 + performance_consistency * 0.3)

    def get_market_data(self):
        """기본 시장 데이터 수집 (S&P 500 기준)"""
        try:
            logger.info("시장 데이터 수집 중...")
            
            # S&P 500 ETF (SPY) 데이터 사용
            spy = yf.download('SPY', period='6mo', progress=False, auto_adjust=True)
            
            if len(spy) < 50:
                logger.warning("SPY 데이터 부족, 기본값 사용")
                return self.get_default_market_data()
            
            # 기본 통계 계산
            current_price = spy['Close'].iloc[-1].item()
            ma20 = spy['Close'].rolling(20).mean().iloc[-1].item()
            ma50 = spy['Close'].rolling(50).mean().iloc[-1].item()
            volatility = spy['Close'].pct_change().rolling(20).std().iloc[-1].item()
            
            # VIX 가져오기
            try:
                vix = yf.download('^VIX', period='5d', progress=False, auto_adjust=True)
                current_vix = vix['Close'].iloc[-1].item() if len(vix) > 0 else 20.0
            except:
                current_vix = 20.0  # 기본값
            
            logger.info(f"시장 데이터 수집 완료: SPY=${current_price:.2f}, VIX={current_vix:.1f}")
            
            return {
                'spy': spy,
                'current_price': current_price,
                'ma20': ma20,
                'ma50': ma50,
                'volatility': volatility,
                'current_vix': current_vix,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.warning(f"시장 데이터 수집 오류: {e}")
            return self.get_default_market_data()

    def get_default_market_data(self):
        """기본 시장 데이터 (오류 시 사용)"""
        return {
            'spy': None,
            'current_price': 500.0,
            'ma20': 500.0,
            'ma50': 500.0,
            'volatility': 0.02,
            'current_vix': 20.0,
            'timestamp': datetime.now()
        }

    def calculate_trend_duration(self, spy_data):
        """추세 지속 기간 계산 (간단 버전)"""
        try:
            # 최근 50일 동안 상승/하락 추세가 지속된 기간 계산
            prices = spy_data['Close'].iloc[-50:]
            ma20 = spy_data['Close'].rolling(20).mean().iloc[-50:]
            
            # MA20 위/아래 있는 날짜 수 계산
            above_ma = (prices > ma20).sum()
            duration = int(above_ma) if above_ma > 25 else int(50 - above_ma)
            
            return max(1, min(50, duration))
        except:
            return 30  # 기본값

    def get_macro_conditions(self):
        """거시경제 정보 (간단 버전)"""
        return {
            'interest_rate_trend': 'stable',
            'economic_cycle': 'expansion',
            'inflation_trend': 'moderate'
        }

    def analyze_technical_indicators(self, market_data):
        """기술적 지표 분석 (간단 버전)"""
        try:
            spy_data = market_data.get('spy')
            if spy_data is None or len(spy_data) < 50:
                return {
                    'market_ma_position': 'neutral',
                    'market_momentum': 0.0,
                    'sector_rotation': False
                }
            
            current_price = spy_data['Close'].iloc[-1].item()
            ma50 = spy_data['Close'].rolling(50).mean().iloc[-1].item()
            
            # 시장 포지션
            if current_price > ma50 * 1.05:
                ma_position = 'strong_above'
            elif current_price > ma50:
                ma_position = 'above'
            elif current_price < ma50 * 0.95:
                ma_position = 'strong_below'
            else:
                ma_position = 'below'
            
            # 모멘텀 (최근 20일 수익률)
            momentum = (current_price / spy_data['Close'].iloc[-20].item() - 1)
            
            return {
                'market_ma_position': ma_position,
                'market_momentum': momentum,
                'sector_rotation': False  # 단순화
            }
            
        except Exception as e:
            logger.warning(f"기술적 지표 분석 오류: {e}")
            return {
                'market_ma_position': 'neutral',
                'market_momentum': 0.0,
                'sector_rotation': False
            }

    def analyze_market_conditions(self, ticker, data):
        """현재 시장 상황 분석"""
        try:
            # 1. 기본 시장 데이터 수집
            market_data = self.get_market_data()
            
            # 2. 시장 체제 분류
            regime = self.classify_market_regime(market_data)
            
            # 3. 변동성 분석
            volatility_info = self.analyze_volatility(market_data)
            
            # 4. 추세 분석
            trend_info = self.analyze_trend(market_data)
            
            # 5. 기술적 지표 분석
            technical_info = self.analyze_technical_indicators(market_data)
            
            # 6. 거시경제 정보 (선택적)
            macro_info = self.get_macro_conditions()
            
            return {
                'regime': regime,
                'volatility': volatility_info,
                'trend': trend_info,
                'technical': technical_info,
                'macro': macro_info,
                'timestamp': datetime.now(),
                'data_quality': 'high'  # high, medium, low
            }
            
        except Exception as e:
            logger.warning(f"시장 상황 분석 오류: {e}")
            # 기본값 반환
            return self.get_default_market_conditions()

    def classify_market_regime(self, market_data):
        """시장 체제 분류"""
        try:
            # S&P 500 또는 시장 지수 데이터 사용
            spy_data = yf.download('SPY', period='6mo', progress=False, auto_adjust=True)
            
            if len(spy_data) < 50:
                return 'unknown'
            
            # 최근 가격 추세 - scalar 값으로 변환
            recent_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-60] - 1).item()
            volatility = spy_data['Close'].pct_change().rolling(20).std().iloc[-1].item()
            
            # VIX 데이터 (가능한 경우)
            vix_level = self.get_vix_level()
            
            # 시장 체제 분류 로직
            if recent_return > 0.05 and volatility < 0.02 and vix_level < 20:
                return 'bull'
            elif recent_return < -0.05 and vix_level > 30:
                return 'bear'
            elif volatility > 0.03 or vix_level > 25:
                return 'volatile'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.warning(f"시장 체제 분류 오류: {e}")
            return 'sideways'  # 기본값

    def analyze_volatility(self, market_data):
        """변동성 분석"""
        try:
            # VIX 지수 조회
            vix_level = self.get_vix_level()
            
            # 과거 대비 변동성 백분위 계산
            spy_data = yf.download('SPY', period='1y', progress=False, auto_adjust=True)
            
            if len(spy_data) > 100:
                current_vol = spy_data['Close'].pct_change().rolling(20).std().iloc[-1].item()
                historical_vols = spy_data['Close'].pct_change().rolling(20).std().dropna()
                volatility_percentile = (historical_vols < current_vol).mean()
                
                # 변동성 추세
                recent_vols = historical_vols.tail(10)
                vol_first = recent_vols.iloc[0].item()
                vol_last = recent_vols.iloc[-1].item()
                
                if vol_last > vol_first * 1.2:
                    vol_trend = 'increasing'
                elif vol_last < vol_first * 0.8:
                    vol_trend = 'decreasing'
                else:
                    vol_trend = 'stable'
            else:
                volatility_percentile = 0.5
                vol_trend = 'stable'
            
            return {
                'current_vix': vix_level,
                'volatility_percentile': volatility_percentile,
                'trend': vol_trend
            }
            
        except Exception as e:
            logger.warning(f"변동성 분석 오류: {e}")
            return {
                'current_vix': 20.0,
                'volatility_percentile': 0.5,
                'trend': 'stable'
            }

    def get_vix_level(self):
        """VIX 지수 조회"""
        try:
            vix = yf.download('^VIX', period='5d', progress=False, auto_adjust=True)
            if len(vix) > 0:
                return vix['Close'].iloc[-1].item()
        except:
            pass
        return 20.0  # 기본값

    def analyze_trend(self, market_data):
        """추세 분석"""
        try:
            spy_data = yf.download('SPY', period='3mo', progress=False, auto_adjust=True)
            
            if len(spy_data) < 30:
                return {'direction': 'sideways', 'strength': 0.5, 'duration_days': 0}
            
            # 단기/장기 이동평균
            spy_data['MA20'] = spy_data['Close'].rolling(20).mean()
            spy_data['MA50'] = spy_data['Close'].rolling(50).mean()
            
            # ✅ .item() 추가
            current_price = spy_data['Close'].iloc[-1].item()
            ma20 = spy_data['MA20'].iloc[-1].item()
            ma50 = spy_data['MA50'].iloc[-1].item()
            
            # 추세 방향
            if current_price > ma20 > ma50:
                direction = 'upward'
                strength = min(1.0, (current_price / ma50 - 1) * 10)  # 정규화
            elif current_price < ma20 < ma50:
                direction = 'downward'
                strength = min(1.0, (1 - current_price / ma50) * 10)  # 정규화
            else:
                direction = 'sideways'
                strength = 0.5
            
            # 추세 지속 기간 (간단한 계산)
            duration_days = self.calculate_trend_duration(spy_data)
            
            return {
                'direction': direction,
                'strength': max(0.1, min(0.9, strength)),
                'duration_days': duration_days
            }
            
        except Exception as e:
            logger.warning(f"추세 분석 오류: {e}")
            return {'direction': 'sideways', 'strength': 0.5, 'duration_days': 30}

    def calculate_market_confidence_adjustment(self, market_conditions):
        """시장 상황 기반 신뢰도 조정"""
        if not market_conditions:
            return 0.8  # 기본값
        
        base_confidence = 0.8
        
        # 1. 시장 체제별 조정
        regime_adjustments = {
            'bull': +0.1,      # 상승장에서 예측이 더 신뢰할만함
            'bear': -0.05,     # 하락장에서 예측 어려움
            'sideways': 0.0,   # 보합장은 중립
            'volatile': -0.15, # 변동성 높을 때 예측 어려움
            'unknown': -0.1    # 불확실할 때 보수적
        }
        
        regime = market_conditions.get('regime', 'unknown')
        base_confidence += regime_adjustments.get(regime, 0)
        
        # 2. 변동성 기반 조정
        volatility_info = market_conditions.get('volatility', {})
        vix_level = volatility_info.get('current_vix', 20)
        
        if vix_level < 15:
            base_confidence += 0.05  # 낮은 변동성 = 높은 신뢰도
        elif vix_level > 30:
            base_confidence -= 0.1   # 높은 변동성 = 낮은 신뢰도
        
        # 3. 추세 강도 기반 조정
        trend_info = market_conditions.get('trend', {})
        trend_strength = trend_info.get('strength', 0.5)
        
        if trend_strength > 0.7:
            base_confidence += 0.05  # 강한 추세 = 예측하기 쉬움
        elif trend_strength < 0.3:
            base_confidence -= 0.05  # 약한 추세 = 예측 어려움
        
        # 최종 범위 제한
        return max(0.1, min(0.9, base_confidence))

    def get_default_market_conditions(self):
        """기본 시장 상황 (오류 시 사용)"""
        return {
            'regime': 'sideways',
            'volatility': {
                'current_vix': 20.0,
                'volatility_percentile': 0.5,
                'trend': 'stable'
            },
            'trend': {
                'direction': 'sideways',
                'strength': 0.5,
                'duration_days': 30
            },
            'technical': {
                'market_ma_position': 'neutral',
                'market_momentum': 0.0,
                'sector_rotation': False
            },
            'macro': {
                'interest_rate_trend': 'stable',
                'economic_cycle': 'expansion',
                'inflation_trend': 'moderate'
            },
            'timestamp': datetime.now(),
            'data_quality': 'medium'
        }


    def load_accuracy_history(self):
        """과거 예측 성능 기록 로드"""
        try:
            if os.path.exists(self.accuracy_history_file):
                with open(self.accuracy_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    logger.info(f"과거 성능 기록 로드: {len(history)}건")
                    return history
            else:
                logger.info("새로운 성능 추적 시작")
                return []
        except Exception as e:
            logger.warning(f"성능 기록 로드 오류: {e}")
            return []

    def save_accuracy_history(self):
        """성능 기록 저장"""
        try:
            # 최대 기록 수 제한
            if len(self.accuracy_history) > self.max_history_records:
                self.accuracy_history = self.accuracy_history[-self.max_history_records:]
            
            with open(self.accuracy_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.accuracy_history, f, indent=2, ensure_ascii=False)
            logger.info(f"성능 기록 저장: {len(self.accuracy_history)}건")
        except Exception as e:
            logger.warning(f"성능 기록 저장 오류: {e}")

    def record_prediction(self, ticker, prediction_data):
        """예측 기록 저장 - 나중에 정확도 평가용"""
        try:
            record = {
                'ticker': ticker,
                'prediction_date': datetime.now().isoformat(),
                'forecast_days': prediction_data.get('forecast_days', 7),
                'predicted_return': prediction_data.get('expected_return', 0),
                'predicted_price': prediction_data.get('predicted_price', 0),
                'current_price': prediction_data.get('current_price', 0),
                'confidence': prediction_data.get('confidence', 0),
                'market_conditions': prediction_data.get('market_conditions', {}),
                'models_used': prediction_data.get('active_models', []),
                
                # 나중에 실제 결과로 업데이트될 필드들
                'actual_price': None,
                'actual_return': None,
                'accuracy_score': None,
                'evaluation_date': None,
                'is_evaluated': False
            }
            
            self.accuracy_history.append(record)
            
            # 주기적으로 저장 (10개마다)
            if len(self.accuracy_history) % 10 == 0:
                self.save_accuracy_history()

            logger.info(f"예측 기록 저장: {ticker}")

        except Exception as e:
            logger.warning(f"예측 기록 오류: {e}")

    def evaluate_past_predictions(self):
        """과거 예측들의 실제 결과 평가"""
        try:
            evaluated_count = 0
            
            for record in self.accuracy_history:
                if record['is_evaluated']:
                    continue
                    
                # 예측 후 충분한 시간이 지났는지 확인
                prediction_date = datetime.fromisoformat(record['prediction_date'])
                forecast_days = record['forecast_days']
                target_date = prediction_date + timedelta(days=forecast_days)
                
                if datetime.now() >= target_date:
                    # 실제 결과 조회 및 평가
                    success = self.evaluate_single_prediction(record)
                    if success:
                        evaluated_count += 1
            
            if evaluated_count > 0:
                logger.info(f"{evaluated_count}개 과거 예측 평가 완료")
                self.save_accuracy_history()

        except Exception as e:
            logger.warning(f"과거 예측 평가 오류: {e}")

    def evaluate_single_prediction(self, record):
        """개별 예측 기록 평가"""
        try:
            ticker = record['ticker']
            prediction_date = datetime.fromisoformat(record['prediction_date'])
            forecast_days = record['forecast_days']
            target_date = prediction_date + timedelta(days=forecast_days + 5)  # 여유 기간
            
            # 실제 주가 데이터 조회 (캐싱 사용)
            # 예측일부터 목표일까지의 기간 계산
            days_diff = (target_date - prediction_date).days + 10
            period_str = f"{days_diff}d"

            actual_data = get_stock_data(ticker, period=period_str)
            
            if len(actual_data) < forecast_days:
                return False  # 데이터 부족
            
            # 실제 결과 계산
            actual_price = float(actual_data['Close'].iloc[min(forecast_days, len(actual_data)-1)])
            initial_price = record['current_price']
            actual_return = (actual_price / initial_price - 1) if initial_price > 0 else 0
            
            # 정확도 점수 계산
            predicted_return = record['predicted_return']
            accuracy_score = self.calculate_prediction_accuracy(predicted_return, actual_return)
            
            # 기록 업데이트
            record['actual_price'] = actual_price
            record['actual_return'] = actual_return
            record['accuracy_score'] = accuracy_score
            record['evaluation_date'] = datetime.now().isoformat()
            record['is_evaluated'] = True
            
            logger.info(f"{ticker} 예측 평가: 예측{predicted_return*100:+.1f}% vs 실제{actual_return*100:+.1f}% (정확도: {accuracy_score:.2f})")

            return True

        except Exception as e:
            logger.warning(f"{record.get('ticker', 'N/A')} 평가 오류: {e}")
            return False

    def calculate_prediction_accuracy(self, predicted_return, actual_return):
        """예측 정확도 점수 계산"""
        try:
            # 1. 방향 정확도 (상승/하락 방향이 맞는지)
            direction_correct = (predicted_return * actual_return > 0) or (abs(predicted_return) < 0.01 and abs(actual_return) < 0.01)
            direction_score = 1.0 if direction_correct else 0.0
            
            # 2. 크기 정확도 (예측 크기가 얼마나 정확한지)
            magnitude_error = abs(predicted_return - actual_return)
            magnitude_score = max(0, 1.0 - magnitude_error * 10)  # 10% 차이에서 0점
            
            # 3. 종합 점수 (방향 60%, 크기 40%)
            total_score = direction_score * 0.6 + magnitude_score * 0.4
            
            return max(0.0, min(1.0, total_score))

        except Exception as e:
            logger.warning(f"정확도 계산 오류: {e}")
            return 0.5  # 기본값

    def calculate_historical_accuracy_adjustment(self):
        """과거 예측 성능 기반 신뢰도 조정"""
        try:
            # 우선 과거 예측들 평가
            self.evaluate_past_predictions()
            
            if not self.accuracy_history:
                logger.info("과거 성능 데이터 없음 - 기본값 사용")
                return 0.8  # 기본값

            # 평가된 기록들만 필터링
            evaluated_records = [r for r in self.accuracy_history if r.get('is_evaluated', False)]

            if len(evaluated_records) < 5:
                logger.info(f"평가된 기록 부족 ({len(evaluated_records)}개) - 기본값 사용")
                return 0.8
            
            # 1. 전체 정확도 계산
            overall_accuracy = self.calculate_overall_accuracy(evaluated_records)
            
            # 2. 최근 성능 추세 계산
            recent_trend = self.calculate_recent_performance_trend(evaluated_records)
            
            # 3. 시장 상황별 성능 계산
            contextual_performance = self.calculate_contextual_performance(evaluated_records)
            
            # 4. 모델별 성능 계산
            model_performance = self.calculate_model_specific_performance(evaluated_records)
            
            # 5. 종합 조정값 계산
            adjustment = (
                overall_accuracy * 0.4 +
                recent_trend * 0.3 +
                contextual_performance * 0.2 +
                model_performance * 0.1
            )
            
            logger.info(f"역사적 성능 조정: {adjustment:.3f} (기록 {len(evaluated_records)}개 기반)")

            # 합리적 범위로 제한
            return max(0.3, min(1.0, adjustment))

        except Exception as e:
            logger.warning(f"역사적 성능 계산 오류: {e}")
            return 0.8  # 기본값

    def calculate_overall_accuracy(self, evaluated_records):
        """전체 정확도 계산"""
        try:
            accuracy_scores = [r['accuracy_score'] for r in evaluated_records if r.get('accuracy_score') is not None]
            
            if not accuracy_scores:
                return 0.8
                
            # 가중평균 (최근 것에 더 높은 가중치)
            weights = [i + 1 for i in range(len(accuracy_scores))]  # 1, 2, 3, ...
            weighted_avg = sum(score * weight for score, weight in zip(accuracy_scores, weights)) / sum(weights)
            
            return weighted_avg

        except Exception as e:
            logger.warning(f"전체 정확도 계산 오류: {e}")
            return 0.8

    def calculate_recent_performance_trend(self, evaluated_records):
        """최근 성능 추세 계산"""
        try:
            # 최근 20개 기록만 사용
            recent_records = evaluated_records[-20:] if len(evaluated_records) >= 20 else evaluated_records
            
            if len(recent_records) < 5:
                return 0.8
            
            # 시간순 정렬
            recent_records.sort(key=lambda x: x['prediction_date'])
            
            # 최근 성능 점수들
            recent_scores = [r['accuracy_score'] for r in recent_records if r.get('accuracy_score') is not None]
            
            if len(recent_scores) < 5:
                return 0.8
            
            # 추세 계산 (선형 회귀)
            x = list(range(len(recent_scores)))
            y = recent_scores
            
            # 간단한 추세 계산
            if len(y) >= 2:
                trend_slope = (y[-1] - y[0]) / (len(y) - 1)
                base_performance = sum(recent_scores) / len(recent_scores)
                
                # 추세를 반영한 조정
                trend_adjustment = base_performance + trend_slope * 2  # 추세 강화
                return max(0.3, min(1.0, trend_adjustment))
            else:
                return sum(recent_scores) / len(recent_scores)

        except Exception as e:
            logger.warning(f"최근 추세 계산 오류: {e}")
            return 0.8

    def calculate_contextual_performance(self, evaluated_records):
        """시장 상황별 성능 계산"""
        try:
            # 현재 시장 상황 분석
            current_market = self.analyze_market_conditions(None, None)
            current_regime = current_market.get('regime', 'sideways')
            
            # 비슷한 시장 상황에서의 과거 성능 찾기
            similar_context_records = []
            for record in evaluated_records:
                record_market = record.get('market_conditions', {})
                record_regime = record_market.get('regime', 'unknown')
                
                if record_regime == current_regime:
                    similar_context_records.append(record)
            
            if len(similar_context_records) >= 3:
                # 비슷한 상황에서의 성능
                context_scores = [r['accuracy_score'] for r in similar_context_records if r.get('accuracy_score') is not None]
                context_performance = sum(context_scores) / len(context_scores)
                logger.info(f"{current_regime} 시장에서 과거 성능: {context_performance:.3f} ({len(context_scores)}건)")
                return context_performance
            else:
                # 전체 평균 사용
                all_scores = [r['accuracy_score'] for r in evaluated_records if r.get('accuracy_score') is not None]
                return sum(all_scores) / len(all_scores) if all_scores else 0.8

        except Exception as e:
            logger.warning(f"상황별 성능 계산 오류: {e}")
            return 0.8

    def calculate_model_specific_performance(self, evaluated_records):
        """모델별 성능 계산"""
        try:
            # 현재 활성화된 모델들
            current_models = set(self.settings.get('models_enabled', {}).keys())
            
            # 각 모델 조합별 성능 계산
            model_performances = {}
            
            for record in evaluated_records:
                record_models = set(record.get('models_used', []))
                
                # 모델 세트를 키로 사용 (정렬하여 일관성 보장)
                model_key = ','.join(sorted(record_models))
                
                if model_key not in model_performances:
                    model_performances[model_key] = []
                
                if record.get('accuracy_score') is not None:
                    model_performances[model_key].append(record['accuracy_score'])
            
            # 현재 모델 조합과 가장 유사한 성능 찾기
            current_model_key = ','.join(sorted(current_models))
            
            if current_model_key in model_performances and len(model_performances[current_model_key]) >= 3:
                # 정확히 같은 모델 조합
                scores = model_performances[current_model_key]
                return sum(scores) / len(scores)
            else:
                # 비슷한 모델 조합 또는 전체 평균
                all_performances = []
                for performances in model_performances.values():
                    all_performances.extend(performances)
                
                return sum(all_performances) / len(all_performances) if all_performances else 0.8

        except Exception as e:
            logger.warning(f"모델별 성능 계산 오류: {e}")
            return 0.8

    def create_advanced_features_deterministic(self, data, ma_periods=None):
        """결정적 고급 특성 생성 - MA 기간 동적"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # MA 기간이 지정되지 않으면 기본값
            if ma_periods is None:
                ma_periods = [5, 10, 20, 50]

            logger.debug(f"MA 기간 사용: {ma_periods}")
            
            # 1. 기본 수익률
            features['returns'] = data['Close'].pct_change()
            features['returns_2'] = data['Close'].pct_change(2)
            features['returns_5'] = data['Close'].pct_change(5)
            
            # 2. 이동평균 기반 특성 (동적 기간)
            for window in ma_periods:
                ma = data['Close'].rolling(window, min_periods=1).mean()
                features[f'ma_{window}_ratio'] = data['Close'] / ma
                features[f'ma_{window}_slope'] = ma.pct_change(5)
                
                # 3. 볼륨 기반 특성
                volume_ma_20 = data['Volume'].rolling(20, min_periods=1).mean()
                features['volume_ratio'] = data['Volume'] / volume_ma_20.replace(0, 1)
                features['price_volume'] = features['returns'] * features['volume_ratio']
                
                # 4. 변동성 특성 (고정 윈도우)
                features['volatility_20'] = features['returns'].rolling(20, min_periods=1).std()
                features['volatility_5'] = features['returns'].rolling(5, min_periods=1).std()
                
                # 5. 가격 위치 특성 (고정 윈도우)
                for window in [14, 20]:
                    high_max = data['High'].rolling(window, min_periods=1).max()
                    low_min = data['Low'].rolling(window, min_periods=1).min()
                    price_range = high_max - low_min
                    features[f'price_position_{window}'] = (data['Close'] - low_min) / price_range.replace(0, 1)
                
                # 6. 모멘텀 지표들 (고정 파라미터)
                # RSI (14일 고정)
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
                rs = gain / loss.replace(0, 1)
                features['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD (12, 26, 9 고정)
                ema_12 = data['Close'].ewm(span=12).mean()
                ema_26 = data['Close'].ewm(span=26).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9).mean()
                features['macd_histogram'] = features['macd'] - features['macd_signal']
                
                # 7. 시간 특성 (결정적)
                features['trend'] = np.arange(len(data), dtype=float)
                features['day_of_week'] = data.index.dayofweek.astype(float)
                features['month'] = data.index.month.astype(float)
                
                # 8. 랙 특성들 (고정 랙)
                for lag in [1, 2, 3, 5]:
                    features[f'price_lag_{lag}'] = data['Close'].shift(lag) / data['Close']
                    features[f'volume_lag_{lag}'] = data['Volume'].shift(lag) / data['Volume'].replace(0, 1)

                # 9. 추가 기술적 지표
                # Stochastic Oscillator
                low_14 = data['Low'].rolling(14).min()
                high_14 = data['High'].rolling(14).max()
                features['stochastic_k'] = (data['Close'] - low_14) / (high_14 - low_14 + 1e-10)
                features['stochastic_d'] = features['stochastic_k'].rolling(3).mean()

                # Williams %R
                features['williams_r'] = (high_14 - data['Close']) / (high_14 - low_14 + 1e-10)

                # Average True Range (ATR)
                tr1 = data['High'] - data['Low']
                tr2 = abs(data['High'] - data['Close'].shift(1))
                tr3 = abs(data['Low'] - data['Close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                features['atr_14'] = tr.rolling(14).mean()

                # 10. 거래량 가중 가격
                try:
                    # Close와 Volume을 명시적으로 Series로 변환
                    if isinstance(data['Close'], pd.DataFrame):
                        close_series = data['Close'].iloc[:, 0]
                    else:
                        close_series = data['Close']

                    if isinstance(data['Volume'], pd.DataFrame):
                        volume_series = data['Volume'].iloc[:, 0]
                    else:
                        volume_series = data['Volume']

                    # VWAP 계산
                    vwap_numerator = (close_series * volume_series).rolling(20).sum()
                    vwap_denominator = volume_series.rolling(20).sum()
                    vwap_calc = vwap_numerator / vwap_denominator

                    # 결과가 DataFrame이면 Series로 변환
                    if isinstance(vwap_calc, pd.DataFrame):
                        vwap_calc = vwap_calc.iloc[:, 0]

                    features['vwap'] = vwap_calc

                    # vwap_ratio 계산
                    vwap_ratio_calc = close_series / vwap_calc

                    # 결과가 DataFrame이면 Series로 변환
                    if isinstance(vwap_ratio_calc, pd.DataFrame):
                        vwap_ratio_calc = vwap_ratio_calc.iloc[:, 0]

                    features['vwap_ratio'] = vwap_ratio_calc

                except Exception as vwap_error:
                    logger.debug(f"VWAP 계산 오류 (스킵): {vwap_error}")
                    # VWAP 실패 시 기본값 사용
                    features['vwap'] = close_series
                    features['vwap_ratio'] = 1.0

                # 데이터 정리 (결정적 방식)
                features = features.replace([np.inf, -np.inf], 0)
                features = features.ffill().bfill().fillna(0)

                logger.info(f"{len(features.columns)}개 결정적 특성 생성 완료")
                return features

        except Exception as e:
            logger.error(f"특성 생성 오류: {e}")
            # 최소한의 특성 생성
            features = pd.DataFrame(index=data.index)
            features['returns'] = data['Close'].pct_change().fillna(0)
            features['trend'] = np.arange(len(data), dtype=float)
            return features

    def prepare_sequences_deterministic(self, features, targets, sequence_length=15, forecast_horizon=7):
        """결정적 시퀀스 데이터 준비 - DataFrame 처리 버전"""
        try:
            # ✅ 수정 1: targets가 DataFrame이면 Series로 자동 변환
            if isinstance(targets, pd.DataFrame):
                logger.debug(f"targets DataFrame을 Series로 자동 변환 중...")
                if targets.shape[1] == 1:
                    targets = targets.iloc[:, 0]  # 첫 번째 컬럼을 Series로
                else:
                    logger.error(f"targets에 여러 컬럼이 있습니다: {targets.columns}")
                    return np.array([]), np.array([])

            logger.debug(f"targets 변환 후 타입: {type(targets)}")
            logger.debug(f"targets NaN 개수: {targets.isna().sum()}/{len(targets)}")
            
            # 유효한 데이터 필터링
            targets_valid = pd.notna(targets)
            features_valid = features.notna().all(axis=1)
            valid_indices = targets_valid & features_valid
            
            valid_features = features[valid_indices].copy()
            valid_targets = targets[valid_indices].copy()

            logger.debug(f"필터링 후 유효 데이터: {len(valid_targets)}개")

            # 데이터 길이 확인
            min_required = sequence_length + forecast_horizon
            if len(valid_features) < min_required:
                logger.error(f"유효 데이터 부족: {len(valid_features)}개 < {min_required}개 필요")
                return np.array([]), np.array([])
            
            # ✅ 수정 2: numpy array로 변환 (Series 문제 완전 회피)
            valid_features_array = valid_features.values
            valid_targets_array = valid_targets.values
            
            # targets가 2차원이면 1차원으로
            if len(valid_targets_array.shape) > 1:
                valid_targets_array = valid_targets_array.flatten()
            
            X, y = [], []
            success_count = 0
            fail_count = 0
            
            # ✅ 수정 3: numpy array로 직접 접근
            for i in range(sequence_length, len(valid_features_array)):
                try:
                    # numpy array이므로 직접 float 변환
                    target_value = float(valid_targets_array[i])
                    
                    # NaN/inf 체크
                    if not np.isfinite(target_value):
                        fail_count += 1
                        continue
                    
                    # sequence 데이터
                    sequence = valid_features_array[i-sequence_length:i]
                    
                    # 시퀀스 유효성 확인
                    if not np.isfinite(sequence).all():
                        fail_count += 1
                        continue
                    
                    # 추가
                    X.append(sequence.flatten())
                    y.append(target_value)
                    success_count += 1
                    
                except Exception as e:
                    fail_count += 1
                    continue
            
            logger.debug(f"시퀀스 생성 결과: 성공 {success_count}개, 실패 {fail_count}개")

            # 결과 확인
            if len(X) == 0 or len(y) == 0:
                logger.error(f"유효한 시퀀스 생성 실패")
                return np.array([]), np.array([])

            X_array = np.array(X, dtype=np.float64)
            y_array = np.array(y, dtype=np.float64)

            logger.info(f"시퀀스 데이터 준비 완료: {len(X_array)}개 샘플, 특성 {X_array.shape[1]}개")

            return X_array, y_array

        except Exception as e:
            logger.error(f"시퀀스 데이터 준비 전체 오류: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([])

    def _backtest_single_period(self, data, test_idx, test_periods, forecast_days, ticker):
        """단일 백테스팅 기간 처리 (병렬 처리용 워커 함수)"""
        try:
            # 예측 시점 설정 (뒤에서부터 역순으로)
            prediction_point = len(data) - (test_periods - test_idx) * forecast_days - forecast_days

            if prediction_point < 300:
                return None

            # 예측 시점까지의 데이터만 사용
            train_data = data.iloc[:prediction_point].copy()

            # 실제 미래 가격 (정답)
            actual_future_point = prediction_point + forecast_days
            if actual_future_point >= len(data):
                return None

            # FutureWarning 방지: .iloc[0] 사용
            actual_price_val = data['Close'].iloc[actual_future_point]
            actual_price = float(actual_price_val.iloc[0] if isinstance(actual_price_val, pd.Series) else actual_price_val)

            current_price_val = train_data['Close'].iloc[-1]
            current_price = float(current_price_val.iloc[0] if isinstance(current_price_val, pd.Series) else current_price_val)

            actual_return = (actual_price / current_price - 1)

            prediction_date = train_data.index[-1]

            # 예측 실행 (과거 시점에서)
            predicted_return = self.predict_with_historical_data(
                train_data, forecast_days
            )

            if predicted_return is None:
                logger.debug(f"테스트 {test_idx+1}: 예측 실패")
                return None

            predicted_price = current_price * (1 + predicted_return)

            # 정확도 계산
            direction_correct = (predicted_return * actual_return > 0)
            magnitude_error = abs(predicted_return - actual_return)

            result = {
                'test_idx': test_idx,
                'date': prediction_date,
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'predicted_return': float(predicted_return),
                'actual_price': float(actual_price),
                'actual_return': float(actual_return),
                'direction_correct': direction_correct,
                'magnitude_error': float(magnitude_error),
                'accuracy_score': 1.0 if direction_correct else 0.0
            }

            logger.debug(f"테스트 {test_idx+1}: {prediction_date.strftime('%Y-%m-%d')} - "
                        f"예측 {predicted_return*100:+.2f}% vs 실제 {actual_return*100:+.2f}%")

            return result

        except Exception as e:
            logger.warning(f"테스트 {test_idx+1} 오류: {e}")
            return None

    def backtest_predictions(self, ticker, test_periods=10, forecast_days=7, progress_callback=None, use_parallel=True, max_workers=None, cancel_callback=None):
        """
        과거 데이터로 예측 알고리즘 검증 (병렬 처리 지원)

        Args:
            ticker: 종목 코드
            test_periods: 테스트할 기간 수 (예: 10 = 10번 예측)
            forecast_days: 예측 기간
            progress_callback: 진행률 콜백 함수 (current, total, message)
            use_parallel: 병렬 처리 사용 여부 (기본: True)
            max_workers: 최대 워커 수 (None이면 CPU 코어 수)
            cancel_callback: 중지 확인 콜백 함수 (True 반환 시 중지)

        Returns:
            검증 결과 딕셔너리
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"{ticker} 백테스팅 시작")
        logger.info(f"   • 테스트 기간: {test_periods}회")
        logger.info(f"   • 예측 기간: {forecast_days}일")
        logger.info(f"   • 병렬 처리: {'활성화' if use_parallel else '비활성화'}")
        logger.info(f"{'='*60}\n")

        # 전체 데이터 다운로드
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2년 데이터

        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if len(data) < 300:
            return None, "데이터 부족"

        results = []

        if use_parallel:
            # 병렬 처리 모드
            logger.info(f"🚀 병렬 처리 모드 시작 (워커: {max_workers or 'CPU 코어 수'})")

            # ThreadPoolExecutor 사용 (ML 모델은 GIL 영향을 덜 받음)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 모든 테스트 작업 제출
                future_to_idx = {}
                for i in range(test_periods):
                    future = executor.submit(
                        self._backtest_single_period,
                        data, i, test_periods, forecast_days, ticker
                    )
                    future_to_idx[future] = i

                # 완료된 작업부터 결과 수집
                completed_count = 0
                for future in as_completed(future_to_idx):
                    # 중지 확인
                    if cancel_callback and cancel_callback():
                        logger.info("⏹ 병렬 백테스팅 중지 요청 - 남은 작업 취소 중...")
                        # 남은 작업 취소
                        for f in future_to_idx:
                            f.cancel()
                        return None, "사용자에 의해 중지됨"

                    test_idx = future_to_idx[future]
                    completed_count += 1

                    # 진행률 콜백 호출
                    if progress_callback:
                        progress_callback(completed_count, test_periods, f"테스트 {completed_count}/{test_periods} 완료")

                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        logger.warning(f"테스트 {test_idx+1} 실패: {e}")

            # 결과를 test_idx 순서로 정렬
            results.sort(key=lambda x: x['test_idx'])

        else:
            # 순차 처리 모드 (기존 방식)
            logger.info("⏳ 순차 처리 모드")
            for i in range(test_periods):
                # 중지 확인
                if cancel_callback and cancel_callback():
                    logger.info(f"⏹ 순차 백테스팅 중지 요청 - {i}/{test_periods}에서 중단")
                    return None, "사용자에 의해 중지됨"

                # 진행률 콜백 호출
                if progress_callback:
                    progress_callback(i + 1, test_periods, f"테스트 {i+1}/{test_periods}")

                # 워커 함수 호출
                result = self._backtest_single_period(data, i, test_periods, forecast_days, ticker)
                if result is not None:
                    results.append(result)
        
        # 전체 통계
        if not results:
            return None, "테스트 결과 없음"
        
        direction_accuracy = sum(r['direction_correct'] for r in results) / len(results)
        avg_magnitude_error = np.mean([r['magnitude_error'] for r in results])
        
        summary = {
            'ticker': ticker,
            'test_count': len(results),
            'direction_accuracy': direction_accuracy,
            'avg_magnitude_error': avg_magnitude_error,
            'results': results
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"백테스팅 결과 요약")
        logger.info(f"{'='*60}")
        logger.info(f"성공한 테스트: {len(results)}/{test_periods}회")
        logger.info(f"방향 정확도: {direction_accuracy*100:.1f}%")
        logger.info(f"평균 오차: {avg_magnitude_error*100:.2f}%")
        logger.info(f"{'='*60}\n")
        
        return summary, None

    def predict_with_historical_data(self, historical_data, forecast_days):
        """과거 데이터만으로 예측 (백테스팅용)"""
        try:
            # 기존 predict_stock의 핵심 로직만 사용
            self.fix_all_random_seeds(42)
            
            # 특성 생성
            features = self.create_advanced_features_deterministic(historical_data)
            
            # 미래 수익률 계산
            future_returns = historical_data['Close'].pct_change(forecast_days).shift(-forecast_days)
            
            # 시퀀스 준비
            X, y = self.prepare_sequences_deterministic(
                features, future_returns, 
                sequence_length=15, 
                forecast_horizon=forecast_days
            )
            
            if len(X) == 0:
                return None
            
            # 전체 데이터로 학습
            X_train = X
            y_train = y
            latest_X = X[-1].reshape(1, -1)
            
            # 모델 예측 (feature names 경고 억제)
            predictions = []
            for model_name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    # sklearn UserWarning 억제
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
                        pred = model.predict(latest_X)[0]
                    predictions.append(pred)
                except:
                    continue
            
            if not predictions:
                return None
            
            return float(np.mean(predictions))

        except Exception as e:
            logger.error(f"예측 오류: {e}")
            return None

class EnhancedStockScreenerMethods:
    """기존 StockScreener 클래스에 추가할 AI 예측 메서드들"""

    def __init__(self):
        """AI 예측 관련 초기화 (Lazy Loading)"""
        self._predictor = None  # Lazy Loading: 실제 사용 시점에 초기화
        self.prediction_settings = self.load_prediction_settings()

    @property
    def predictor(self):
        """예측기 Lazy Loading: 처음 호출될 때만 초기화"""
        if self._predictor is None and ML_AVAILABLE:
            logger.info("🚀 AI 예측기 첫 사용 - 초기화 중...")
            self._predictor = EnhancedCPUPredictor()
            logger.info("✅ AI 예측기 초기화 완료")
        return self._predictor
        
    def load_prediction_settings(self):
        """예측 설정 로드"""
        default_settings = {
            'forecast_days': 7,
            'confidence_threshold': 0.6,
            'batch_delay': 1.0,
            'min_data_days': 300,
            'use_arima_validation': True,
            'models_enabled': {
                'xgboost': True,
                'lightgbm': True,
                'random_forest': True,
                'extra_trees': True,
                'gradient_boosting': True
            }
        }
        
        try:
            if os.path.exists('prediction_settings.json'):
                with open('prediction_settings.json', 'r') as f:
                    saved_settings = json.load(f)
                default_settings.update(saved_settings)
        except:
            pass
        
        return default_settings
    
    def save_prediction_settings(self):
        """예측 설정 저장"""
        try:
            with open('prediction_settings.json', 'w') as f:
                json.dump(self.prediction_settings, f, indent=2)
        except Exception as e:
            logger.error(f"설정 저장 오류: {e}")
    
    def enhance_ui_with_ai_features(self):
        """기존 UI에 AI 기능 추가 - 단순화된 버전"""
        if not hasattr(self, 'menubar'):
            self.menubar = self.menuBar()
        
        # 🔧 단순화된 AI 메뉴
        ai_menu = self.menubar.addMenu('🤖 AI 분석')
        
        # 통합된 예측
        prediction_action = QAction('🎯 종목 예측', self)
        prediction_action.triggered.connect(lambda: self.show_prediction_dialog())  # ticker=None
        ai_menu.addAction(prediction_action)
        
        # 배치 예측
        batch_prediction_action = QAction('📊 배치 예측', self)
        batch_prediction_action.triggered.connect(self.show_batch_prediction)
        ai_menu.addAction(batch_prediction_action)
        
        ai_menu.addSeparator()
        
        # 설정
        settings_action = QAction('⚙️ 예측 설정', self)
        settings_action.triggered.connect(self.show_prediction_settings)
        ai_menu.addAction(settings_action)
        
        # 도움말
        help_action = QAction('❓ 도움말', self)
        help_action.triggered.connect(self.show_ai_help)
        ai_menu.addAction(help_action)
    
    def enhance_table_context_menus(self):
        """테이블 우클릭 메뉴에 AI 예측 추가"""
        # 매수 후보 테이블
        if hasattr(self, 'buy_table'):
            self.buy_table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.buy_table.customContextMenuRequested.connect(
                lambda pos: self.show_table_context_menu(pos, self.buy_table, 'buy')
            )
        
        # 매도 후보 테이블
        if hasattr(self, 'sell_table'):
            self.sell_table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.sell_table.customContextMenuRequested.connect(
                lambda pos: self.show_table_context_menu(pos, self.sell_table, 'sell')
            )
    
    def show_table_context_menu(self, position, table, table_type):
        """테이블 우클릭 메뉴 표시 - 정리된 버전"""
        if not table.itemAt(position):
            return
        
        menu = QMenu()
        
        # 기존 메뉴 항목들 (차트 보기 등)
        chart_action = QAction('📈 차트 보기', self)
        chart_action.triggered.connect(lambda: self.show_chart_from_table(table))
        menu.addAction(chart_action)
        
        if ML_AVAILABLE:
            menu.addSeparator()
            
            # AI 예측 메뉴 (통합)
            predict_action = QAction('🤖 AI 예측', self)
            predict_action.triggered.connect(lambda: self.predict_from_table(table))
            menu.addAction(predict_action)
        
        global_pos = table.mapToGlobal(position)
        menu.exec_()
    
    def show_chart_from_table(self, table):
        """테이블에서 선택된 종목의 차트 표시"""
        current_row = table.currentRow()
        if current_row >= 0:
            ticker_item = table.item(current_row, 0)  # 첫 번째 열이 종목 코드
            if ticker_item:
                ticker = ticker_item.text()
                if hasattr(self, 'show_chart'):
                    self.show_chart(ticker)
    
    def predict_from_table(self, table):
        """테이블에서 선택된 종목 예측 - 다이얼로그 열기"""
        current_row = table.currentRow()
        if current_row >= 0:
            ticker_item = table.item(current_row, 0)
            if ticker_item:
                ticker = ticker_item.text()
                # 예측 다이얼로그를 열고 종목 자동 입력
                self.show_prediction_dialog(ticker)

    def show_prediction_dialog(self, ticker=None):
        """예측 다이얼로그 표시 - 종목 자동 입력 지원"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "AI 예측에 필요한 라이브러리가 설치되지 않았습니다.")
            return

        # 예측 다이얼로그 표시
        try:
            from prediction_window import StockPredictionDialog
            dialog = StockPredictionDialog(self)

            # 종목이 지정된 경우 자동 입력
            if ticker:
                dialog.ticker_input.setText(ticker)

            dialog.exec_()
        except ImportError:
            QMessageBox.critical(self, "Import 오류", "StockPredictionDialog를 찾을 수 없습니다.")

    def show_prediction_result(self, result, settings_info=""):
        """예측 결과 표시"""
        ticker = result.get('ticker', '')
        current_price = result.get('current_price', 0)
        predicted_price = result.get('predicted_price', 0)
        return_rate = result.get('expected_return', 0)
        confidence = result.get('confidence', 0)
        forecast_days = result.get('forecast_days', 7)
        
        # 추천 결정
        if return_rate > 0.02:  # 2% 이상
            recommendation = "🚀 매수 추천"
            color = "🟢"
        elif return_rate < -0.02:  # -2% 이하
            recommendation = "📉 매도 고려"
            color = "🔴"
        else:
            recommendation = "⏸️ 관망"
            color = "🟡"
        
        # ✅ 설정 정보 포함된 메시지
        message = f"""
🎯 {ticker} AI 예측 결과 {settings_info}

💰 현재 가격: ${current_price:.2f}
🎯 예측 가격: ${predicted_price:.2f} ({forecast_days}일 후)
📊 예상 수익률: {return_rate*100:+.2f}%
🎚️ 신뢰도: {confidence*100:.1f}%

{color} {recommendation}

🔧 적용된 설정:
• 예측 기간: {forecast_days}일
• 활성 모델: {len(result.get('active_models', []))}개
• 모델 목록: {', '.join(result.get('active_models', []))}
        """
        
        QMessageBox.information(self, f"AI 예측 - {ticker}", message)


    def show_batch_prediction(self):
        """배치 예측 다이얼로그 표시 - 데이터 구조 개선 버전"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "ML 라이브러리가 설치되지 않았습니다.")
            return
        
        # 스크리닝 결과 수집 및 변환
        candidates = []

        logger.info("스크리닝 결과 확인 중...")

        # 매수 후보 처리
        if hasattr(self, 'last_buy_candidates') and self.last_buy_candidates:
            logger.info(f"매수 후보 발견: {len(self.last_buy_candidates)}개")
            for candidate in self.last_buy_candidates:
                # 다양한 데이터 구조에 대응
                converted = self.convert_candidate_format(candidate, '매수')
                if converted:
                    candidates.append(converted)

        # 매도 후보 처리
        if hasattr(self, 'last_sell_candidates') and self.last_sell_candidates:
            logger.info(f"매도 후보 발견: {len(self.last_sell_candidates)}개")
            for candidate in self.last_sell_candidates:
                converted = self.convert_candidate_format(candidate, '매도')
                if converted:
                    candidates.append(converted)

        logger.info(f"변환된 후보: {len(candidates)}개")
        
        # 후보가 없는 경우 처리
        if not candidates:
            # 디버그 정보 표시
            debug_info = self.get_screening_debug_info()
            
            reply = QMessageBox.question(
                self, "배치 예측", 
                f"스크리닝 결과가 없습니다.\n\n{debug_info}\n\n샘플 종목으로 테스트하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # 샘플 종목들
                candidates = [
                    {'Symbol': 'AAPL', 'Name': 'Apple Inc.', 'Type': '샘플'},
                    {'Symbol': 'MSFT', 'Name': 'Microsoft Corp.', 'Type': '샘플'},
                    {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.', 'Type': '샘플'},
                    {'Symbol': 'TSLA', 'Name': 'Tesla Inc.', 'Type': '샘플'},
                    {'Symbol': '005930.KS', 'Name': 'Samsung Electronics', 'Type': '샘플'}
                ]
            else:
                return
        
        # 중복 제거 (동일 종목 코드)
        unique_candidates = []
        seen_symbols = set()
        
        for candidate in candidates:
            symbol = candidate.get('Symbol', '')
            if symbol and symbol not in seen_symbols:
                unique_candidates.append(candidate)
                seen_symbols.add(symbol)
        
        logger.info(f"최종 예측 대상: {len(unique_candidates)}개 (중복 제거 후)")
        
        try:
            # 배치 예측 다이얼로그 실행
            dialog = BatchPredictionDialog(unique_candidates, self)
            dialog.exec_()
            
        except NameError as e:
            QMessageBox.critical(self, "Import 오류", f"BatchPredictionDialog를 찾을 수 없습니다:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"배치 예측 다이얼로그 오류:\n{str(e)}")

    def convert_candidate_format(self, candidate, candidate_type):
        """스크리닝 결과를 배치 예측 형식으로 변환"""
        try:
            # 다양한 키 이름에 대응하여 종목 코드 추출
            symbol = None
            name = None
            
            # 가능한 종목 코드 키들
            symbol_keys = ['ticker', 'Ticker', 'symbol', 'Symbol', 'code', 'Code', 'stock_code']
            for key in symbol_keys:
                if key in candidate and candidate[key]:
                    symbol = str(candidate[key]).strip().upper()
                    break
            
            # 가능한 종목 이름 키들  
            name_keys = ['name', 'Name', 'company', 'Company', 'stock_name', 'company_name']
            for key in name_keys:
                if key in candidate and candidate[key]:
                    name = str(candidate[key]).strip()
                    break
            
            if not symbol:
                logger.warning(f"종목 코드 없음: {candidate}")
                return None
            
            if not name:
                name = f"종목 {symbol}"
            
            converted = {
                'Symbol': symbol,
                'Name': name,
                'Type': candidate_type,  # '매수' 또는 '매도'
            }
            
            # 추가 정보 포함 (선택적)
            if 'current_price' in candidate:
                converted['CurrentPrice'] = candidate['current_price']
            if 'recommendation_score' in candidate:
                converted['Score'] = candidate['recommendation_score']
            
            return converted

        except Exception as e:
            logger.warning(f"후보 변환 오류: {e}, 데이터: {candidate}")
            return None

    def get_screening_debug_info(self):
        """스크리닝 결과 디버그 정보"""
        debug_lines = ["디버그 정보:"]
        
        # 매수 후보 확인
        if hasattr(self, 'last_buy_candidates'):
            count = len(self.last_buy_candidates) if self.last_buy_candidates else 0
            debug_lines.append(f"• 매수 후보 변수 존재: {count}개")
            
            if count > 0:
                # 첫 번째 데이터 구조 확인
                first_item = self.last_buy_candidates[0]
                keys = list(first_item.keys()) if isinstance(first_item, dict) else ["데이터 구조 오류"]
                debug_lines.append(f"• 매수 후보 키들: {', '.join(keys[:5])}")
        else:
            debug_lines.append("• 매수 후보 변수 없음")
        
        # 매도 후보 확인
        if hasattr(self, 'last_sell_candidates'):
            count = len(self.last_sell_candidates) if self.last_sell_candidates else 0
            debug_lines.append(f"• 매도 후보 변수 존재: {count}개")
            
            if count > 0:
                first_item = self.last_sell_candidates[0]
                keys = list(first_item.keys()) if isinstance(first_item, dict) else ["데이터 구조 오류"]
                debug_lines.append(f"• 매도 후보 키들: {', '.join(keys[:5])}")
        else:
            debug_lines.append("• 매도 후보 변수 없음")
        
        return "\n".join(debug_lines)

    def show_prediction_settings(self):
        """예측 설정 다이얼로그 표시"""
        dialog = PredictionSettingsDialog(self.prediction_settings, self)
        if dialog.exec_() == QDialog.Accepted:
            self.prediction_settings = dialog.get_settings()
            self.save_prediction_settings()
    
    def show_ai_help(self):
        """AI 도움말 표시"""
        help_text = """
🤖 AI 예측 시스템 도움말

═══════════════════════════════════════════════════

📊 사용 가능한 기능:

1. 📈 개별 종목 예측
   • 메뉴 → AI 분석 → 종목 예측
   • 종목 코드 입력 후 예측 실행
   • 여러 ML 모델의 앙상블 예측

2. 📊 배치 예측
   • 스크리닝 결과를 일괄 예측
   • 매수/매도 후보 전체 분석
   • 진행률 표시 및 중단 가능

3. ⚙️ 예측 설정
   • 예측 기간, 신뢰도 임계값 설정
   • 사용할 모델 선택
   • 배치 예측 딜레이 설정

═══════════════════════════════════════════════════

🧠 사용된 AI 모델:

• XGBoost: 주식 예측에 특화된 그래디언트 부스팅
• LightGBM: AMD CPU에 최적화된 고속 모델
• Random Forest: 안정적이고 해석 가능한 모델
• Extra Trees: 과적합을 방지하는 랜덤 모델
• Gradient Boosting: 견고한 성능의 부스팅 모델

═══════════════════════════════════════════════════

📋 사용법:

1. 우클릭 예측:
   • 매수/매도 테이블에서 종목 우클릭
   • 'AI 예측' 메뉴 선택

2. 메뉴 예측:
   • 상단 메뉴 → AI 분석 → 종목 예측
   • 종목 코드 직접 입력

3. 결과 해석:
   • 예상 수익률: 7일 후 예상 수익률
   • 신뢰도: 예측의 신뢰도 (0-100%)
   • 예측가: 현재가 기준 예상 가격

═══════════════════════════════════════════════════

⚠️ 주의사항:

• AI 예측은 참고용이며 투자 보장이 아닙니다
• 과거 데이터 기반이므로 미래가 다를 수 있습니다
• 신뢰도가 낮은 예측은 신중히 판단하세요
• 다양한 정보를 종합하여 투자 결정하세요

═══════════════════════════════════════════════════

🔧 기술 정보:

• 300일 이상의 과거 데이터 필요
• 30개 기술적 지표 사용
• 5개 모델의 앙상블 예측
• CPU 최적화로 빠른 처리
• 랜덤 시드 고정으로 일관성 보장

════════════════════════════════════════════════════
"""
        
        QMessageBox.information(self, "AI 예측 도움말", help_text)


class BatchPredictionDialog(QDialog):
    """배치 예측 다이얼로그"""
    
    def __init__(self, candidates, parent=None):
        super().__init__(parent)
        self.candidates = candidates
        self.predictor = EnhancedCPUPredictor() if ML_AVAILABLE else None
        self.parent = parent  # ✅ 추가: 부모 객체 저장
        self.is_running = False
        self.current_index = 0
        self.results = []
        
        # ✅ prediction_settings 접근 방법 수정
        if parent and hasattr(parent, 'prediction_settings'):
            self.prediction_settings = parent.prediction_settings
        else:
            # 기본 설정값
            self.prediction_settings = {
                'forecast_days': 7,
                'batch_delay': 1.0,
                'confidence_threshold': 0.6
            }
        
        self.setWindowTitle(f'🤖 배치 AI 예측 - {len(candidates)}개 종목')
        self.setGeometry(200, 200, 900, 700)
        self.setModal(True)
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout()
        
        # 상단 정보
        info_label = QLabel(f"📊 총 {len(self.candidates)}개 종목에 대한 AI 예측을 실행합니다")
        info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2196F3; padding: 10px;")
        layout.addWidget(info_label)
        
        # 진행률 표시
        progress_layout = self.create_progress_layout()
        layout.addLayout(progress_layout)
        
        # 통계 요약
        stats_panel = self.create_stats_panel()
        layout.addWidget(stats_panel)
        
        # 결과 테이블
        self.result_table = self.create_result_table()
        layout.addWidget(self.result_table)
        
        # 버튼들
        button_layout = self.create_button_layout()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        if not ML_AVAILABLE:
            info_label.setText("❌ ML 라이브러리가 설치되지 않았습니다")
            info_label.setStyleSheet("color: red; font-weight: bold;")
    
    def create_progress_layout(self):
        """진행률 레이아웃 생성"""
        layout = QVBoxLayout()
        
        # 전체 진행률
        self.overall_progress = QProgressBar()
        self.overall_progress.setMaximum(len(self.candidates))
        self.overall_progress.setValue(0)
        self.overall_progress.setFormat("전체 진행률: %v / %m (%p%)")
        layout.addWidget(self.overall_progress)
        
        # 현재 작업
        self.current_work_label = QLabel("대기 중...")
        self.current_work_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.current_work_label)
        
        # 세부 진행률
        self.detail_progress = QProgressBar()
        self.detail_progress.setMaximum(100)
        self.detail_progress.setValue(0)
        self.detail_progress.setFormat("현재 종목: %p%")
        layout.addWidget(self.detail_progress)
        
        return layout
    
    def create_stats_panel(self):
        """통계 패널 생성"""
        panel = QGroupBox("📈 예측 통계")
        layout = QGridLayout()
        
        self.stats_labels = {
            'completed': QLabel("완료: 0"),
            'success_rate': QLabel("성공률: 0%"),
            'buy_signals': QLabel("매수 신호: 0"),
            'sell_signals': QLabel("매도 신호: 0"),
            'avg_confidence': QLabel("평균 신뢰도: 0%")
        }
        
        row, col = 0, 0
        for key, label in self.stats_labels.items():
            layout.addWidget(label, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1
        
        panel.setLayout(layout)
        return panel
    
    def create_result_table(self):
        """결과 테이블 생성"""
        table = QTableWidget()
        table.setColumnCount(8)
        table.setHorizontalHeaderLabels([
            '종목코드', '종목명', '현재가', '예측가', '예상수익률', '신뢰도', '추천', '상태'
        ])
        
        # 테이블 설정
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSortingEnabled(True)
        
        return table
    
    def create_button_layout(self):
        """버튼 레이아웃 생성"""
        layout = QHBoxLayout()
        
        self.start_btn = QPushButton('🚀 배치 예측 시작')
        self.start_btn.clicked.connect(self.start_batch_prediction)
        self.start_btn.setEnabled(ML_AVAILABLE)
        layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton('⏸️ 일시정지')
        self.pause_btn.clicked.connect(self.pause_prediction)
        self.pause_btn.setEnabled(False)
        layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton('⏹️ 중지')
        self.stop_btn.clicked.connect(self.stop_prediction)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)
        
        layout.addStretch()
        
        self.export_btn = QPushButton('📊 결과 내보내기')
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        
        close_btn = QPushButton('닫기')
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        return layout
    
    def start_batch_prediction(self):
        """배치 예측 시작 - 안전성 개선"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "ML 라이브러리가 설치되지 않았습니다.")
            return
        
        if not self.candidates:
            QMessageBox.warning(self, "오류", "예측할 종목이 없습니다.")
            return
            
        self.is_running = True
        self.current_index = 0
        self.results = []
        
        # UI 상태 변경
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # 테이블 초기화
        self.result_table.setRowCount(len(self.candidates))
        
        for i, candidate in enumerate(self.candidates):
            try:
                # 종목 코드와 이름 추출
                ticker = self.extract_ticker_from_candidate(candidate)
                name = candidate.get('Name', candidate.get('name', f'종목 {i+1}'))
                
                self.result_table.setItem(i, 0, QTableWidgetItem(ticker or 'N/A'))
                self.result_table.setItem(i, 1, QTableWidgetItem(name))
                self.result_table.setItem(i, 7, QTableWidgetItem("⏳ 대기 중"))
                
            except Exception as e:
                logger.warning(f"테이블 초기화 오류 (행 {i}): {e}")
                self.result_table.setItem(i, 0, QTableWidgetItem('오류'))
                self.result_table.setItem(i, 1, QTableWidgetItem('데이터 오류'))
                self.result_table.setItem(i, 7, QTableWidgetItem("❌ 초기화 오류"))

        logger.info(f"배치 예측 시작: {len(self.candidates)}개 종목")
        
        # 예측 시작
        self.run_next_prediction()
    
    def run_next_prediction(self):
        """다음 종목 예측 실행 - 오류 방지 개선"""
        if not self.is_running or self.current_index >= len(self.candidates):
            self.finish_batch_prediction()
            return
        
        candidate = self.candidates[self.current_index]
        
        # ✅ 개선된 종목 코드 추출
        ticker = self.extract_ticker_from_candidate(candidate)
        
        if not ticker:
            logger.warning(f"종목 코드 추출 실패: {candidate}")
            # 실패한 경우 다음으로 넘어감
            self.result_table.setItem(self.current_index, 7, QTableWidgetItem("❌ 종목코드 오류"))
            self.current_index += 1
            # ✅ 함수명 수정
            self.update_stats()
            QTimer.singleShot(100, self.run_next_prediction)
            return

        logger.info(f"예측 시작: {ticker} ({self.current_index + 1}/{len(self.candidates)})")
        
        # UI 업데이트
        self.current_work_label.setText(f"예측 중: {ticker}")
        self.overall_progress.setValue(self.current_index)
        
        # 테이블 상태 업데이트
        self.result_table.setItem(self.current_index, 7, QTableWidgetItem("🔄 예측 중"))
        self.result_table.scrollToItem(self.result_table.item(self.current_index, 0))
        
        # 비동기 예측 실행
        QTimer.singleShot(100, lambda: self.execute_prediction_for_ticker(ticker))


    def extract_ticker_from_candidate(self, candidate):
        """후보 데이터에서 종목 코드 추출 - 여러 형식 지원"""
        if isinstance(candidate, str):
            return candidate.strip().upper()
        
        if not isinstance(candidate, dict):
            logger.error(f"잘못된 데이터 타입: {type(candidate)}")
            return None

        # 가능한 키 이름들 시도
        possible_keys = [
            'Symbol', 'symbol', 'Ticker', 'ticker',
            'Code', 'code', 'stock_code', 'stock_symbol'
        ]

        for key in possible_keys:
            if key in candidate and candidate[key]:
                ticker = str(candidate[key]).strip().upper()
                if ticker and ticker != 'N/A':
                    return ticker

        logger.error(f"종목 코드를 찾을 수 없음. 사용 가능한 키: {list(candidate.keys())}")
        return None

    def execute_prediction_for_ticker(self, ticker):
        """특정 종목에 대한 예측 실행 - 오류 수정 버전"""
        try:
            logger.info(f"예측 실행: {ticker}")

            # Enhanced Screener를 사용한 예측
            forecast_days = self.prediction_settings.get('forecast_days', 7)
            result, error = self.predictor.predict_stock(ticker, forecast_days=forecast_days)

            if error:
                logger.error(f"예측 실패 ({ticker}): {error}")
                # 예측 실패
                self.result_table.setItem(self.current_index, 7, QTableWidgetItem(f"❌ {error[:15]}..."))

            elif result:
                logger.info(f"예측 성공 ({ticker})")
                # 예측 성공 - 결과를 테이블에 표시
                self.display_prediction_result(result, self.current_index)
                self.results.append(result)
                self.result_table.setItem(self.current_index, 7, QTableWidgetItem("✅ 완료"))

            else:
                logger.warning(f"결과 없음 ({ticker})")
                # 결과 없음
                self.result_table.setItem(self.current_index, 7, QTableWidgetItem("❌ 결과 없음"))

        except Exception as e:
            logger.error(f"예측 오류 ({ticker}): {e}")
            error_msg = str(e)[:15] + "..." if len(str(e)) > 15 else str(e)
            self.result_table.setItem(self.current_index, 7, QTableWidgetItem(f"❌ {error_msg}"))
        
        finally:
            # ✅ 함수명 수정: update_statistics → update_stats
            self.update_stats()
            
            # 다음 종목으로 이동
            self.current_index += 1
            
            # 지연 후 다음 예측 실행
            delay = int(self.prediction_settings.get('batch_delay', 1.0) * 1000)
            QTimer.singleShot(delay, self.run_next_prediction)

    def display_prediction_result(self, result, row):
        """예측 결과를 테이블에 표시 - 오류 방지 버전"""
        try:
            # 결과에서 필요한 정보 추출 (안전하게)
            ticker = result.get('ticker', 'N/A')
            current_price = result.get('current_price', 0)
            predicted_price = result.get('predicted_price', 0)
            expected_return = result.get('expected_return', 0)
            confidence = result.get('confidence', 0)

            # 신뢰도 임계값
            confidence_threshold = result.get('confidence_threshold', 0.6)
            is_high_confidence = confidence >= confidence_threshold

            # 추천 결정
            if expected_return > 0.05:  # 5% 이상
                recommendation = "강력 매수"
                color = "green"
            elif expected_return > 0.02:  # 2% 이상  
                recommendation = "매수"
                color = "lightgreen"
            elif expected_return < -0.05:  # -5% 이하
                recommendation = "매도"
                color = "red"
            elif expected_return < -0.02:  # -2% 이하
                recommendation = "매도 고려"
                color = "orange"
            else:
                recommendation = "보유"
                color = "gray"
            
            # 테이블 업데이트 (안전하게)
            try:
                # 현재가
                price_item = QTableWidgetItem(f"${current_price:.2f}")
                self.result_table.setItem(row, 2, price_item)
                
                # 예측가
                pred_item = QTableWidgetItem(f"${predicted_price:.2f}")
                self.result_table.setItem(row, 3, pred_item)
                
                # 예상 수익률
                return_item = QTableWidgetItem(f"{expected_return*100:+.1f}%")
                return_item.setBackground(QColor(color))
                self.result_table.setItem(row, 4, return_item)
                
                # 신뢰도
                confidence_text = f"{confidence*100:.1f}%"
                if is_high_confidence:
                    confidence_text += " ✅"
                else:
                    confidence_text += " ⚠️"
                
                confidence_item = QTableWidgetItem(confidence_text)
                
                # 신뢰도에 따른 배경색
                if is_high_confidence:
                    confidence_item.setBackground(QColor(200, 255, 200))  # 녹색
                else:
                    confidence_item.setBackground(QColor(255, 255, 200))  # 노란색
                
                self.result_table.setItem(row, 5, confidence_item)
                
                # 추천
                rec_item = QTableWidgetItem(recommendation)
                rec_item.setBackground(QColor(color))
                self.result_table.setItem(row, 6, rec_item)

                logger.info(f"결과 표시 완료: {ticker} - {expected_return*100:+.1f}%")

            except Exception as table_error:
                logger.warning(f"테이블 업데이트 오류: {table_error}")

        except Exception as e:
            logger.warning(f"결과 표시 오류: {e}")
    
    def update_stats(self):
        """통계 업데이트"""
        completed = len(self.results)
        total = len(self.candidates)
        
        if completed > 0:
            # 성공률
            success_rate = (completed / self.current_index) * 100 if self.current_index > 0 else 0
            
            # 매수/매도 신호
            buy_signals = sum(1 for r in self.results if r.get('expected_return', 0) > 0.02)
            sell_signals = sum(1 for r in self.results if r.get('expected_return', 0) < -0.02)
            
            # 평균 신뢰도
            avg_confidence = np.mean([r.get('confidence', 0) for r in self.results]) * 100
            
            # UI 업데이트
            self.stats_labels['completed'].setText(f"완료: {completed}")
            self.stats_labels['success_rate'].setText(f"성공률: {success_rate:.1f}%")
            self.stats_labels['buy_signals'].setText(f"매수 신호: {buy_signals}")
            self.stats_labels['sell_signals'].setText(f"매도 신호: {sell_signals}")
            self.stats_labels['avg_confidence'].setText(f"평균 신뢰도: {avg_confidence:.1f}%")
    
    def pause_prediction(self):
        """예측 일시정지"""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.current_work_label.setText("일시정지됨")
    
    def stop_prediction(self):
        """예측 중지"""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.current_work_label.setText("중지됨")
        self.export_btn.setEnabled(True)
    
    def finish_batch_prediction(self):
        """배치 예측 완료"""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(True)
        
        self.current_work_label.setText("✅ 모든 예측 완료!")
        self.overall_progress.setValue(len(self.candidates))
        self.detail_progress.setValue(100)
        
        # 완료 메시지
        completed = len(self.results)
        QMessageBox.information(self, "예측 완료", 
                               f"배치 예측이 완료되었습니다!\n\n"
                               f"총 처리: {self.current_index}개\n"
                               f"성공: {completed}개\n"
                               f"실패: {self.current_index - completed}개")
    
    def export_results(self):
        """결과 내보내기"""
        if not self.results:
            QMessageBox.warning(self, "경고", "내보낼 결과가 없습니다.")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename, _ = QFileDialog.getSaveFileName(
            self, "배치 예측 결과 저장", 
            f"batch_prediction_{timestamp}.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            try:
                # 결과를 DataFrame으로 변환
                df_data = []
                for result in self.results:
                    df_data.append({
                        '종목코드': result.get('ticker', ''),
                        '현재가': result.get('current_price', 0),
                        '예측가': result.get('predicted_price', 0),
                        '예상수익률': result.get('expected_return', 0),
                        '신뢰도': result.get('confidence', 0),
                        '예측일자': result.get('prediction_date', ''),
                        '예측기간': result.get('forecast_days', 7)
                    })
                
                df = pd.DataFrame(df_data)
                df.to_csv(filename, index=False, encoding='utf-8-sig')
                
                QMessageBox.information(self, "저장 완료", 
                                       f"배치 예측 결과가 저장되었습니다:\n{filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "저장 오류", f"파일 저장 중 오류:\n{str(e)}")


class PredictionSettingsDialog(QDialog):
    """예측 설정 다이얼로그"""
    
    def __init__(self, current_settings, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings
        
        self.setWindowTitle('⚙️ AI 예측 설정')
        self.setGeometry(300, 300, 500, 400)
        self.setModal(True)
        
        self.initUI()
        self.load_current_settings()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 기본 설정
        basic_group = QGroupBox("📊 기본 설정")
        basic_layout = QGridLayout()
        
        # 예측 기간
        basic_layout.addWidget(QLabel("예측 기간:"), 0, 0)
        self.forecast_days_spin = QSpinBox()
        self.forecast_days_spin.setRange(1, 30)
        self.forecast_days_spin.setSuffix(" 일")
        basic_layout.addWidget(self.forecast_days_spin, 0, 1)
        
        # 신뢰도 임계값
        basic_layout.addWidget(QLabel("신뢰도 임계값:"), 1, 0)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 0.9)
        self.confidence_spin.setDecimals(1)
        self.confidence_spin.setSingleStep(0.1)
        basic_layout.addWidget(self.confidence_spin, 1, 1)
        
        # 최소 데이터 일수
        basic_layout.addWidget(QLabel("최소 데이터 일수:"), 2, 0)
        self.min_data_spin = QSpinBox()
        self.min_data_spin.setRange(100, 1000)
        self.min_data_spin.setSuffix(" 일")
        basic_layout.addWidget(self.min_data_spin, 2, 1)
        
        # 배치 딜레이
        basic_layout.addWidget(QLabel("배치 딜레이:"), 3, 0)
        self.batch_delay_spin = QDoubleSpinBox()
        self.batch_delay_spin.setRange(0.1, 5.0)
        self.batch_delay_spin.setDecimals(1)
        self.batch_delay_spin.setSuffix(" 초")
        basic_layout.addWidget(self.batch_delay_spin, 3, 1)
        
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)
        
        # 모델 설정
        model_group = QGroupBox("🧠 사용할 AI 모델")
        model_layout = QVBoxLayout()
        
        self.model_checkboxes = {}
        model_names = {
            'xgboost': 'XGBoost (주식 특화)',
            'lightgbm': 'LightGBM (AMD 최적화)',
            'random_forest': 'Random Forest (안정적)',
            'extra_trees': 'Extra Trees (과적합 방지)',
            'gradient_boosting': 'Gradient Boosting (견고함)'
        }
        
        for key, name in model_names.items():
            checkbox = QCheckBox(name)
            self.model_checkboxes[key] = checkbox
            model_layout.addWidget(checkbox)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # ARIMA 검증
        arima_group = QGroupBox("📈 추가 검증")
        arima_layout = QVBoxLayout()
        
        self.arima_checkbox = QCheckBox("ARIMA 모델로 추가 검증")
        arima_layout.addWidget(self.arima_checkbox)
        
        arima_group.setLayout(arima_layout)
        layout.addWidget(arima_group)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        # 기본값 복원
        default_btn = QPushButton('🔄 기본값')
        default_btn.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(default_btn)
        
        # 저장 및 취소 버튼
        cancel_btn = QPushButton('취소')
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton('💾 저장')
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def load_current_settings(self):
        """현재 설정 로드"""
        self.forecast_days_spin.setValue(self.current_settings.get('forecast_days', 7))
        self.confidence_spin.setValue(self.current_settings.get('confidence_threshold', 0.6))
        self.batch_delay_spin.setValue(self.current_settings.get('batch_delay', 1.0))
        self.min_data_spin.setValue(self.current_settings.get('min_data_days', 300))
        
        # 모델 체크박스 설정
        models_enabled = self.current_settings.get('models_enabled', {})
        for model_key, checkbox in self.model_checkboxes.items():
            checkbox.setChecked(models_enabled.get(model_key, True))
        
        self.arima_checkbox.setChecked(self.current_settings.get('use_arima_validation', True))
    
    def reset_to_defaults(self):
        """기본값으로 복원"""
        self.forecast_days_spin.setValue(7)
        self.confidence_spin.setValue(0.6)
        self.batch_delay_spin.setValue(1.0)
        self.min_data_spin.setValue(300)
        
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(True)
        
        self.arima_checkbox.setChecked(True)
    
    def get_settings(self):
        """현재 설정 반환"""
        return {
            'forecast_days': self.forecast_days_spin.value(),
            'confidence_threshold': self.confidence_spin.value(),
            'batch_delay': self.batch_delay_spin.value(),
            'min_data_days': self.min_data_spin.value(),
            'use_arima_validation': self.arima_checkbox.isChecked(),
            'models_enabled': {
                model_key: checkbox.isChecked()
                for model_key, checkbox in self.model_checkboxes.items()
            }
        }


# 사용 예제 및 테스트
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    logger.info("Enhanced Screener 테스트")

    if ML_AVAILABLE:
        logger.info("CPU 최적화 ML 스택 사용 가능")

        # 예제 1: CPU 최적화 예측기 테스트
        predictor = EnhancedCPUPredictor()
        logger.info("예측기 초기화 완료")
        
        # 예제 2: 배치 예측 다이얼로그 테스트
        sample_candidates = [
            {'Symbol': 'AAPL', 'Name': 'Apple Inc.'},
            {'Symbol': 'MSFT', 'Name': 'Microsoft Corp.'},
            {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.'}
        ]
        
        batch_dialog = BatchPredictionDialog(sample_candidates)
        batch_dialog.show()
        
        # 예제 3: 설정 다이얼로그 테스트
        default_settings = {
            'forecast_days': 7,
            'confidence_threshold': 0.6,
            'batch_delay': 1.0,
            'min_data_days': 300,
            'use_arima_validation': True,
            'models_enabled': {
                'xgboost': True,
                'lightgbm': True,
                'random_forest': True,
                'extra_trees': True,
                'gradient_boosting': True
            }
        }
        
        settings_dialog = PredictionSettingsDialog(default_settings)
        settings_dialog.show()
        
    else:
        logger.warning("ML 라이브러리 설치 필요")
        logger.warning("설치 명령어: pip install scikit-learn xgboost lightgbm statsmodels")
    
    sys.exit(app.exec_())
