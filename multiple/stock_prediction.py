#!/usr/bin/env python3
"""
stock_prediction.py
TensorFlow 없이 scikit-learn, XGBoost, LightGBM, statsmodels를 사용한 주가 예측 시스템
- Kalman Filter (순수 NumPy 구현)
- XGBoost Regressor
- LightGBM Regressor  
- Random Forest
- ARIMA
- Ensemble Method
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 로깅 설정
from logger_config import get_logger
logger = get_logger(__name__)

# 최적화 모듈
from cache_manager import get_stock_data

# 필요한 라이브러리들
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    SKLEARN_AVAILABLE = True
    logger.info("scikit-learn 사용 가능")
except ImportError:
    logger.error("scikit-learn이 설치되지 않았습니다: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logger.info(f"XGBoost 사용 가능 (v{xgb.__version__})")
except ImportError:
    logger.error("XGBoost가 설치되지 않았습니다: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info(f"LightGBM 사용 가능 (v{lgb.__version__})")
except ImportError:
    logger.error("LightGBM이 설치되지 않았습니다: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
    logger.info("statsmodels 사용 가능")
except ImportError:
    logger.error("statsmodels가 설치되지 않았습니다: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# TensorFlow Lazy Import (실제 사용 시점에 import)
TENSORFLOW_AVAILABLE = None  # None = 아직 확인 안 함, True/False = 확인 완료
_tensorflow_modules = {}  # 캐싱용

def _lazy_import_tensorflow():
    """TensorFlow를 필요할 때만 import (Lazy Loading)"""
    global TENSORFLOW_AVAILABLE, _tensorflow_modules

    if TENSORFLOW_AVAILABLE is not None:
        return TENSORFLOW_AVAILABLE

    try:
        logger.info("TensorFlow 로딩 중... (최초 1회, 시간이 걸릴 수 있습니다)")
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, Model
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        _tensorflow_modules['tf'] = tf
        _tensorflow_modules['keras'] = keras
        _tensorflow_modules['layers'] = layers
        _tensorflow_modules['Model'] = Model
        _tensorflow_modules['EarlyStopping'] = EarlyStopping
        _tensorflow_modules['ReduceLROnPlateau'] = ReduceLROnPlateau

        TENSORFLOW_AVAILABLE = True
        logger.info(f"✅ TensorFlow 로딩 완료 (v{tf.__version__})")
        return True
    except ImportError as e:
        logger.warning(f"TensorFlow를 사용할 수 없습니다: {e}")
        logger.info("LSTM/Transformer 모델은 비활성화됩니다 (XGBoost/LightGBM 등 다른 모델은 정상 작동)")
        TENSORFLOW_AVAILABLE = False
        return False

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    HYPEROPT_AVAILABLE = True
    logger.info("Hyperparameter 최적화 도구 사용 가능")
except ImportError:
    logger.warning("scikit-optimize 설치 권장: pip install scikit-optimize")
    HYPEROPT_AVAILABLE = False

class KalmanFilterPredictor:
    """
    순수 NumPy로 구현한 Kalman Filter를 사용한 주가 예측
    
    원리:
    - 상태 공간 모델로 주가의 숨겨진 상태(추세, 노이즈)를 추적
    - 이전 상태 + 관측값을 결합하여 최적 추정
    - 단기 노이즈를 제거하고 진짜 추세 파악
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-1):
        """
        Args:
            process_variance: 프로세스 노이즈 (작을수록 안정적)
            measurement_variance: 측정 노이즈 (클수록 관측값 덜 신뢰)
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.reset()
    
    def reset(self):
        """칼만 필터 상태 초기화"""
        self.x = 0  # 상태 추정값 (가격)
        self.P = 1  # 오차 공분산
        self.Q = self.process_variance  # 프로세스 노이즈
        self.R = self.measurement_variance  # 측정 노이즈
        
        self.predictions = []
        self.states = []
    
    def predict_and_update(self, measurement):
        """칼만 필터의 예측-업데이트 단계"""
        # 1. 예측 단계
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # 2. 업데이트 단계
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        # 결과 저장
        self.predictions.append(self.x)
        self.states.append({'x': self.x, 'P': self.P, 'K': K})
        
        return self.x
    
    def fit_predict(self, prices, forecast_days=5):
        """가격 시계열에 칼만 필터 적용 및 미래 예측"""
        self.reset()
        
        # 모든 관측값에 대해 칼만 필터 적용
        filtered_prices = []
        for price in prices:
            filtered_price = self.predict_and_update(price)
            filtered_prices.append(filtered_price)
        
        # ✅ Kalman Filter 미래 예측
        # 현재가 기준으로 예측 (필터링 값과의 차이 보정)
        current_price = prices[-1]
        last_filtered = filtered_prices[-1]

        # 필터링된 추세 계산
        if len(filtered_prices) >= 10:
            trend = (filtered_prices[-1] - filtered_prices[-10]) / 10
        else:
            trend = filtered_prices[-1] - filtered_prices[-2] if len(filtered_prices) >= 2 else 0

        # 현재가 기준으로 예측 (필터링 오차 보정)
        future_predictions = []
        for i in range(forecast_days):
            # 현재가 + 추세
            future_pred = current_price + trend * (i + 1)
            future_predictions.append(future_pred)
        
        # 신뢰구간 계산
        confidence_interval = 2 * np.sqrt(self.P)
        
        return {
            'filtered_prices': np.array(filtered_prices),
            'future_predictions': np.array(future_predictions),
            'confidence_interval': confidence_interval,
            'last_state': self.x,
            'uncertainty': self.P
        }

class HyperparameterOptimizer:
    """Bayesian Optimization을 사용한 하이퍼파라미터 최적화"""

    @staticmethod
    def optimize_xgboost(X_train, y_train, n_iter=20):
        """XGBoost 하이퍼파라미터 최적화"""
        if not HYPEROPT_AVAILABLE or not XGBOOST_AVAILABLE:
            logger.warning("Bayesian Optimization 불가 - 기본 파라미터 사용")
            return None

        search_spaces = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'gamma': Real(0.0, 0.5),
            'reg_alpha': Real(0.0, 1.0),
            'reg_lambda': Real(0.0, 2.0)
        }

        bayes_cv = BayesSearchCV(
            xgb.XGBRegressor(random_state=42, verbosity=0),
            search_spaces,
            n_iter=n_iter,
            cv=3,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=0
        )

        bayes_cv.fit(X_train, y_train)
        logger.info(f"XGBoost 최적 파라미터: {bayes_cv.best_params_}")

        return bayes_cv.best_estimator_

    @staticmethod
    def optimize_lightgbm(X_train, y_train, n_iter=20):
        """LightGBM 하이퍼파라미터 최적화"""
        if not HYPEROPT_AVAILABLE or not LIGHTGBM_AVAILABLE:
            logger.warning("Bayesian Optimization 불가 - 기본 파라미터 사용")
            return None

        search_spaces = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(3, 15),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'num_leaves': Integer(20, 100),
            'min_child_samples': Integer(10, 50),
            'subsample': Real(0.6, 1.0),
            'colsample_bytree': Real(0.6, 1.0),
            'reg_alpha': Real(0.0, 1.0),
            'reg_lambda': Real(0.0, 2.0)
        }

        bayes_cv = BayesSearchCV(
            lgb.LGBMRegressor(random_state=42, verbosity=-1),
            search_spaces,
            n_iter=n_iter,
            cv=3,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
            verbose=0
        )

        bayes_cv.fit(X_train, y_train)
        logger.info(f"LightGBM 최적 파라미터: {bayes_cv.best_params_}")

        return bayes_cv.best_estimator_

    @staticmethod
    def optimize_random_forest(X_train, y_train, n_iter=20):
        """Random Forest 하이퍼파라미터 최적화"""
        if not HYPEROPT_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Bayesian Optimization 불가 - 기본 파라미터 사용")
            return None

        search_spaces = {
            'n_estimators': Integer(100, 500),
            'max_depth': Integer(5, 30),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Categorical(['sqrt', 'log2', None])
        }

        bayes_cv = BayesSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            search_spaces,
            n_iter=n_iter,
            cv=3,
            n_jobs=1,  # RF 자체가 병렬처리
            scoring='neg_mean_squared_error',
            verbose=0
        )

        bayes_cv.fit(X_train, y_train)
        logger.info(f"Random Forest 최적 파라미터: {bayes_cv.best_params_}")

        return bayes_cv.best_estimator_


class AdvancedMLPredictor:
    """
    XGBoost, LightGBM, Random Forest를 사용한 고급 머신러닝 예측기
    """

    def __init__(self, sequence_length=30, use_optimization=False, ticker=None, auto_load=True):
        self.sequence_length = sequence_length
        self.scaler = RobustScaler()  # StandardScaler -> RobustScaler (이상치에 강함)
        self.models = {}
        self.use_optimization = use_optimization
        self.progress_callback = None  # 진행 콜백 (외부에서 설정)
        self.ticker = ticker
        self.persistence = None

        # 모델 저장/로드 시스템 초기화
        try:
            from model_persistence import get_model_persistence
            self.persistence = get_model_persistence()

            # 자동 로드
            if auto_load and ticker:
                self._try_load_models()
        except ImportError:
            logger.debug("model_persistence 모듈 없음, 저장/로드 비활성화")

    def _try_load_models(self):
        """저장된 ML 모델들 자동 로드 시도"""
        if not self.persistence or not self.ticker:
            return

        loaded_count = 0
        for model_type in ['random_forest', 'xgboost', 'lightgbm']:
            try:
                model, metadata, scaler = self.persistence.load_sklearn_model(self.ticker, model_type)
                if model is not None:
                    self.models[model_type] = model
                    if scaler is not None and loaded_count == 0:  # 첫 번째 모델의 scaler 사용
                        self.scaler = scaler
                    loaded_count += 1
                    logger.info(f"✅ 저장된 {model_type} 모델 로드: {self.ticker}")
            except Exception as e:
                logger.debug(f"{model_type} 모델 로드 실패: {e}")

        if loaded_count > 0:
            logger.info(f"✅ 총 {loaded_count}개 ML 모델 로드 완료")
            return True

        return False
        
    def create_features(self, data):
        """기술적 지표를 포함한 피처 생성 - 고급 지표 추가"""
        df = pd.DataFrame()

        # 기본 가격 정보
        df['close'] = data
        df['high'] = data  # 간단화를 위해 close와 동일하게 처리
        df['low'] = data
        df['volume'] = 1000000  # 더미 거래량

        # === 기존 지표 ===
        # 이동평균
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['ma200'] = df['close'].rolling(200).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 변화율
        df['pct_change_1'] = df['close'].pct_change()
        df['pct_change_5'] = df['close'].pct_change(5)
        df['pct_change_10'] = df['close'].pct_change(10)

        # 변동성
        df['volatility'] = df['close'].rolling(20).std()

        # === 새로운 고급 지표 ===
        # ATR (Average True Range) - 변동성
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # ADX (Average Directional Index) - 추세 강도
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr14 = true_range.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr14)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Stochastic Oscillator - 과매수/과매도
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # OBV (On-Balance Volume) - 거래량 기반
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        df['obv_ma'] = df['obv'].rolling(20).mean()

        # Ichimoku Cloud
        nine_period_high = df['high'].rolling(9).max()
        nine_period_low = df['low'].rolling(9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2

        period26_high = df['high'].rolling(26).max()
        period26_low = df['low'].rolling(26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2

        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

        period52_high = df['high'].rolling(52).max()
        period52_low = df['low'].rolling(52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # ROC (Rate of Change)
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

        # MFI (Money Flow Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = []
        negative_flow = []
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow

        positive_mf = pd.Series(positive_flow).rolling(14).sum()
        negative_mf = pd.Series(negative_flow).rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        df['mfi'] = mfi

        return df
    
    def prepare_data(self, prices):
        """ML 모델용 데이터 전처리 - 이상치 제거 추가"""
        # 피처 생성
        df = self.create_features(prices)

        # NaN 제거
        df = df.dropna()

        # 이상치 제거 (Z-score > 3)
        from scipy import stats
        z_scores = np.abs(stats.zscore(df['close']))
        df = df[(z_scores < 3)]

        if len(df) == 0:
            raise ValueError("이상치 제거 후 데이터가 없습니다")

        logger.debug(f"이상치 제거 후 데이터 포인트: {len(df)}개")
        
        if len(df) < self.sequence_length + 1:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.sequence_length + 1}개 필요")
        
        # 피처 선택 - 고급 지표 포함
        feature_columns = [
            # 기존 지표
            'ma5', 'ma10', 'ma20', 'ma50', 'ma200',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_lower', 'bb_width',
            'pct_change_1', 'pct_change_5', 'pct_change_10',
            'volatility',
            # 새로운 고급 지표
            'atr', 'adx', 'plus_di', 'minus_di',
            'stoch_k', 'stoch_d',
            'obv_ma', 'williams_r', 'cci', 'roc', 'mfi',
            'tenkan_sen', 'kijun_sen'
        ]
        
        # 시퀀스 데이터 생성
        # ✅ 수정: 다음날 변화율(%)을 예측하도록 변경
        X, y = [], []
        for i in range(self.sequence_length, len(df) - 1):  # -1: 다음날 데이터가 있어야 함
            # 과거 sequence_length개의 피처들
            sequence_features = []
            for j in range(i - self.sequence_length, i):
                row_features = df[feature_columns].iloc[j].values
                sequence_features.extend(row_features)

            X.append(sequence_features)

            # ✅ 타겟: 다음날 변화율 (%)
            current_price = df['close'].iloc[i]
            next_price = df['close'].iloc[i + 1]
            price_change_pct = (next_price - current_price) / current_price * 100
            y.append(price_change_pct)

        return np.array(X), np.array(y)
    
    def train_models_with_cv(self, X, y):
        """Time Series Cross-Validation을 사용한 모델 훈련"""
        if not SKLEARN_AVAILABLE:
            return

        # Time Series Split 설정
        tscv = TimeSeriesSplit(n_splits=5)

        # 1. Random Forest with CV
        if self.progress_callback:
            self.progress_callback('ml', 'Random Forest 학습 중 (1/3)...')

        rf_scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_val)
            rf_scores.append(np.sqrt(mean_squared_error(y_val, rf_pred)))

        # 전체 데이터로 최종 훈련
        rf_final = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_final.fit(X, y)
        self.models['random_forest'] = rf_final
        logger.info(f"Random Forest CV RMSE: {np.mean(rf_scores):.2f} (±{np.std(rf_scores):.2f})")

        # 모델 저장
        if self.persistence and self.ticker:
            try:
                metadata = {'cv_rmse_mean': np.mean(rf_scores), 'cv_rmse_std': np.std(rf_scores)}
                self.persistence.save_sklearn_model(rf_final, self.ticker, 'random_forest', metadata, self.scaler)
                # 저장 직후, 오래된 버전 정리(최신 5개만 유지)
                self.persistence.delete_old_models(self.ticker, keep_latest=5)                
            except Exception as e:
                logger.warning(f"Random Forest 저장 실패: {e}")

        # 2. XGBoost with CV
        if XGBOOST_AVAILABLE:
            if self.progress_callback:
                self.progress_callback('ml', 'XGBoost 학습 중 (2/3)...')

            xgb_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                xgb_model = xgb.XGBRegressor(
                    n_estimators=150,  # 300 → 150 (속도 2배 향상)
                    max_depth=8,
                    learning_rate=0.05,  # 0.03 → 0.05 (learning rate 높여서 보완)
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0
                )
                xgb_model.fit(X_train, y_train)
                xgb_pred = xgb_model.predict(X_val)
                xgb_scores.append(np.sqrt(mean_squared_error(y_val, xgb_pred)))

            xgb_final = xgb.XGBRegressor(
                n_estimators=150,  # 300 → 150
                max_depth=8,
                learning_rate=0.05,  # 0.03 → 0.05
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            xgb_final.fit(X, y)
            self.models['xgboost'] = xgb_final
            logger.info(f"XGBoost CV RMSE: {np.mean(xgb_scores):.2f} (±{np.std(xgb_scores):.2f})")

            # 모델 저장
            if self.persistence and self.ticker:
                try:
                    metadata = {'cv_rmse_mean': np.mean(xgb_scores), 'cv_rmse_std': np.std(xgb_scores)}
                    self.persistence.save_sklearn_model(xgb_final, self.ticker, 'xgboost', metadata, self.scaler)
                    # 저장 직후, 오래된 버전 정리(최신 5개만 유지)
                    self.persistence.delete_old_models(self.ticker, keep_latest=5)
                except Exception as e:
                    logger.warning(f"XGBoost 저장 실패: {e}")

        # 3. LightGBM with CV
        if LIGHTGBM_AVAILABLE:
            if self.progress_callback:
                self.progress_callback('ml', 'LightGBM 학습 중 (3/3)...')

            lgb_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                lgb_model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=-1
                )
                lgb_model.fit(X_train, y_train)
                lgb_pred = lgb_model.predict(X_val)
                lgb_scores.append(np.sqrt(mean_squared_error(y_val, lgb_pred)))

            lgb_final = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=-1
            )
            lgb_final.fit(X, y)
            self.models['lightgbm'] = lgb_final
            logger.info(f"LightGBM CV RMSE: {np.mean(lgb_scores):.2f} (±{np.std(lgb_scores):.2f})")

            # 모델 저장
            if self.persistence and self.ticker:
                try:
                    metadata = {'cv_rmse_mean': np.mean(lgb_scores), 'cv_rmse_std': np.std(lgb_scores)}
                    self.persistence.save_sklearn_model(lgb_final, self.ticker, 'lightgbm', metadata, self.scaler)
                    # 저장 직후, 오래된 버전 정리(최신 5개만 유지)
                    self.persistence.delete_old_models(self.ticker, keep_latest=5)
                except Exception as e:
                    logger.warning(f"LightGBM 저장 실패: {e}")

    def train_models(self, X_train, y_train, X_val, y_val):
        """여러 ML 모델 훈련 - 하이퍼파라미터 최적화 옵션"""

        # 1. Random Forest
        if SKLEARN_AVAILABLE:
            if self.use_optimization:
                logger.info("Random Forest Bayesian Optimization 실행...")
                rf_model = HyperparameterOptimizer.optimize_random_forest(X_train, y_train, n_iter=15)

            if not self.use_optimization or rf_model is None:
                rf_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train, y_train)

            self.models['random_forest'] = rf_model
            rf_pred = rf_model.predict(X_val)
            rf_score = np.sqrt(mean_squared_error(y_val, rf_pred))
            logger.info(f"Random Forest RMSE: {rf_score:.2f}")

        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            if self.use_optimization:
                logger.info("XGBoost Bayesian Optimization 실행...")
                xgb_model = HyperparameterOptimizer.optimize_xgboost(X_train, y_train, n_iter=15)

            if not self.use_optimization or xgb_model is None:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.03,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0
                )
                xgb_model.fit(X_train, y_train)

            self.models['xgboost'] = xgb_model
            xgb_pred = xgb_model.predict(X_val)
            xgb_score = np.sqrt(mean_squared_error(y_val, xgb_pred))
            logger.info(f"XGBoost RMSE: {xgb_score:.2f}")

        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            if self.use_optimization:
                logger.info("LightGBM Bayesian Optimization 실행...")
                lgb_model = HyperparameterOptimizer.optimize_lightgbm(X_train, y_train, n_iter=15)

            if not self.use_optimization or lgb_model is None:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.03,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=-1
                )
                lgb_model.fit(X_train, y_train)

            self.models['lightgbm'] = lgb_model
            lgb_pred = lgb_model.predict(X_val)
            lgb_score = np.sqrt(mean_squared_error(y_val, lgb_pred))
            logger.info(f"LightGBM RMSE: {lgb_score:.2f}")
    
    def fit_predict(self, prices, forecast_days=5, use_cv=True):
        """ML 모델 훈련 및 예측 - CV 옵션 추가"""
        if len(prices) < self.sequence_length + 20:
            raise ValueError(f"최소 {self.sequence_length + 20}개의 데이터 포인트가 필요합니다")

        # 데이터 준비
        X, y = self.prepare_data(prices)

        # Time Series Cross-Validation 사용 여부
        if use_cv and len(X) > 100:
            logger.info("Time Series Cross-Validation 사용")
            self.train_models_with_cv(X, y)
        else:
            # 훈련/검증 분할
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # 모델 훈련
            self.train_models(X_train, y_train, X_val, y_val)
        
        # 앙상블 예측
        if not self.models:
            raise ValueError("사용 가능한 ML 모델이 없습니다")
        
        # 최근 시퀀스로 미래 예측
        last_sequence = X[-1].reshape(1, -1)
        
        # ✅ 모델 예측: 변화율(%)을 예측
        predictions_pct = {}
        for model_name, model in self.models.items():
            pred_pct = model.predict(last_sequence)[0]  # 변화율(%)
            predictions_pct[model_name] = pred_pct
            logger.info(f"{model_name} 예측 변화율: {pred_pct:+.2f}%")

        # 앙상블 (평균 변화율)
        ensemble_pct = np.mean(list(predictions_pct.values()))
        current_price = prices[-1]

        logger.info(f"현재가: {current_price:.2f}, 앙상블 예측 변화율: {ensemble_pct:+.2f}%")

        # ✅ 변화율을 절대 가격으로 변환
        day1_pred = current_price * (1 + ensemble_pct / 100)

        # ✅ 재귀적 예측: 매일 같은 변화율 적용 (단순화)
        future_predictions = []
        future_predictions.append(day1_pred)

        # 2일차 이후: 변화율을 반복 적용 (감쇠 적용)
        for i in range(1, forecast_days):
            # 변화율 감쇠: 멀어질수록 0%에 수렴
            decay = 0.9 ** i
            adjusted_pct = ensemble_pct * decay
            next_pred = current_price * (1 + adjusted_pct * (i + 1) / 100)
            future_predictions.append(next_pred)

        logger.info(f"ML 예측 완료: 1일차={future_predictions[0]:.2f} ({ensemble_pct:+.2f}%), "
                   f"최종={future_predictions[-1]:.2f} ({(future_predictions[-1]-current_price)/current_price*100:+.2f}%)")
        
        return {
            'future_predictions': np.array(future_predictions),
            'individual_predictions': predictions_pct,  # 변화율로 저장
            'ensemble_prediction_pct': ensemble_pct,  # 변화율(%)
            'model_performance': {
                'models_used': list(self.models.keys()),
                'validation_score': 'calculated_above'
            }
        }

class LSTMPredictor:
    """LSTM 딥러닝 모델을 사용한 주가 예측"""

    def __init__(self, sequence_length=60, units=128, ticker=None, auto_load=True):
        self.sequence_length = sequence_length
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler()
        self.ticker = ticker
        self.persistence = None

        # 모델 저장/로드 시스템 초기화
        try:
            from model_persistence import get_model_persistence
            self.persistence = get_model_persistence()

            # 자동 로드 (티커가 있고, auto_load=True인 경우)
            if auto_load and ticker:
                self._try_load_model()
        except ImportError:
            logger.debug("model_persistence 모듈 없음, 저장/로드 비활성화")

    def _try_load_model(self):
        """저장된 모델 자동 로드 시도"""
        if not self.persistence or not self.ticker:
            return

        try:
            model, metadata, scaler = self.persistence.load_keras_model(self.ticker, 'lstm')
            if model is not None:
                self.model = model
                if scaler is not None:
                    self.scaler = scaler
                logger.info(f"✅ 저장된 LSTM 모델 로드: {self.ticker} (버전: {metadata.get('version', 'unknown')})")
                return True
        except Exception as e:
            logger.debug(f"LSTM 모델 로드 실패 (새로 훈련): {e}")

        return False

    def build_model(self, input_shape):
        """LSTM 모델 구축"""
        # Lazy Import TensorFlow
        if not _lazy_import_tensorflow():
            raise ImportError("TensorFlow가 필요합니다")

        keras = _tensorflow_modules['keras']
        layers = _tensorflow_modules['layers']

        model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(self.units // 2, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.units // 4),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_sequences(self, data):
        """시계열 시퀀스 준비"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def fit_predict(self, prices, forecast_days=5, force_retrain=False):
        """LSTM 모델 학습 및 예측"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow 없음 - LSTM 건너뜀")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback'}

        try:
            # 저장된 모델이 있고 재훈련 강제가 아니면 예측만 수행
            if self.model is not None and not force_retrain:
                logger.info("✅ 기존 LSTM 모델 사용 (재훈련 없음)")
                return self._predict_only(prices, forecast_days)

            X, y = self.prepare_sequences(prices)

            if len(X) < 50:
                raise ValueError("LSTM 학습에 충분한 데이터 없음")

            # 학습/검증 분할
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # 3D 형태로 reshape
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            # 모델 구축
            self.model = self.build_model((X_train.shape[1], 1))

            # 콜백 설정
            EarlyStopping = _tensorflow_modules['EarlyStopping']
            ReduceLROnPlateau = _tensorflow_modules['ReduceLROnPlateau']
            early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # patience 10 → 15
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

            # 학습
            logger.info("🔄 LSTM 모델 훈련 시작...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,  # 100 → 150
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            # 미래 예측
            last_sequence = X[-1].reshape((1, self.sequence_length, 1))
            predictions = []

            for _ in range(forecast_days):
                pred = self.model.predict(last_sequence, verbose=0)[0, 0]
                predictions.append(pred)

                # 시퀀스 업데이트
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred

            # 역스케일링
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            # 모델 저장
            if self.persistence and self.ticker:
                try:
                    metadata = {
                        'train_loss': history.history['loss'][-1],
                        'val_loss': history.history['val_loss'][-1],
                        'epochs_trained': len(history.history['loss']),
                        'sequence_length': self.sequence_length,
                        'units': self.units,
                        'data_size': len(prices)
                    }
                    self.persistence.save_keras_model(self.model, self.ticker, 'lstm', metadata, self.scaler)
                    # 저장 직후, 오래된 버전 정리(최신 5개만 유지)
                    self.persistence.delete_old_models(self.ticker, keep_latest=5)
                except Exception as e:
                    logger.warning(f"모델 저장 실패: {e}")

            return {
                'future_predictions': predictions,
                'model_type': 'LSTM',
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }

        except Exception as e:
            logger.error(f"LSTM 실패: {e}")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback', 'error': str(e)}

    def _predict_only(self, prices, forecast_days):
        """저장된 모델로 예측만 수행 (재훈련 없음)"""
        try:
            X, y = self.prepare_sequences(prices)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # 미래 예측
            last_sequence = X[-1].reshape((1, self.sequence_length, 1))
            predictions = []

            for _ in range(forecast_days):
                pred = self.model.predict(last_sequence, verbose=0)[0, 0]
                predictions.append(pred)

                # 시퀀스 업데이트
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred

            # 역스케일링
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            return {
                'future_predictions': predictions,
                'model_type': 'LSTM',
                'using_cached_model': True
            }

        except Exception as e:
            logger.error(f"LSTM 예측 실패: {e}")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback', 'error': str(e)}


class TransformerPredictor:
    """Transformer 모델을 사용한 주가 예측"""

    def __init__(self, sequence_length=60, d_model=64, num_heads=4, num_layers=2, ticker=None, auto_load=True):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model = None
        self.scaler = MinMaxScaler()
        self.ticker = ticker
        self.persistence = None

        # 모델 저장/로드 시스템 초기화
        try:
            from model_persistence import get_model_persistence
            self.persistence = get_model_persistence()

            # 자동 로드
            if auto_load and ticker:
                self._try_load_model()
        except ImportError:
            logger.debug("model_persistence 모듈 없음, 저장/로드 비활성화")

    def _try_load_model(self):
        """저장된 모델 자동 로드 시도"""
        if not self.persistence or not self.ticker:
            return

        try:
            model, metadata, scaler = self.persistence.load_keras_model(self.ticker, 'transformer')
            if model is not None:
                self.model = model
                if scaler is not None:
                    self.scaler = scaler
                logger.info(f"✅ 저장된 Transformer 모델 로드: {self.ticker} (버전: {metadata.get('version', 'unknown')})")
                return True
        except Exception as e:
            logger.debug(f"Transformer 모델 로드 실패 (새로 훈련): {e}")

        return False

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0.1):
        """Transformer Encoder Block"""
        layers = _tensorflow_modules['layers']

        # Multi-Head Attention
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

        # Feed Forward
        ff = layers.Dense(ff_dim, activation="relu")(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(inputs.shape[-1])(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)

        return x

    def build_model(self, input_shape):
        """Transformer 모델 구축"""
        # Lazy Import TensorFlow
        if not _lazy_import_tensorflow():
            raise ImportError("TensorFlow가 필요합니다")

        keras = _tensorflow_modules['keras']
        layers = _tensorflow_modules['layers']

        inputs = keras.Input(shape=input_shape)

        # Positional Encoding
        x = layers.Dense(self.d_model)(inputs)

        # Transformer Encoder Blocks
        for _ in range(self.num_layers):
            x = self.transformer_encoder(
                x,
                head_size=self.d_model // self.num_heads,
                num_heads=self.num_heads,
                ff_dim=self.d_model * 4,
                dropout=0.1
            )

        # Output layers
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_sequences(self, data):
        """시계열 시퀀스 준비"""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def fit_predict(self, prices, forecast_days=5, force_retrain=False):
        """Transformer 모델 학습 및 예측"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow 없음 - Transformer 건너뜀")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback'}

        try:
            # 저장된 모델이 있고 재훈련 강제가 아니면 예측만 수행
            if self.model is not None and not force_retrain:
                logger.info("✅ 기존 Transformer 모델 사용 (재훈련 없음)")
                return self._predict_only(prices, forecast_days)

            X, y = self.prepare_sequences(prices)

            if len(X) < 50:
                raise ValueError("Transformer 학습에 충분한 데이터 없음")

            # 학습/검증 분할
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            # 3D 형태로 reshape
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

            # 모델 구축
            self.model = self.build_model((X_train.shape[1], 1))

            # 콜백 설정
            EarlyStopping = _tensorflow_modules['EarlyStopping']
            ReduceLROnPlateau = _tensorflow_modules['ReduceLROnPlateau']
            early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  # patience 15 → 20
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

            # 학습
            logger.info("🔄 Transformer 모델 훈련 시작...")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,  # 100 → 150
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )

            # 미래 예측
            last_sequence = X[-1].reshape((1, self.sequence_length, 1))
            predictions = []

            for _ in range(forecast_days):
                pred = self.model.predict(last_sequence, verbose=0)[0, 0]
                predictions.append(pred)

                # 시퀀스 업데이트
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred

            # 역스케일링
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            # 모델 저장
            if self.persistence and self.ticker:
                try:
                    metadata = {
                        'train_loss': history.history['loss'][-1],
                        'val_loss': history.history['val_loss'][-1],
                        'epochs_trained': len(history.history['loss']),
                        'sequence_length': self.sequence_length,
                        'd_model': self.d_model,
                        'num_heads': self.num_heads,
                        'num_layers': self.num_layers,
                        'data_size': len(prices)
                    }
                    self.persistence.save_keras_model(self.model, self.ticker, 'transformer', metadata, self.scaler)
                    # 저장 직후, 오래된 버전 정리(최신 5개만 유지)
                    self.persistence.delete_old_models(self.ticker, keep_latest=5)
                except Exception as e:
                    logger.warning(f"모델 저장 실패: {e}")

            return {
                'future_predictions': predictions,
                'model_type': 'Transformer',
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            }

        except Exception as e:
            logger.error(f"Transformer 실패: {e}")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback', 'error': str(e)}

    def _predict_only(self, prices, forecast_days):
        """저장된 모델로 예측만 수행 (재훈련 없음)"""
        try:
            X, y = self.prepare_sequences(prices)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # 미래 예측
            last_sequence = X[-1].reshape((1, self.sequence_length, 1))
            predictions = []

            for _ in range(forecast_days):
                pred = self.model.predict(last_sequence, verbose=0)[0, 0]
                predictions.append(pred)

                # 시퀀스 업데이트
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred

            # 역스케일링
            predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

            return {
                'future_predictions': predictions,
                'model_type': 'Transformer',
                'using_cached_model': True
            }

        except Exception as e:
            logger.error(f"Transformer 예측 실패: {e}")
            return {'future_predictions': np.full(forecast_days, prices[-1]), 'method': 'fallback', 'error': str(e)}


class ARIMAPredictor:
    """ARIMA 모델을 사용한 주가 예측"""
    
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
    
    def fit_predict(self, prices, forecast_days=5):
        """ARIMA 모델 피팅 및 예측"""
        if not STATSMODELS_AVAILABLE:
            # ✅ ARIMA 대체: 이동평균 기반 추세
            ma_window = min(10, len(prices) // 2)
            if ma_window < 2:
                ma_window = 2
            trend = np.mean(np.diff(prices[-ma_window:]))
            last_price = prices[-1]

            future_predictions = []
            for i in range(forecast_days):
                future_pred = last_price + trend * (i + 1)
                future_predictions.append(future_pred)

            return {
                'future_predictions': np.array(future_predictions),
                'method': 'moving_average_fallback',
                'trend': trend
            }
        
        try:
            # ARIMA 모델 피팅
            model = ARIMA(prices, order=self.order)
            self.model = model.fit()
            
            # 예측
            forecast_result = self.model.forecast(steps=forecast_days)
            confidence_intervals = self.model.get_forecast(steps=forecast_days).conf_int()
            
            return {
                'future_predictions': forecast_result.values if hasattr(forecast_result, 'values') else forecast_result,
                'confidence_intervals': confidence_intervals.values if hasattr(confidence_intervals, 'values') else confidence_intervals,
                'aic': self.model.aic,
                'bic': self.model.bic
            }
        
        except Exception as e:
            logger.error(f"ARIMA 모델 오류: {e}")
            # ✅ ARIMA 실패시 선형 추세 외삽
            trend = np.mean(np.diff(prices[-10:]))
            last_price = prices[-1]

            future_predictions = []
            for i in range(forecast_days):
                future_pred = last_price + trend * (i + 1)
                future_predictions.append(future_pred)

            return {
                'future_predictions': np.array(future_predictions),
                'method': 'linear_trend_fallback',
                'error': str(e)
            }

class MarketCorrelationAnalyzer:
    """시장 지수 상관관계 분석"""

    @staticmethod
    def get_market_indices(ticker):
        """종목에 맞는 시장 지수 선택"""
        if ticker.endswith('.KS') or ticker.endswith('.KQ'):
            # 한국 종목
            return {
                'KOSPI': '^KS11',
                'KOSDAQ': '^KQ11'
            }
        else:
            # 미국 종목
            return {
                'S&P500': '^GSPC',
                'Nasdaq': '^IXIC',
                'Dow': '^DJI'
            }

    @staticmethod
    def calculate_correlation(stock_symbol, period='1y'):
        """시장 지수와 상관계수 계산"""
        try:
            # 종목 데이터
            stock_data = get_stock_data(stock_symbol, period=period)
            if stock_data is None or len(stock_data) < 30:
                return {}

            # 지수 데이터
            indices = MarketCorrelationAnalyzer.get_market_indices(stock_symbol)
            correlations = {}

            for index_name, index_symbol in indices.items():
                try:
                    index_data = get_stock_data(index_symbol, period=period)
                    if index_data is not None and len(index_data) >= 30:
                        # 날짜 맞추기
                        merged = stock_data['Close'].to_frame().join(
                            index_data['Close'].to_frame(),
                            how='inner',
                            rsuffix='_index'
                        )

                        if len(merged) >= 30:
                            corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
                            correlations[index_name] = corr
                except Exception as e:
                    logger.warning(f"{index_name} 상관관계 계산 실패: {e}")

            return correlations

        except Exception as e:
            logger.error(f"시장 상관관계 분석 오류: {e}")
            return {}


class SectorAnalyzer:
    """섹터/산업 동향 분석"""

    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Healthcare': 'XLV',
        'Financial': 'XLF',
        'Energy': 'XLE',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Communication': 'XLC'
    }

    @staticmethod
    def get_sector_performance(period='1mo'):
        """섹터별 성과 분석"""
        sector_performance = {}

        for sector, etf in SectorAnalyzer.SECTOR_ETFS.items():
            try:
                data = get_stock_data(etf, period=period)
                if data is not None and len(data) >= 2:
                    start_price = data['Close'].iloc[0]
                    end_price = data['Close'].iloc[-1]
                    performance = ((end_price - start_price) / start_price) * 100
                    sector_performance[sector] = performance
            except Exception as e:
                logger.warning(f"{sector} 성과 분석 실패: {e}")

        return sector_performance

    @staticmethod
    def compare_with_sector(stock_symbol, sector_etf, period='1y'):
        """종목과 섹터 ETF 비교"""
        try:
            stock_data = get_stock_data(stock_symbol, period=period)
            sector_data = get_stock_data(sector_etf, period=period)

            if stock_data is None or sector_data is None:
                return None

            # 상대 성과
            stock_return = ((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0])
                           / stock_data['Close'].iloc[0]) * 100
            sector_return = ((sector_data['Close'].iloc[-1] - sector_data['Close'].iloc[0])
                            / sector_data['Close'].iloc[0]) * 100

            return {
                'stock_return': stock_return,
                'sector_return': sector_return,
                'outperformance': stock_return - sector_return
            }
        except Exception as e:
            logger.error(f"섹터 비교 오류: {e}")
            return None


class InstitutionalFlowAnalyzer:
    """외국인/기관 매매 동향 분석 (한국 종목)"""

    @staticmethod
    def is_korean_stock(ticker):
        """한국 종목 여부 확인"""
        return ticker.endswith('.KS') or ticker.endswith('.KQ')

    @staticmethod
    def fetch_institutional_data(ticker):
        """외국인/기관 매매 데이터 수집"""
        if not InstitutionalFlowAnalyzer.is_korean_stock(ticker):
            return None

        try:
            # FinanceDataReader 사용 (설치 필요: pip install finance-datareader)
            import FinanceDataReader as fdr
            from datetime import datetime, timedelta

            # 최근 60일 데이터 가져오기 (여유있게 30일 이상 확보)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)

            df = fdr.DataReader(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

            if df is not None and len(df) > 0 and 'ForeignBuy' in df.columns and 'InstitutionBuy' in df.columns:
                # 최근 30영업일 데이터 사용
                recent_30d = df.tail(30)

                return {
                    'foreign_net_buy': recent_30d['ForeignBuy'].sum() - recent_30d['ForeignSell'].sum(),
                    'institution_net_buy': recent_30d['InstitutionBuy'].sum() - recent_30d['InstitutionSell'].sum(),
                    'foreign_ownership': recent_30d['ForeignOwnership'].iloc[-1] if 'ForeignOwnership' in df.columns else None
                }
            else:
                return None

        except ImportError:
            logger.warning("FinanceDataReader 설치 필요: pip install finance-datareader")
            return None
        except Exception as e:
            logger.error(f"외국인/기관 데이터 수집 오류: {e}")
            return None


class MarketRegimeDetector:
    """시장 상황(상승장/하락장/횡보장) 감지 시스템"""

    @staticmethod
    def detect_regime(prices, window=50):
        """
        시장 상황 감지
        Returns: 'bull' (상승장), 'bear' (하락장), 'sideways' (횡보장)
        """
        if len(prices) < window:
            return 'sideways'

        recent_prices = prices[-window:]

        # 추세 분석
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        avg_price = np.mean(recent_prices)
        trend_pct = (trend / avg_price) * 100

        # 변동성 분석
        volatility = np.std(recent_prices) / np.mean(recent_prices)

        # 상승/하락 일수 비율
        price_changes = np.diff(recent_prices)
        up_days = np.sum(price_changes > 0)
        down_days = np.sum(price_changes < 0)
        up_ratio = up_days / len(price_changes) if len(price_changes) > 0 else 0.5

        # 시장 상황 판단
        if trend_pct > 0.5 and up_ratio > 0.55:
            regime = 'bull'  # 상승장
        elif trend_pct < -0.5 and up_ratio < 0.45:
            regime = 'bear'  # 하락장
        else:
            regime = 'sideways'  # 횡보장

        logger.info(f"시장 상황: {regime} (추세: {trend_pct:.2f}%, 변동성: {volatility:.2%}, 상승비율: {up_ratio:.1%})")

        return regime

    @staticmethod
    def get_regime_weights(regime):
        """시장 상황별 모델 가중치 반환"""
        weights = {
            'bull': {  # 상승장: 트렌드 추종 모델 강화
                'kalman': 0.20,
                'ml_models': 0.40,
                'arima': 0.15,
                'lstm': 0.15,
                'transformer': 0.10
            },
            'bear': {  # 하락장: 안정적인 모델 강화
                'kalman': 0.30,
                'ml_models': 0.25,
                'arima': 0.25,
                'lstm': 0.10,
                'transformer': 0.10
            },
            'sideways': {  # 횡보장: 균형잡힌 가중치
                'kalman': 0.20,
                'ml_models': 0.25,
                'arima': 0.25,
                'lstm': 0.15,
                'transformer': 0.10
            }
        }

        return weights.get(regime, weights['sideways'])


class EnsemblePredictor:
    """여러 모델을 결합한 앙상블 예측기 - 동적 가중치 + 시장 상황 인식"""

    def __init__(self, use_deep_learning=False, use_optimization=False, ticker=None):
        self.ticker = ticker
        self.kalman = KalmanFilterPredictor()
        self.ml_predictor = AdvancedMLPredictor(use_optimization=use_optimization, ticker=ticker) if (SKLEARN_AVAILABLE or XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE) else None
        self.arima = ARIMAPredictor()

        # 딥러닝 모델 (옵션)
        self.use_deep_learning = use_deep_learning and TENSORFLOW_AVAILABLE
        if self.use_deep_learning:
            self.lstm = LSTMPredictor(ticker=ticker)
            self.transformer = TransformerPredictor(ticker=ticker)
        else:
            self.lstm = None
            self.transformer = None

        # 초기 가중치 (동적으로 조정됨)
        self.weights = {
            'kalman': 0.25,
            'ml_models': 0.40 if self.ml_predictor else 0,
            'arima': 0.25 if self.ml_predictor else 0.75,
            'lstm': 0.05 if self.use_deep_learning else 0,
            'transformer': 0.05 if self.use_deep_learning else 0
        }

        # 모델 성능 추적
        self.performance_history = {
            'kalman': [],
            'ml_models': [],
            'arima': [],
            'lstm': [],
            'transformer': []
        }

        # ✅ 진행 상태 콜백
        self.progress_callback = None

        # 시장 상황
        self.current_regime = 'sideways'

    def update_weights_dynamically(self, validation_errors):
        """검증 오류를 기반으로 가중치 동적 조정"""
        # 역오류 가중치: 오류가 작을수록 높은 가중치
        inverse_errors = {}
        total_inverse = 0

        for model_name, error in validation_errors.items():
            if error > 0:
                inverse_errors[model_name] = 1.0 / error
                total_inverse += inverse_errors[model_name]

        # 정규화하여 가중치 계산
        if total_inverse > 0:
            for model_name in self.weights.keys():
                if model_name in inverse_errors:
                    self.weights[model_name] = inverse_errors[model_name] / total_inverse
                else:
                    self.weights[model_name] = 0

        logger.info(f"동적 가중치 업데이트: {self.weights}")
    
    def fit_predict(self, prices, forecast_days=5):
        """앙상블 예측 실행 - 동적 가중치 + 시장 상황 인식"""
        results = {}
        predictions = []
        validation_errors = {}

        logger.info("앙상블 예측 시작...")

        # 시장 상황 감지
        self.current_regime = MarketRegimeDetector.detect_regime(prices)
        regime_weights = MarketRegimeDetector.get_regime_weights(self.current_regime)

        # 검증 세트 분리 (마지막 10% 사용)
        split_point = int(len(prices) * 0.9)
        train_prices = prices[:split_point]
        val_prices = prices[split_point:]

        # 1. Kalman Filter 예측
        if self.progress_callback:
            self.progress_callback('kalman', 'Kalman Filter 예측 중...')
        logger.debug("Kalman Filter 실행 중...")
        try:
            kalman_result = self.kalman.fit_predict(train_prices, len(val_prices))
            results['kalman'] = self.kalman.fit_predict(prices, forecast_days)
            predictions.append(results['kalman']['future_predictions'])

            # 예측 결과 출력
            kalman_preds = results['kalman']['future_predictions']
            logger.info(f"Kalman 예측: 1일차={kalman_preds[0]:.2f} ({(kalman_preds[0]-prices[-1])/prices[-1]*100:+.2f}%), "
                       f"최종={kalman_preds[-1]:.2f} ({(kalman_preds[-1]-prices[-1])/prices[-1]*100:+.2f}%)")

            # 검증 오류 계산
            kalman_val_error = np.mean(np.abs(kalman_result['future_predictions'][:len(val_prices)] - val_prices))
            validation_errors['kalman'] = kalman_val_error
            logger.debug(f"Kalman 검증 MAE: {kalman_val_error:.2f}")
        except Exception as e:
            logger.warning(f"Kalman 실패: {e}")
            validation_errors['kalman'] = float('inf')

        # 2. ML 모델 예측 (XGBoost, LightGBM, Random Forest)
        if self.ml_predictor and len(prices) >= 50:
            if self.progress_callback:
                self.progress_callback('ml', 'ML 모델 검증 데이터 학습 중...')

            # ML 예측기에 진행 콜백 전달
            if hasattr(self.ml_predictor, 'progress_callback'):
                self.ml_predictor.progress_callback = self.progress_callback

            logger.debug("ML 모델들 훈련 중...")
            try:
                ml_val_result = self.ml_predictor.fit_predict(train_prices, len(val_prices))

                if self.progress_callback:
                    self.progress_callback('ml', 'ML 모델 전체 데이터 학습 중...')

                ml_result = self.ml_predictor.fit_predict(prices, forecast_days)
                results['ml_models'] = ml_result
                predictions.append(ml_result['future_predictions'])

                # 예측 결과 출력
                ml_preds = ml_result['future_predictions']
                logger.info(f"ML 앙상블 예측: 1일차={ml_preds[0]:.2f} ({(ml_preds[0]-prices[-1])/prices[-1]*100:+.2f}%), "
                           f"최종={ml_preds[-1]:.2f} ({(ml_preds[-1]-prices[-1])/prices[-1]*100:+.2f}%)")

                # 검증 오류 계산
                ml_val_error = np.mean(np.abs(ml_val_result['future_predictions'][:len(val_prices)] - val_prices))
                validation_errors['ml_models'] = ml_val_error
                logger.debug(f"ML 검증 MAE: {ml_val_error:.2f}")
            except Exception as e:
                logger.warning(f"ML 모델 실패: {e}")
                validation_errors['ml_models'] = float('inf')

        # 3. ARIMA 예측
        if self.progress_callback:
            self.progress_callback('arima', 'ARIMA 모델 실행 중...')
        logger.debug("ARIMA 모델 피팅 중...")
        try:
            arima_val_result = self.arima.fit_predict(train_prices, len(val_prices))
            arima_result = self.arima.fit_predict(prices, forecast_days)
            results['arima'] = arima_result
            predictions.append(arima_result['future_predictions'])

            # 예측 결과 출력
            arima_preds = arima_result['future_predictions']
            logger.info(f"ARIMA 예측: 1일차={arima_preds[0]:.2f} ({(arima_preds[0]-prices[-1])/prices[-1]*100:+.2f}%), "
                       f"최종={arima_preds[-1]:.2f} ({(arima_preds[-1]-prices[-1])/prices[-1]*100:+.2f}%)")

            # 검증 오류 계산
            arima_val_error = np.mean(np.abs(arima_val_result['future_predictions'][:len(val_prices)] - val_prices))
            validation_errors['arima'] = arima_val_error
            logger.debug(f"ARIMA 검증 MAE: {arima_val_error:.2f}")
        except Exception as e:
            logger.warning(f"ARIMA 실패: {e}")
            validation_errors['arima'] = float('inf')

        # 4. LSTM 예측 (딥러닝 모드)
        if self.use_deep_learning and self.lstm and len(prices) >= 100:
            if self.progress_callback:
                self.progress_callback('lstm', 'LSTM 딥러닝 모델 훈련 중...')
            logger.debug("LSTM 모델 훈련 중...")
            try:
                lstm_result = self.lstm.fit_predict(prices, forecast_days)
                if 'error' not in lstm_result:
                    results['lstm'] = lstm_result
                    predictions.append(lstm_result['future_predictions'])

                    # 검증 오류 (val_loss 사용)
                    validation_errors['lstm'] = lstm_result.get('val_loss', float('inf'))
                    logger.debug(f"LSTM 검증 Loss: {validation_errors['lstm']:.4f}")
            except Exception as e:
                logger.warning(f"LSTM 실패: {e}")
                validation_errors['lstm'] = float('inf')

        # 5. Transformer 예측 (딥러닝 모드)
        if self.use_deep_learning and self.transformer and len(prices) >= 100:
            if self.progress_callback:
                self.progress_callback('transformer', 'Transformer 모델 훈련 중...')
            logger.debug("Transformer 모델 훈련 중...")
            try:
                transformer_result = self.transformer.fit_predict(prices, forecast_days)
                if 'error' not in transformer_result:
                    results['transformer'] = transformer_result
                    predictions.append(transformer_result['future_predictions'])

                    # 검증 오류 (val_loss 사용)
                    validation_errors['transformer'] = transformer_result.get('val_loss', float('inf'))
                    logger.debug(f"Transformer 검증 Loss: {validation_errors['transformer']:.4f}")
            except Exception as e:
                logger.warning(f"Transformer 실패: {e}")
                validation_errors['transformer'] = float('inf')

        # 6. 가중치 결정: 시장 상황 기반 + 동적 조정
        if self.progress_callback:
            self.progress_callback('ensemble', '모델 가중치 최적화 중...')

        valid_errors = {k: v for k, v in validation_errors.items() if v != float('inf')}
        if valid_errors:
            # 성능 기반 동적 가중치
            self.update_weights_dynamically(valid_errors)

            # 시장 상황 가중치와 혼합 (70% 성능, 30% 시장상황)
            for model_name in self.weights.keys():
                if model_name in regime_weights:
                    self.weights[model_name] = (
                        0.7 * self.weights[model_name] +
                        0.3 * regime_weights[model_name]
                    )

            logger.info(f"최종 가중치 (시장상황: {self.current_regime}): {self.weights}")

        # 7. 앙상블 결합
        if self.progress_callback:
            self.progress_callback('ensemble', '최종 앙상블 예측 생성 중...')

        if len(predictions) > 1:
            # 가중평균 계산
            weighted_predictions = np.zeros(forecast_days)
            total_weight = 0

            model_order = ['kalman', 'ml_models', 'arima', 'lstm', 'transformer']
            active_models = [m for m in model_order if m in results]

            for i, model_name in enumerate(active_models):
                if self.weights.get(model_name, 0) > 0:
                    weighted_predictions += predictions[i] * self.weights[model_name]
                    total_weight += self.weights[model_name]

            ensemble_predictions = weighted_predictions / total_weight if total_weight > 0 else predictions[0]
        else:
            ensemble_predictions = predictions[0]

        # 최종 앙상블 예측 결과 출력
        logger.info(f"✅ 최종 앙상블 예측: 1일차={ensemble_predictions[0]:.2f} ({(ensemble_predictions[0]-prices[-1])/prices[-1]*100:+.2f}%), "
                   f"최종일={ensemble_predictions[-1]:.2f} ({(ensemble_predictions[-1]-prices[-1])/prices[-1]*100:+.2f}%)")

        # ✅ 신뢰도 계산: 검증 오류 기반 (낮을수록 신뢰도 높음)
        if len(valid_errors) > 0:
            # 평균 검증 오류를 신뢰도로 변환
            avg_error = np.mean(list(valid_errors.values()))
            avg_price = np.mean(prices[-30:])  # 최근 30일 평균가

            # 오류율 계산 (오류 / 평균가)
            error_rate = avg_error / avg_price if avg_price > 0 else 1.0

            # 신뢰도: 오류율이 5% 미만이면 높음, 20% 이상이면 낮음
            if error_rate < 0.05:
                confidence_score = 0.8 + (0.05 - error_rate) / 0.05 * 0.2  # 0.8~1.0
            elif error_rate < 0.20:
                confidence_score = 0.5 + (0.20 - error_rate) / 0.15 * 0.3  # 0.5~0.8
            else:
                confidence_score = max(0.3, 0.5 - (error_rate - 0.20) * 0.5)  # 0.3~0.5

            logger.info(f"검증 오류율: {error_rate*100:.2f}%, 신뢰도: {confidence_score*100:.1f}%")
        else:
            confidence_score = 0.5
            logger.warning("검증 오류 정보 없음, 기본 신뢰도 50% 적용")

        return {
            'ensemble_predictions': ensemble_predictions,
            'individual_results': results,
            'confidence_score': confidence_score,
            'model_weights': self.weights,
            'prediction_variance': np.var(predictions, axis=0) if len(predictions) > 1 else np.zeros(forecast_days),
            'validation_errors': validation_errors,
            'market_regime': self.current_regime
        }

class StockPredictor:
    """통합 주가 예측 시스템 - 딥러닝 + 하이퍼파라미터 최적화 지원"""

    def __init__(self, use_deep_learning=False, use_optimization=False, ticker=None):
        """
        Args:
            use_deep_learning: LSTM, Transformer 사용 여부
            use_optimization: Bayesian Optimization 사용 여부
            ticker: 주식 티커 (모델 저장/로드용)
        """
        self.ticker = ticker
        self.ensemble = EnsemblePredictor(
            use_deep_learning=use_deep_learning,
            use_optimization=use_optimization,
            ticker=ticker
        )
        self.use_deep_learning = use_deep_learning
        self.use_optimization = use_optimization
        self.progress_callback = None  # ✅ 진행 상태 콜백

    def set_progress_callback(self, callback):
        """진행 상태 콜백 설정"""
        self.progress_callback = callback
        if self.ensemble:
            self.ensemble.progress_callback = callback
    
    def get_stock_data(self, symbol, period=None):
        """
        주식 데이터 가져오기 - 동적 기간 설정

        Args:
            symbol: 주식 티커
            period: 기간 (None이면 자동 결정)
        """
        try:
            # 기간 자동 결정
            if period is None:
                try:
                    from optimal_period_config import get_optimal_training_period
                    period = get_optimal_training_period(symbol)
                    logger.info(f"📅 {symbol} 최적 훈련 기간: {period}")
                except ImportError:
                    period = "3y"  # 기본값: 3년 (2y → 3y 개선)
                    logger.debug(f"기본 훈련 기간 사용: {period}")

            data = get_stock_data(symbol, period=period)
            return data
        except Exception as e:
            logger.error(f"데이터 가져오기 실패: {e}")
            return None
    
    def predict_stock_price(self, symbol, forecast_days=5, show_plot=True):
        """종목의 미래 주가 예측 - 시장 분석 추가"""
        logger.info(f"{symbol} 주가 예측 시작...")

        # 1. 데이터 수집
        if self.progress_callback:
            self.progress_callback('data', '데이터 수집 중...')

        data = self.get_stock_data(symbol)
        if data is None or len(data) < 50:
            return {"error": "충분한 데이터가 없습니다"}

        prices = data['Close'].values
        dates = data.index

        # === 증분 학습: 최신 데이터로 XGBoost/LightGBM 미세 업데이트 ===
        try:
            mlp = self.ensemble.ml_predictor if hasattr(self, 'ensemble') else None
            if mlp and mlp.persistence and self.ticker:
                # 최신 시퀀스 일부만 사용해 빠른 증분 업데이트
                X_all, y_all = mlp.prepare_data(prices)
                tail = min(200, len(y_all))  # 최근 200개 샘플 사용 (데이터 적으면 가능한 범위)
                if tail > 0:
                    X_new, y_new = X_all[-tail:], y_all[-tail:]

                    if mlp.persistence.supports_incremental_learning('xgboost'):
                        mlp.persistence.incremental_train_xgboost(self.ticker, X_new, y_new, n_estimators_add=50)

                    if mlp.persistence.supports_incremental_learning('lightgbm'):
                        mlp.persistence.incremental_train_lightgbm(self.ticker, X_new, y_new, n_estimators_add=50)
        except Exception as e:
            logger.warning(f"증분 학습 건너뜀: {e}")

        logger.info(f"분석 기간: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
        logger.info(f"데이터 포인트: {len(prices)}개")

        # 1-1. 시장 지수 상관관계 분석
        if self.progress_callback:
            self.progress_callback('market_analysis', '시장 분석 중...')

        logger.info("시장 지수 상관관계 분석 중...")
        market_correlations = MarketCorrelationAnalyzer.calculate_correlation(symbol)

        # 1-2. 섹터 성과 분석 (미국 종목만)
        sector_info = None
        if not (symbol.endswith('.KS') or symbol.endswith('.KQ')):
            logger.info("섹터 성과 분석 중...")
            sector_performance = SectorAnalyzer.get_sector_performance()
            sector_info = sector_performance

        # 1-3. 외국인/기관 매매 동향 (한국 종목만)
        institutional_flow = None
        if InstitutionalFlowAnalyzer.is_korean_stock(symbol):
            logger.info("외국인/기관 매매 동향 분석 중...")
            institutional_flow = InstitutionalFlowAnalyzer.fetch_institutional_data(symbol)

        # 2. 앙상블 예측 실행
        if self.progress_callback:
            self.progress_callback('ensemble', '앙상블 예측 실행 중...')

        result = self.ensemble.fit_predict(prices, forecast_days)

        # 진행 상태: 결과 정리 중
        if self.progress_callback:
            self.progress_callback('complete', '결과 정리 중...')

        # 3. 결과 정리
        last_price = prices[-1]
        predicted_prices = result['ensemble_predictions']

        # 예측 정확도 추정
        confidence = result['confidence_score']
        
        # 미래 날짜 생성 (영업일 기준)
        future_dates = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), 
                                     periods=forecast_days)
        
        # 수익률 계산
        returns = [(pred / last_price - 1) * 100 for pred in predicted_prices]
        
        # 4. 결과 반환 (시장 분석 정보 추가)
        prediction_result = {
            'symbol': symbol,
            'current_price': last_price,
            'predicted_prices': predicted_prices,
            'future_dates': future_dates,
            'expected_returns': returns,
            'confidence_score': confidence,
            'model_weights': result['model_weights'],
            'recommendation': self._generate_recommendation(returns, confidence),
            'models_used': self._get_models_used(),
            # 새로운 시장 분석 정보
            'market_correlations': market_correlations,
            'sector_performance': sector_info,
            'institutional_flow': institutional_flow,
            'market_regime': result.get('market_regime', 'unknown')
        }
        
        # 5. 그래프 표시
        if show_plot:
            self._plot_predictions(dates, prices, future_dates, predicted_prices, symbol)
        
        return prediction_result
    
    def _get_models_used(self):
        """사용된 모델 목록 반환"""
        models = ['Kalman Filter', 'ARIMA']

        if SKLEARN_AVAILABLE:
            models.append('Random Forest')
        if XGBOOST_AVAILABLE:
            models.append('XGBoost')
        if LIGHTGBM_AVAILABLE:
            models.append('LightGBM')
        if STATSMODELS_AVAILABLE:
            models.append('ARIMA (Full)')

        if self.use_deep_learning:
            if TENSORFLOW_AVAILABLE:
                models.extend(['LSTM', 'Transformer'])

        if self.use_optimization:
            models.append('+ Bayesian Optimization')

        return models
    
    def _generate_recommendation(self, returns, confidence):
        """예측 결과 기반 투자 추천"""
        avg_return = np.mean(returns)
        
        if confidence > 0.7:
            if avg_return > 5:
                return "강력 매수 추천"
            elif avg_return > 2:
                return "매수 추천"
            elif avg_return < -5:
                return "매도 추천"
            elif avg_return < -2:
                return "매도 고려"
            else:
                return "보유"
        else:
            return "불확실 - 신중한 판단 필요"
    
    def _plot_predictions(self, historical_dates, historical_prices, 
                         future_dates, predicted_prices, symbol):
        """예측 결과 시각화"""
        plt.figure(figsize=(12, 8))
        
        # 과거 데이터
        plt.plot(historical_dates[-60:], historical_prices[-60:], 
                'b-', label='실제 가격', linewidth=2)
        
        # 예측 데이터
        plt.plot(future_dates, predicted_prices, 
                'r--', label='예측 가격', linewidth=2, marker='o')
        
        # 연결선
        plt.plot([historical_dates[-1], future_dates[0]], 
                [historical_prices[-1], predicted_prices[0]], 
                'g:', linewidth=1)
        
        plt.title(f'{symbol} 주가 예측 결과 (TensorFlow 없음)', fontsize=16)
        plt.xlabel('날짜')
        plt.ylabel('가격')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _backtest_single_point(self, args):
        """단일 백테스팅 포인트 실행 (병렬/순차 처리용)"""
        i, test_point, train_prices, actual_future_prices, forecast_days, test_date = args

        try:
            # ProcessPool 병렬 처리: 각 프로세스마다 독립 인스턴스 생성
            # (GIL 우회로 진정한 병렬 처리, 모듈은 각 프로세스에서 1회 로딩)
            ensemble = EnsemblePredictor(
                use_deep_learning=self.use_deep_learning,
                use_optimization=False  # 백테스팅에서는 최적화 비활성화 (속도 향상)
            )

            result = ensemble.fit_predict(train_prices, forecast_days)
            predicted_prices = result['ensemble_predictions']

            # 실제 vs 예측 비교
            last_train_price = train_prices[-1]
            actual_return = (actual_future_prices[-1] - last_train_price) / last_train_price * 100
            predicted_return = (predicted_prices[-1] - last_train_price) / last_train_price * 100

            # MAE, RMSE 계산
            mae = np.mean(np.abs(predicted_prices - actual_future_prices))
            rmse = np.sqrt(np.mean((predicted_prices - actual_future_prices) ** 2))
            mape = np.mean(np.abs((actual_future_prices - predicted_prices) / actual_future_prices)) * 100

            # 개별 모델 예측 수집 (모델별 성능 분석용)
            individual_predictions = {}

            # 디버깅: result 키 확인
            logger.debug(f"Result keys: {result.keys()}")

            if 'individual_results' in result:
                logger.debug(f"Individual results found: {result['individual_results'].keys()}")
                for model_name, model_result in result['individual_results'].items():
                    logger.debug(f"Processing model: {model_name}, type: {type(model_result)}")
                    if isinstance(model_result, dict) and 'future_predictions' in model_result:
                        model_pred_price = model_result['future_predictions'][-1]

                        model_pred_return = (model_pred_price - last_train_price) / last_train_price * 100
                        individual_predictions[model_name] = {
                            'predicted_return': model_pred_return,
                            'direction_match': (actual_return > 0) == (model_pred_return > 0)
                        }
                        logger.debug(f"{model_name} 예측 추가: {model_pred_return:.2f}%")
                    else:
                        logger.debug(f"{model_name} 스킵: future_predictions 없음")
            else:
                logger.warning("individual_results 키가 없습니다!")

            return {
                'success': True,
                'index': i,
                'date': test_date,
                'actual_return': actual_return,
                'predicted_return': predicted_return,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'direction_match': (actual_return > 0) == (predicted_return > 0),
                'individual_predictions': individual_predictions  # 개별 모델 예측
            }

        except Exception as e:
            logger.warning(f"백테스팅 {i+1} 실패: {e}")
            return {
                'success': False,
                'index': i,
                'error': str(e)
            }

    def backtest_predictions(self, ticker, test_periods=30, forecast_days=7,
                           progress_callback=None, use_parallel=False, cancel_callback=None):
        """
        백테스팅: 과거 데이터로 예측 성능 검증

        Args:
            ticker: 종목 코드
            test_periods: 테스트할 기간 수
            forecast_days: 예측 일수
            progress_callback: 진행 상태 콜백 (current, total, message)
            use_parallel: 병렬 처리 여부
            cancel_callback: 중지 확인 콜백

        Returns:
            (summary, error): 백테스팅 결과 요약과 에러 메시지
        """
        try:
            # 초기 진행 상태 업데이트
            if progress_callback:
                progress_callback(0, test_periods, f"데이터 로딩 중... ({ticker})")

            # 데이터 가져오기 (충분한 기간)
            data = self.get_stock_data(ticker, period="2y")
            if data is None or len(data) < 100:
                return None, "충분한 데이터가 없습니다"

            prices = data['Close'].values
            dates = data.index

            # 백테스팅 작업 준비
            tasks = []
            for i in range(test_periods):
                test_point = len(prices) - (test_periods - i) * forecast_days - forecast_days

                if test_point < 100:
                    continue

                train_prices = prices[:test_point]
                actual_future_prices = prices[test_point:test_point + forecast_days]

                if len(actual_future_prices) < forecast_days:
                    continue

                tasks.append((
                    i,
                    test_point,
                    train_prices,
                    actual_future_prices,
                    forecast_days,
                    dates[test_point]
                ))

            if len(tasks) == 0:
                return None, "백테스팅할 데이터 포인트가 없습니다"

            # 작업 준비 완료 알림
            if progress_callback:
                progress_callback(0, len(tasks), f"백테스팅 시작 중... ({len(tasks)}개 작업)")

            results = []
            errors = []
            actual_returns = []
            predicted_returns = []

            # 병렬 처리 vs 순차 처리
            if use_parallel:
                import time
                start_time = time.time()

                # ProcessPoolExecutor로 병렬 실행 (GIL 우회, 진정한 병렬 처리)
                max_workers = min(multiprocessing.cpu_count(), len(tasks))
                logger.info(f"🚀 병렬 처리 모드: {max_workers}개 프로세스 사용 (CPU 코어: {multiprocessing.cpu_count()}, 작업 수: {len(tasks)})")

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(self._backtest_single_point, task): task for task in tasks}

                    completed = 0
                    for future in as_completed(futures):
                        # 중지 확인
                        if cancel_callback and cancel_callback():
                            executor.shutdown(wait=False, cancel_futures=True)
                            return None, "사용자에 의해 중지됨"

                        try:
                            result = future.result()
                        except Exception as e:
                            completed += 1
                            logger.error(f"백테스팅 작업 실패: {e}")
                            errors.append(str(e))
                            if progress_callback:
                                progress_callback(completed, len(tasks), f"오류 발생 {completed}/{len(tasks)}")
                            continue

                        completed += 1

                        # 진행 상태 업데이트
                        if progress_callback:
                            task = futures[future]
                            progress_callback(completed, len(tasks),
                                            f"백테스팅 {completed}/{len(tasks)}: {task[5]:%Y-%m-%d}")

                        if result.get('success', False):
                            results.append({
                                'date': result['date'],
                                'actual_return': result['actual_return'],
                                'predicted_return': result['predicted_return'],
                                'mae': result['mae'],
                                'rmse': result['rmse'],
                                'mape': result['mape'],
                                'direction_match': result['direction_match'],
                                'individual_predictions': result.get('individual_predictions', {})
                            })
                            actual_returns.append(result['actual_return'])
                            predicted_returns.append(result['predicted_return'])
                        else:
                            errors.append(result.get('error', 'Unknown error'))

                elapsed_time = time.time() - start_time
                logger.info(f"⏱️ 병렬 처리 완료: {elapsed_time:.2f}초 (평균 {elapsed_time/len(tasks):.2f}초/작업)")

            else:
                # 순차 처리
                import time
                start_time = time.time()
                logger.info("순차 처리 모드")

                for i, task in enumerate(tasks):
                    # 중지 확인
                    if cancel_callback and cancel_callback():
                        return None, "사용자에 의해 중지됨"

                    # 진행 상태 업데이트
                    if progress_callback:
                        progress_callback(i + 1, len(tasks),
                                        f"백테스팅 {i+1}/{len(tasks)}: {task[5]:%Y-%m-%d}")

                    result = self._backtest_single_point(task)

                    if result['success']:
                        results.append({
                            'date': result['date'],
                            'actual_return': result['actual_return'],
                            'predicted_return': result['predicted_return'],
                            'mae': result['mae'],
                            'rmse': result['rmse'],
                            'mape': result['mape'],
                            'direction_match': result['direction_match'],
                            'individual_predictions': result.get('individual_predictions', {})
                        })
                        actual_returns.append(result['actual_return'])
                        predicted_returns.append(result['predicted_return'])
                    else:
                        errors.append(result['error'])

                elapsed_time = time.time() - start_time
                logger.info(f"⏱️ 순차 처리 완료: {elapsed_time:.2f}초 (평균 {elapsed_time/len(tasks):.2f}초/작업)")

            # 요약 통계 계산
            if len(results) == 0:
                return None, "백테스팅 결과가 없습니다"

            direction_accuracy = np.mean([r['direction_match'] for r in results]) * 100
            avg_mae = np.mean([r['mae'] for r in results])
            avg_rmse = np.mean([r['rmse'] for r in results])
            avg_mape = np.mean([r['mape'] for r in results])

            # 상관관계
            if len(actual_returns) > 1:
                correlation = np.corrcoef(actual_returns, predicted_returns)[0, 1]
            else:
                correlation = 0

            # 상세 분석 추가
            bull_correct = sum(1 for r in results if r['actual_return'] > 0 and r['direction_match'])
            bull_total = sum(1 for r in results if r['actual_return'] > 0)
            bear_correct = sum(1 for r in results if r['actual_return'] <= 0 and r['direction_match'])
            bear_total = sum(1 for r in results if r['actual_return'] <= 0)

            pred_bull = sum(1 for r in results if r['predicted_return'] > 0)
            pred_bear = len(results) - pred_bull

            # 개별 모델 성능 분석
            logger.debug(f"📊 모델 성능 분석 시작 - 총 {len(results)}개 결과")
            model_performance = {}
            for idx, result in enumerate(results):
                logger.debug(f"결과 {idx}: 'individual_predictions' 존재 = {'individual_predictions' in result}")
                if 'individual_predictions' in result:
                    logger.debug(f"결과 {idx} individual_predictions: {result['individual_predictions'].keys()}")
                    for model_name, model_pred in result['individual_predictions'].items():
                        if model_name not in model_performance:
                            model_performance[model_name] = {'correct': 0, 'total': 0}

                        model_performance[model_name]['total'] += 1
                        if model_pred['direction_match']:
                            model_performance[model_name]['correct'] += 1
                else:
                    logger.warning(f"결과 {idx}에 individual_predictions 없음: {result.keys()}")

            logger.debug(f"최종 model_performance: {model_performance}")

            # 모델별 적중률 계산
            model_accuracies = {}
            for model_name, perf in model_performance.items():
                if perf['total'] > 0:
                    accuracy = (perf['correct'] / perf['total']) * 100
                    model_accuracies[model_name] = accuracy
                    logger.info(f"📊 {model_name} 적중률: {accuracy:.1f}% ({perf['correct']}/{perf['total']})")

            logger.debug(f"최종 model_accuracies: {model_accuracies}")

            summary = {
                'ticker': ticker,
                'test_periods': len(results),
                'forecast_days': forecast_days,
                'direction_accuracy': direction_accuracy,
                'avg_mae': avg_mae,
                'avg_rmse': avg_rmse,
                'avg_mape': avg_mape,
                'correlation': correlation,
                'actual_returns': actual_returns,
                'predicted_returns': predicted_returns,
                'results': results,
                'errors': errors,
                # 상세 분석
                'bull_accuracy': (bull_correct / bull_total * 100) if bull_total > 0 else 0,
                'bear_accuracy': (bear_correct / bear_total * 100) if bear_total > 0 else 0,
                'bull_total': bull_total,
                'bear_total': bear_total,
                'pred_bull': pred_bull,
                'pred_bear': pred_bear,
                # 모델별 성능
                'model_accuracies': model_accuracies
            }

            return summary, None

        except Exception as e:
            logger.error(f"백테스팅 오류: {e}")
            return None, str(e)

# 사용 예제
def example_usage():
    """사용 예제"""
    # 옵션 설정:
    # use_deep_learning=True: LSTM, Transformer 사용 (TensorFlow 필요)
    # use_optimization=True: Bayesian Optimization 사용 (scikit-optimize 필요)
    predictor = StockPredictor(
        use_deep_learning=True,      # 딥러닝 모델 사용
        use_optimization=False        # 하이퍼파라미터 최적화 (시간이 오래 걸림)
    )

    logger.info("사용 가능한 라이브러리:")
    logger.info(f"- scikit-learn: {'사용 가능' if SKLEARN_AVAILABLE else '사용 불가'}")
    logger.info(f"- XGBoost: {'사용 가능' if XGBOOST_AVAILABLE else '사용 불가'}")
    logger.info(f"- LightGBM: {'사용 가능' if LIGHTGBM_AVAILABLE else '사용 불가'}")
    logger.info(f"- statsmodels: {'사용 가능' if STATSMODELS_AVAILABLE else '사용 불가'}")
    logger.info(f"- TensorFlow: {'사용 가능' if TENSORFLOW_AVAILABLE else '사용 불가'}")
    logger.info(f"- Bayesian Opt: {'사용 가능' if HYPEROPT_AVAILABLE else '사용 불가'}")

    # 예제 1: 애플 주식 예측
    logger.info("=" * 50)
    logger.info("APPLE (AAPL) 주가 예측")
    logger.info("=" * 50)

    result = predictor.predict_stock_price('AAPL', forecast_days=7, show_plot=False)

    if 'error' not in result:
        logger.info(f"현재가: ${result['current_price']:.2f}")
        logger.info(f"예측 가격: {[f'${p:.2f}' for p in result['predicted_prices']]}")
        logger.info(f"예상 수익률: {[f'{r:.1f}%' for r in result['expected_returns']]}")
        logger.info(f"신뢰도: {result['confidence_score']:.1%}")
        logger.info(f"추천: {result['recommendation']}")
        logger.info(f"사용 모델: {', '.join(result['models_used'])}")

if __name__ == "__main__":
    example_usage()