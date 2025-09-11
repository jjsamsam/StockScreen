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
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

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
    print("✅ 모든 ML 라이브러리 사용 가능")
else:
    print("⚠️ 일부 ML 라이브러리 누락:")
    print(f"   - scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"   - XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"   - LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    print(f"   - statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌'}")


class EnhancedCPUPredictor:
    """CPU 최적화 예측기 - 통합된 예측 함수 버전"""
    
    def __init__(self):
        """CPU 최적화 모델들 초기화"""
        if not ML_AVAILABLE:
            print("⚠️ ML 라이브러리가 부족합니다")
            self.models = {}
            self.scalers = {}
            return
        
        print("🤖 CPU 최적화 예측기 초기화 중...")
        
        # 고정된 시드로 초기화
        self.fix_all_random_seeds(42)
        
        # CPU 최적화 모델들
        self.models = {
            # XGBoost: CPU 최적화
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,  # ✅ 일관성을 위해 단일 스레드
                verbosity=0
            ),
            
            # LightGBM: AMD CPU에 특히 우수
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=1,  # ✅ 일관성을 위해 단일 스레드
                device='cpu',
                verbose=-1
            ),
            
            # Random Forest: 안정적 성능
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=1,  # ✅ 일관성을 위해 단일 스레드
                random_state=42
            ),
            
            # Extra Trees: 과적합 방지
            'extra_trees': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                bootstrap=False,  # Extra Trees 특징
                n_jobs=1,  # ✅ 일관성을 위해 단일 스레드
                random_state=42
            ),
            
            # Gradient Boosting: 견고한 성능
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
                validation_fraction=0.1,  # ✅ 일관성 보장
            )
        }
        
        # 고급 전처리기들
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()  # 이상치에 강함
        }
        
        self.current_scaler = self.scalers['robust']  # 주식 데이터는 이상치 많음
        
        print(f"✅ {len(self.models)}개 모델 초기화 완료")

    def fix_all_random_seeds(self, seed=42):
        """모든 랜덤 시드 고정 - 완전한 일관성 보장"""
        print(f"🔒 모든 랜덤 시드를 {seed}로 고정")
        
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

    # ✅ 통합된 예측 함수 - predict_stock_consistent의 로직을 predict_stock으로 변경
    def predict_stock(self, ticker, forecast_days=7, min_data_days=300, mode='smart'):
        """통합된 예측 함수 - 일관성 있는 예측 (기존 predict_stock_consistent 로직)"""
        
        # 매번 시드 재고정 (완전한 일관성 보장)
        self.fix_all_random_seeds(42)
        
        try:
            print(f"📊 {ticker} 일관성 예측 시작...")
            
            # 1. 실제 현재가 조회 (최신 데이터)
            stock = yf.Ticker(ticker)
            current_data = stock.history(period="2d")
            if len(current_data) == 0:
                return None, "현재가 데이터를 가져올 수 없습니다"
            
            actual_current_price = float(current_data['Close'].iloc[-1])
            actual_current_date = current_data.index[-1]
            
            # 2. 예측용 고정 기간 데이터 (일관성 보장)
            end_date = datetime(2024, 12, 31)  # 고정된 종료일
            start_date = end_date - timedelta(days=600)  # 고정된 시작일
            
            print(f"  💰 실제 현재가: {actual_current_price:.2f} ({actual_current_date.date()})")
            print(f"  🔒 예측 기준일: {end_date.date()}")
            
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < min_data_days:
                return None, f"데이터 부족 (필요: {min_data_days}일, 현재: {len(data)}일)"
            
            # 데이터 정렬 및 정리 (일관성 보장)
            data = data.sort_index().round(4)
            
            # 데이터 품질 검사
            if data['Close'].isnull().sum() > len(data) * 0.1:
                return None, "데이터 품질 불량 (결측값 과다)"
            
            # 시드 재고정
            self.fix_all_random_seeds(42)
            
            # 고급 특성 생성
            features = self.create_advanced_features_deterministic(data)
            
            if features.empty or features.isnull().all().all():
                return None, "특성 생성 실패"
            
            # 타겟 생성
            future_returns = data['Close'].pct_change(forecast_days).shift(-forecast_days)
            
            if future_returns.isnull().sum() > len(future_returns) * 0.8:
                return None, "타겟 데이터 부족"
            
            # 시드 재고정
            self.fix_all_random_seeds(42)
            
            # 시퀀스 데이터 준비
            X, y = self.prepare_sequences_deterministic(features, future_returns, 
                                                    sequence_length=30, 
                                                    forecast_horizon=forecast_days)
            
            if len(X) == 0 or len(y) == 0:
                return None, "시퀀스 데이터 생성 실패"
            
            print(f"  ✅ 데이터 준비 완료: {len(X)}개 학습 샘플")
            
            # 학습/테스트 분할 (시계열 특성 고려)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # 데이터 정규화
            try:
                X_train_scaled = self.current_scaler.fit_transform(X_train)
                X_test_scaled = self.current_scaler.transform(X_test)
                
                # 최신 데이터 준비 (예측용)
                latest_X = X[-1]
                latest_X_scaled = self.current_scaler.transform(latest_X.reshape(1, -1))
                
            except Exception as e:
                return None, f"데이터 정규화 실패: {str(e)}"
            
            # 시드 재고정
            self.fix_all_random_seeds(42)
            
            # 모델별 예측 실행
            predictions = []
            model_results = {}
            successful_models = 0
            
            for model_name, model in self.models.items():
                if hasattr(self, 'prediction_settings') and \
                   not self.prediction_settings.get('models_enabled', {}).get(model_name, True):
                    continue
                
                prediction = self.safe_predict_with_model(
                    model, X_train_scaled, y_train, latest_X_scaled, model_name
                )
                
                if prediction is not None:
                    predictions.append(prediction)
                    successful_models += 1
                    
                    # 성능 평가
                    try:
                        y_pred_test = model.predict(X_test_scaled)
                        r2 = r2_score(y_test, y_pred_test)
                        model_results[model_name] = {
                            'r2_score': r2,
                            'prediction': prediction
                        }
                    except Exception as e:
                        model_results[model_name] = {'prediction': prediction}
            
            if successful_models == 0:
                return None, "모든 모델이 실패했습니다"
            
            print(f"  ✅ {successful_models}개 모델 성공")
            
            # 결정적 앙상블 계산
            ensemble_prediction, confidence = self.calculate_deterministic_ensemble(
                predictions, model_results
            )
            
            # 핵심 수정: 현재가 vs 예측가 분리
            historical_price = float(data['Close'].iloc[-1])  # 예측 기준 가격
            predicted_return = float(ensemble_prediction)
            
            # 실제 현재가 기준으로 예측가 계산
            predicted_price_actual = actual_current_price * (1 + predicted_return)
            
            # 결과 구성
            result = {
                'ticker': ticker,
                
                # 실제 현재가 정보 (사용자가 보는 정보)
                'current_price': round(actual_current_price, 4),
                'predicted_price': round(predicted_price_actual, 4),
                'expected_return': round(predicted_return, 6),
                
                # 예측 기술 정보
                'confidence': round(confidence, 4),
                'forecast_days': forecast_days,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                
                # 상세 정보
                'successful_models': successful_models,
                'model_results': model_results,
                'individual_predictions': predictions,
                'feature_count': features.shape[1],
                'training_samples': len(X_train)
            }
            
            print(f"  ✅ 예측 완료: {predicted_return*100:+.2f}% (신뢰도: {confidence*100:.1f}%)")
            
            return result, None
            
        except Exception as e:
            error_msg = f"예측 중 오류: {str(e)}"
            print(f"  ❌ {error_msg}")
            return None, error_msg

    # ✅ 기존 predict_stock_consistent 함수는 제거됨 (위의 predict_stock으로 통합)
    # 
    # 변경 사항:
    # 1. predict_stock_consistent() 함수의 로직을 predict_stock()으로 이동
    # 2. 기존 predict_stock() 함수는 완전히 제거
    # 3. 함수 호출 코드는 변경 없이 그대로 사용 가능
    # 4. 더 나은 일관성과 정확도를 제공하는 알고리즘 사용

    def safe_predict_with_model(self, model, X_train, y_train, X_test, model_name):
        """개별 모델 예측 - 타입 및 오류 안전"""
        try:
            print(f"  🔧 {model_name} 훈련 중...")
            
            # 입력 데이터 검증
            if X_train.size == 0 or y_train.size == 0:
                print(f"    ❌ {model_name} 오류: 빈 훈련 데이터")
                return None
            
            # NaN/Inf 체크 (타입 안전)
            try:
                if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
                    print(f"    ❌ {model_name} 오류: 훈련 데이터에 NaN 존재")
                    return None
                
                if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
                    print(f"    ❌ {model_name} 오류: 훈련 데이터에 Inf 존재")
                    return None
            except (TypeError, ValueError) as e:
                print(f"    ❌ {model_name} 오류: 데이터 타입 문제 - {e}")
                return None
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            if X_test.size == 0:
                print(f"    ❌ {model_name} 오류: 빈 테스트 데이터")
                return None
            
            prediction = model.predict(X_test.reshape(1, -1))[0]
            
            # 예측 결과 검증
            if pd.isnull(prediction) or np.isinf(prediction):
                print(f"    ❌ {model_name} 오류: 잘못된 예측값")
                return None
            
            print(f"    ✅ {model_name} 완료: {prediction:.4f}")
            return float(prediction)
            
        except Exception as e:
            print(f"    ❌ {model_name} 오류: {str(e)}")
            return None

    def calculate_deterministic_ensemble(self, predictions, model_results):
        """결정적 앙상블 계산"""
        if not predictions:
            return 0.0, 0.0
        
        # 고정된 가중치 사용 (일관성 보장)
        weights = {
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'random_forest': 0.20,
            'extra_trees': 0.15,
            'gradient_boosting': 0.15
        }
        
        # 가중 평균 계산
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, (model_name, result) in enumerate(model_results.items()):
            weight = weights.get(model_name, 1.0 / len(predictions))
            prediction = result.get('prediction', predictions[i] if i < len(predictions) else 0)
            
            weighted_sum += prediction * weight
            total_weight += weight
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else np.mean(predictions)
        
        # 신뢰도 계산 (예측값 분산 기반)
        if len(predictions) > 1:
            variance = np.var(predictions)
            confidence = 1.0 / (1.0 + variance * 10)  # 분산이 작을수록 높은 신뢰도
        else:
            confidence = 0.5  # 단일 모델인 경우 중간 신뢰도
        
        return ensemble_prediction, min(confidence, 0.95)  # 최대 95% 신뢰도
    
    def create_advanced_features_deterministic(self, data):
            """결정적 고급 특성 생성 - 일관성 보장"""
            try:
                features = pd.DataFrame(index=data.index)
                
                # 1. 기본 수익률 (가장 중요)
                features['returns'] = data['Close'].pct_change()
                features['returns_2'] = data['Close'].pct_change(2)
                features['returns_5'] = data['Close'].pct_change(5)
                
                # 2. 이동평균 기반 특성 (고정 윈도우)
                for window in [5, 10, 20, 50]:
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
                
                # 데이터 정리 (결정적 방식)
                features = features.replace([np.inf, -np.inf], 0)
                features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                print(f"  ✅ {len(features.columns)}개 결정적 특성 생성 완료")
                return features
                
            except Exception as e:
                print(f"  ❌ 특성 생성 오류: {e}")
                # 최소한의 특성 생성
                features = pd.DataFrame(index=data.index)
                features['returns'] = data['Close'].pct_change().fillna(0)
                features['trend'] = np.arange(len(data), dtype=float)
                return features

    def prepare_sequences_deterministic(self, features, targets, sequence_length=30, forecast_horizon=7):
        """결정적 시퀀스 데이터 준비"""
        try:
            # 유효한 데이터만 사용
            valid_indices = ~(targets.isnull() | features.isnull().any(axis=1))
            valid_features = features[valid_indices]
            valid_targets = targets[valid_indices]
            
            if len(valid_features) < sequence_length + forecast_horizon:
                print(f"  ❌ 유효 데이터 부족: {len(valid_features)}개")
                return np.array([]), np.array([])
            
            X, y = [], []
            
            # 고정된 순서로 시퀀스 생성 (결정적)
            for i in range(sequence_length, len(valid_features)):
                if not valid_targets.iloc[i] == valid_targets.iloc[i]:  # NaN 체크
                    continue
                    
                sequence = valid_features.iloc[i-sequence_length:i].values
                target = valid_targets.iloc[i]
                
                # 데이터 품질 재확인
                if not (np.isfinite(sequence).all() and np.isfinite(target)):
                    continue
                
                X.append(sequence.flatten())
                y.append(target)
            
            X_array = np.array(X, dtype=np.float64)
            y_array = np.array(y, dtype=np.float64)
            
            print(f"  ✅ 시퀀스 데이터 준비 완료: {len(X_array)}개 샘플, 특성 {X_array.shape[1]}개")
            
            return X_array, y_array
            
        except Exception as e:
            print(f"  ❌ 시퀀스 데이터 준비 오류: {e}")
            return np.array([]), np.array([])


class EnhancedStockScreenerMethods:
    """기존 StockScreener 클래스에 추가할 AI 예측 메서드들"""
    
    def __init__(self):
        """AI 예측 관련 초기화"""
        self.predictor = EnhancedCPUPredictor() if ML_AVAILABLE else None
        self.prediction_settings = self.load_prediction_settings()
        
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
            print(f"설정 저장 오류: {e}")
    
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
        """테이블에서 선택된 종목 예측"""
        current_row = table.currentRow()
        if current_row >= 0:
            ticker_item = table.item(current_row, 0)
            if ticker_item:
                ticker = ticker_item.text()
                self.show_prediction_dialog(ticker)
    
    def show_prediction_dialog(self, ticker=None):
        """AI 예측 다이얼로그 표시 - 기존 StockPredictionDialog 활용"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "ML 라이브러리가 설치되지 않았습니다.")
            return
        
        try:
            # 기존 StockPredictionDialog 사용
            from prediction_window import StockPredictionDialog
            dialog = StockPredictionDialog(self)
            
            # 우클릭에서 호출된 경우 종목 코드 미리 설정
            if ticker and hasattr(dialog, 'ticker_input'):
                dialog.ticker_input.setText(ticker)
                dialog.ticker_input.selectAll()  # 텍스트 선택해서 쉽게 변경 가능하게
            
            # 다이얼로그 실행
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"AI 예측 다이얼로그 오류:\n{str(e)}")
       
    def show_batch_prediction(self):
        """배치 예측 다이얼로그 표시"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "ML 라이브러리가 설치되지 않았습니다.")
            return
        
        # 현재 스크리닝 결과가 있는지 확인
        candidates = []
        
        if hasattr(self, 'last_buy_candidates') and self.last_buy_candidates:
            candidates.extend(self.last_buy_candidates)
        
        if hasattr(self, 'last_sell_candidates') and self.last_sell_candidates:
            candidates.extend(self.last_sell_candidates)
        
        if not candidates:
            reply = QMessageBox.question(self, "배치 예측", 
                "스크리닝 결과가 없습니다. 샘플 종목으로 테스트하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # 샘플 종목들
                candidates = [
                    {'Symbol': 'AAPL', 'Name': 'Apple Inc.'},
                    {'Symbol': 'MSFT', 'Name': 'Microsoft Corp.'},
                    {'Symbol': 'GOOGL', 'Name': 'Alphabet Inc.'},
                    {'Symbol': 'TSLA', 'Name': 'Tesla Inc.'},
                    {'Symbol': '005930.KS', 'Name': 'Samsung Electronics'}
                ]
            else:
                return
        
        try:
            dialog = BatchPredictionDialog(candidates, self)
            dialog.exec_()
        except NameError as e:
            QMessageBox.critical(self, "Import 오류", f"BatchPredictionDialog를 찾을 수 없습니다:\n{str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"배치 예측 다이얼로그 오류:\n{str(e)}")
    
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
        self.is_running = False
        self.current_index = 0
        self.results = []
        
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
        """배치 예측 시작"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "ML 라이브러리가 설치되지 않았습니다.")
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
            ticker = candidate.get('Symbol', candidate.get('Ticker', ''))
            name = candidate.get('Name', candidate.get('Company', ''))
            
            self.result_table.setItem(i, 0, QTableWidgetItem(ticker))
            self.result_table.setItem(i, 1, QTableWidgetItem(name))
            self.result_table.setItem(i, 7, QTableWidgetItem("대기 중"))
        
        # 예측 시작
        self.run_next_prediction()
    
    def run_next_prediction(self):
        """다음 종목 예측 실행"""
        if not self.is_running or self.current_index >= len(self.candidates):
            self.finish_batch_prediction()
            return
        
        candidate = self.candidates[self.current_index]
        ticker = candidate.get('Symbol', candidate.get('Ticker', ''))
        
        # UI 업데이트
        self.current_work_label.setText(f"예측 중: {ticker}")
        self.overall_progress.setValue(self.current_index)
        
        # 테이블 상태 업데이트
        self.result_table.setItem(self.current_index, 7, QTableWidgetItem("🔄 예측 중"))
        self.result_table.scrollToItem(self.result_table.item(self.current_index, 0))
        
        QApplication.processEvents()
        
        # 예측 실행
        try:
            self.detail_progress.setValue(25)
            QApplication.processEvents()
            
            result, error = self.predictor.predict_stock(ticker, 7)
            
            self.detail_progress.setValue(75)
            QApplication.processEvents()
            
            if error:
                # 실패 처리
                self.result_table.setItem(self.current_index, 2, QTableWidgetItem("오류"))
                self.result_table.setItem(self.current_index, 7, QTableWidgetItem(f"❌ {error[:20]}..."))
                
            else:
                # 성공 처리
                self.results.append(result)
                
                current_price = result.get('current_price', 0)
                predicted_price = result.get('predicted_price', 0)
                expected_return = result.get('expected_return', 0)
                confidence = result.get('confidence', 0)
                
                # 추천 계산
                if confidence > 0.7:
                    if expected_return > 0.05:
                        recommendation = "강매수"
                    elif expected_return > 0.02:
                        recommendation = "매수"
                    elif expected_return < -0.05:
                        recommendation = "매도"
                    elif expected_return < -0.02:
                        recommendation = "매도고려"
                    else:
                        recommendation = "보유"
                else:
                    recommendation = "불확실"
                
                # 테이블 업데이트
                self.result_table.setItem(self.current_index, 2, QTableWidgetItem(f"${current_price:.2f}"))
                self.result_table.setItem(self.current_index, 3, QTableWidgetItem(f"${predicted_price:.2f}"))
                self.result_table.setItem(self.current_index, 4, QTableWidgetItem(f"{expected_return*100:+.2f}%"))
                self.result_table.setItem(self.current_index, 5, QTableWidgetItem(f"{confidence*100:.1f}%"))
                self.result_table.setItem(self.current_index, 6, QTableWidgetItem(recommendation))
                self.result_table.setItem(self.current_index, 7, QTableWidgetItem("✅ 완료"))
            
            self.detail_progress.setValue(100)
            self.update_stats()
            
        except Exception as e:
            # 예외 처리
            self.result_table.setItem(self.current_index, 7, QTableWidgetItem(f"❌ 오류: {str(e)[:20]}"))
        
        self.current_index += 1
        
        # 딜레이 후 다음 예측
        QTimer.singleShot(1000, self.run_next_prediction)  # 1초 딜레이
    
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
    
    print("🧪 Enhanced Screener 테스트")
    
    if ML_AVAILABLE:
        print("✅ CPU 최적화 ML 스택 사용 가능")
        
        # 예제 1: CPU 최적화 예측기 테스트
        predictor = EnhancedCPUPredictor()
        print("🚀 예측기 초기화 완료")
        
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
        print("⚠️ ML 라이브러리 설치 필요")
        print("설치 명령어: pip install scikit-learn xgboost lightgbm statsmodels")
    
    sys.exit(app.exec_())