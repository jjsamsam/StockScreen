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

# 최적화 모듈
from cache_manager import get_stock_data

# 필요한 라이브러리들
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn 사용 가능")
except ImportError:
    print("❌ scikit-learn이 설치되지 않았습니다: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(f"✅ XGBoost 사용 가능 (v{xgb.__version__})")
except ImportError:
    print("❌ XGBoost가 설치되지 않았습니다: pip install xgboost")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print(f"✅ LightGBM 사용 가능 (v{lgb.__version__})")
except ImportError:
    print("❌ LightGBM이 설치되지 않았습니다: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
    print("✅ statsmodels 사용 가능")
except ImportError:
    print("❌ statsmodels가 설치되지 않았습니다: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

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
        
        # 미래 예측 (단순 추세 외삽)
        if len(filtered_prices) >= 2:
            trend = filtered_prices[-1] - filtered_prices[-2]
        else:
            trend = 0
        
        future_predictions = []
        current_state = self.x
        
        for i in range(forecast_days):
            future_predictions.append(current_state + trend * (i + 1))
        
        # 신뢰구간 계산
        confidence_interval = 2 * np.sqrt(self.P)
        
        return {
            'filtered_prices': np.array(filtered_prices),
            'future_predictions': np.array(future_predictions),
            'confidence_interval': confidence_interval,
            'last_state': self.x,
            'uncertainty': self.P
        }

class AdvancedMLPredictor:
    """
    XGBoost, LightGBM, Random Forest를 사용한 고급 머신러닝 예측기
    """
    
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.models = {}
        
    def create_features(self, data):
        """기술적 지표를 포함한 피처 생성"""
        df = pd.DataFrame()
        
        # 기본 가격 정보
        df['close'] = data
        df['high'] = data  # 간단화를 위해 close와 동일하게 처리
        df['low'] = data
        df['volume'] = 1000000  # 더미 거래량
        
        # 이동평균
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
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
        
        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # 변화율
        df['pct_change_1'] = df['close'].pct_change()
        df['pct_change_5'] = df['close'].pct_change(5)
        df['pct_change_10'] = df['close'].pct_change(10)
        
        # 변동성
        df['volatility'] = df['close'].rolling(20).std()
        
        return df
    
    def prepare_data(self, prices):
        """ML 모델용 데이터 전처리"""
        # 피처 생성
        df = self.create_features(prices)
        
        # NaN 제거
        df = df.dropna()
        
        if len(df) < self.sequence_length + 1:
            raise ValueError(f"데이터가 부족합니다. 최소 {self.sequence_length + 1}개 필요")
        
        # 피처 선택
        feature_columns = [
            'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'pct_change_1', 'pct_change_5', 
            'pct_change_10', 'volatility'
        ]
        
        # 시퀀스 데이터 생성
        X, y = [], []
        for i in range(self.sequence_length, len(df)):
            # 과거 sequence_length개의 피처들
            sequence_features = []
            for j in range(i - self.sequence_length, i):
                row_features = df[feature_columns].iloc[j].values
                sequence_features.extend(row_features)
            
            X.append(sequence_features)
            y.append(df['close'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """여러 ML 모델 훈련"""
        
        # 1. Random Forest
        if SKLEARN_AVAILABLE:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)
            self.models['random_forest'] = rf_model
            
            rf_pred = rf_model.predict(X_val)
            rf_score = np.sqrt(mean_squared_error(y_val, rf_pred))
            print(f"Random Forest RMSE: {rf_score:.2f}")
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            
            xgb_pred = xgb_model.predict(X_val)
            xgb_score = np.sqrt(mean_squared_error(y_val, xgb_pred))
            print(f"XGBoost RMSE: {xgb_score:.2f}")
        
        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
            lgb_model.fit(X_train, y_train)
            self.models['lightgbm'] = lgb_model
            
            lgb_pred = lgb_model.predict(X_val)
            lgb_score = np.sqrt(mean_squared_error(y_val, lgb_pred))
            print(f"LightGBM RMSE: {lgb_score:.2f}")
    
    def fit_predict(self, prices, forecast_days=5):
        """ML 모델 훈련 및 예측"""
        if len(prices) < self.sequence_length + 20:
            raise ValueError(f"최소 {self.sequence_length + 20}개의 데이터 포인트가 필요합니다")
        
        # 데이터 준비
        X, y = self.prepare_data(prices)
        
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
        
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(last_sequence)[0]
            predictions[model_name] = pred
        
        # 앙상블 (평균)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # 간단한 미래 예측 (추세 적용)
        if len(prices) >= 2:
            trend = prices[-1] - prices[-2]
        else:
            trend = 0
        
        future_predictions = []
        for i in range(forecast_days):
            future_pred = ensemble_pred + trend * i
            future_predictions.append(future_pred)
        
        return {
            'future_predictions': np.array(future_predictions),
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'model_performance': {
                'models_used': list(self.models.keys()),
                'validation_score': 'calculated_above'
            }
        }

class ARIMAPredictor:
    """ARIMA 모델을 사용한 주가 예측"""
    
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
    
    def fit_predict(self, prices, forecast_days=5):
        """ARIMA 모델 피팅 및 예측"""
        if not STATSMODELS_AVAILABLE:
            # 간단한 이동평균 대체
            ma_window = min(20, len(prices) // 2)
            if ma_window < 2:
                ma_window = 2
            trend = np.mean(np.diff(prices[-ma_window:]))
            last_price = prices[-1]
            future_predictions = [last_price + trend * (i+1) for i in range(forecast_days)]
            
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
            print(f"ARIMA 모델 오류: {e}")
            # ARIMA 실패시 단순 추세 외삽
            trend = np.mean(np.diff(prices[-20:]))
            last_price = prices[-1]
            future_predictions = [last_price + trend * (i+1) for i in range(forecast_days)]
            
            return {
                'future_predictions': np.array(future_predictions),
                'method': 'linear_trend_fallback',
                'error': str(e)
            }

class EnsemblePredictor:
    """여러 모델을 결합한 앙상블 예측기 (TensorFlow 제외)"""
    
    def __init__(self):
        self.kalman = KalmanFilterPredictor()
        self.ml_predictor = AdvancedMLPredictor() if (SKLEARN_AVAILABLE or XGBOOST_AVAILABLE or LIGHTGBM_AVAILABLE) else None
        self.arima = ARIMAPredictor()
        
        # 모델별 가중치
        self.weights = {
            'kalman': 0.3,
            'ml_models': 0.4 if self.ml_predictor else 0,
            'arima': 0.3 if self.ml_predictor else 0.7
        }
    
    def fit_predict(self, prices, forecast_days=5):
        """앙상블 예측 실행"""
        results = {}
        predictions = []
        
        print("🔄 앙상블 예측 시작...")
        
        # 1. Kalman Filter 예측
        print("📊 Kalman Filter 실행 중...")
        kalman_result = self.kalman.fit_predict(prices, forecast_days)
        results['kalman'] = kalman_result
        predictions.append(kalman_result['future_predictions'])
        
        # 2. ML 모델 예측 (XGBoost, LightGBM, Random Forest)
        if self.ml_predictor and len(prices) >= 50:
            print("🤖 ML 모델들 훈련 중...")
            try:
                ml_result = self.ml_predictor.fit_predict(prices, forecast_days)
                results['ml_models'] = ml_result
                predictions.append(ml_result['future_predictions'])
            except Exception as e:
                print(f"⚠️ ML 모델 실패: {e}")
                self.weights['ml_models'] = 0
                self.weights['kalman'] += 0.2
                self.weights['arima'] += 0.2
        
        # 3. ARIMA 예측
        print("📈 ARIMA 모델 피팅 중...")
        arima_result = self.arima.fit_predict(prices, forecast_days)
        results['arima'] = arima_result
        predictions.append(arima_result['future_predictions'])
        
        # 4. 앙상블 결합
        if len(predictions) > 1:
            # 가중평균 계산
            weighted_predictions = np.zeros(forecast_days)
            total_weight = 0
            
            active_models = ['kalman', 'ml_models', 'arima'][:len(predictions)]
            
            for i, model_name in enumerate(active_models):
                if self.weights[model_name] > 0:
                    weighted_predictions += predictions[i] * self.weights[model_name]
                    total_weight += self.weights[model_name]
            
            ensemble_predictions = weighted_predictions / total_weight if total_weight > 0 else predictions[0]
        else:
            ensemble_predictions = predictions[0]
        
        # 신뢰도 계산 (예측값들의 분산)
        if len(predictions) > 1:
            prediction_std = np.std(predictions, axis=0)
            confidence_score = 1 / (1 + np.mean(prediction_std) / np.mean(ensemble_predictions))
        else:
            confidence_score = 0.5
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'individual_results': results,
            'confidence_score': confidence_score,
            'model_weights': self.weights,
            'prediction_variance': np.var(predictions, axis=0) if len(predictions) > 1 else np.zeros(forecast_days)
        }

class StockPredictor:
    """통합 주가 예측 시스템 (TensorFlow 없음)"""
    
    def __init__(self):
        self.ensemble = EnsemblePredictor()
    
    def get_stock_data(self, symbol, period="1y"):
        """주식 데이터 가져오기 - 캐싱 사용"""
        try:
            data = get_stock_data(symbol, period=period)
            return data
        except Exception as e:
            print(f"❌ 데이터 가져오기 실패: {e}")
            return None
    
    def predict_stock_price(self, symbol, forecast_days=5, show_plot=True):
        """종목의 미래 주가 예측"""
        print(f"🎯 {symbol} 주가 예측 시작...")
        
        # 1. 데이터 수집
        data = self.get_stock_data(symbol)
        if data is None or len(data) < 50:
            return {"error": "충분한 데이터가 없습니다"}
        
        prices = data['Close'].values
        dates = data.index
        
        print(f"📅 분석 기간: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
        print(f"📊 데이터 포인트: {len(prices)}개")
        
        # 2. 앙상블 예측 실행
        result = self.ensemble.fit_predict(prices, forecast_days)
        
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
        
        # 4. 결과 반환
        prediction_result = {
            'symbol': symbol,
            'current_price': last_price,
            'predicted_prices': predicted_prices,
            'future_dates': future_dates,
            'expected_returns': returns,
            'confidence_score': confidence,
            'model_weights': result['model_weights'],
            'recommendation': self._generate_recommendation(returns, confidence),
            'models_used': self._get_models_used()
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

# 사용 예제
def example_usage():
    """사용 예제"""
    predictor = StockPredictor()
    
    print("사용 가능한 라이브러리:")
    print(f"- scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"- XGBoost: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"- LightGBM: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    print(f"- statsmodels: {'✅' if STATSMODELS_AVAILABLE else '❌'}")
    
    # 예제 1: 애플 주식 예측
    print("=" * 50)
    print("📈 APPLE (AAPL) 주가 예측")
    print("=" * 50)
    
    result = predictor.predict_stock_price('AAPL', forecast_days=7, show_plot=False)
    
    if 'error' not in result:
        print(f"현재가: ${result['current_price']:.2f}")
        print(f"예측 가격: {[f'${p:.2f}' for p in result['predicted_prices']]}")
        print(f"예상 수익률: {[f'{r:.1f}%' for r in result['expected_returns']]}")
        print(f"신뢰도: {result['confidence_score']:.1%}")
        print(f"추천: {result['recommendation']}")
        print(f"사용 모델: {', '.join(result['models_used'])}")

if __name__ == "__main__":
    example_usage()