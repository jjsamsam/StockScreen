"""
ensemble_weight_optimizer.py
앙상블 가중치 최적화 메타모델

핵심 기능:
1. 레짐, 변동성, 모델 성능을 입력으로 받아 최적 가중치 출력
2. Brier Score, Sharpe Ratio 기반 동적 가중치 조정
3. 최적화 방법: Grid Search, Bayesian Optimization, 메타학습
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from logger_config import get_logger
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.metrics import brier_score_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EnsembleWeightOptimizer:
    """
    앙상블 가중치 최적화기

    입력:
    - 레짐 (bull/neutral/bear)
    - 변동성 레벨
    - 최근 모델별 Brier Score
    - 최근 모델별 방향 정확도

    출력:
    - LSTM 가중치, Transformer 가중치
    """

    def __init__(self, method='adaptive'):
        """
        Args:
            method: 'fixed', 'adaptive', 'meta_model'
                - fixed: 고정 가중치 (레짐별)
                - adaptive: 성능 기반 동적 조정
                - meta_model: ML 기반 메타모델
        """
        self.method = method
        self.meta_model = None
        self.performance_history = []

        logger.info(f"EnsembleWeightOptimizer initialized (method: {method})")

    def get_weights(self,
                    regime: str,
                    volatility: float,
                    lstm_brier: float,
                    transformer_brier: float,
                    lstm_accuracy: Optional[float] = None,
                    transformer_accuracy: Optional[float] = None) -> Tuple[float, float]:
        """
        가중치 계산

        Args:
            regime: 'bull', 'neutral', 'bear'
            volatility: 변동성 (0~1)
            lstm_brier: LSTM Brier Score (낮을수록 좋음)
            transformer_brier: Transformer Brier Score
            lstm_accuracy: LSTM 방향 정확도 (옵션)
            transformer_accuracy: Transformer 방향 정확도 (옵션)

        Returns:
            (w_lstm, w_transformer) - 합이 1.0
        """
        if self.method == 'fixed':
            return self._get_fixed_weights(regime, volatility)

        elif self.method == 'adaptive':
            return self._get_adaptive_weights(
                regime, volatility,
                lstm_brier, transformer_brier,
                lstm_accuracy, transformer_accuracy
            )

        elif self.method == 'meta_model':
            if self.meta_model is None:
                logger.warning("Meta model not trained, falling back to adaptive")
                return self._get_adaptive_weights(
                    regime, volatility,
                    lstm_brier, transformer_brier,
                    lstm_accuracy, transformer_accuracy
                )

            return self._get_meta_model_weights(
                regime, volatility,
                lstm_brier, transformer_brier,
                lstm_accuracy, transformer_accuracy
            )

        else:
            logger.warning(f"Unknown method: {self.method}, using fixed")
            return self._get_fixed_weights(regime, volatility)

    def _get_fixed_weights(self, regime: str, volatility: float) -> Tuple[float, float]:
        """
        고정 가중치 (레짐 + 변동성 기반)
        """
        # 기본 가중치 테이블
        base_weights = {
            'bull': {'low_vol': (0.40, 0.60), 'high_vol': (0.55, 0.45)},
            'neutral': {'low_vol': (0.50, 0.50), 'high_vol': (0.60, 0.40)},
            'bear': {'low_vol': (0.65, 0.35), 'high_vol': (0.70, 0.30)}
        }

        vol_level = 'high_vol' if volatility > 0.05 else 'low_vol'
        weights = base_weights.get(regime, {'low_vol': (0.5, 0.5), 'high_vol': (0.5, 0.5)})[vol_level]

        return weights

    def _get_adaptive_weights(self,
                              regime: str,
                              volatility: float,
                              lstm_brier: float,
                              transformer_brier: float,
                              lstm_accuracy: Optional[float] = None,
                              transformer_accuracy: Optional[float] = None) -> Tuple[float, float]:
        """
        적응형 가중치 (성능 기반 조정)

        로직:
        1. 고정 가중치로 시작
        2. Brier Score 기반 보정
        3. 방향 정확도 기반 추가 보정 (있으면)
        """
        # 1. 베이스 가중치
        w_lstm_base, w_transformer_base = self._get_fixed_weights(regime, volatility)

        # 2. Brier Score 기반 조정
        # Brier Score가 낮을수록 좋음 (0에 가까울수록)
        # 역수를 사용하여 점수 계산
        epsilon = 0.01  # 0으로 나누기 방지

        lstm_score = 1.0 / (lstm_brier + epsilon)
        transformer_score = 1.0 / (transformer_brier + epsilon)
        total_score = lstm_score + transformer_score

        # 정규화
        w_lstm_brier = lstm_score / total_score
        w_transformer_brier = transformer_score / total_score

        # 3. 방향 정확도 기반 조정 (있으면)
        if lstm_accuracy is not None and transformer_accuracy is not None:
            total_accuracy = lstm_accuracy + transformer_accuracy + epsilon
            w_lstm_acc = lstm_accuracy / total_accuracy
            w_transformer_acc = transformer_accuracy / total_accuracy
        else:
            w_lstm_acc = 0.5
            w_transformer_acc = 0.5

        # 4. 가중 평균 (베이스 50%, Brier 30%, 정확도 20%)
        w_lstm = 0.5 * w_lstm_base + 0.3 * w_lstm_brier + 0.2 * w_lstm_acc
        w_transformer = 0.5 * w_transformer_base + 0.3 * w_transformer_brier + 0.2 * w_transformer_acc

        # 정규화 (합 1.0)
        total = w_lstm + w_transformer
        w_lstm /= total
        w_transformer /= total

        # 최소/최대 제한 (극단적 가중치 방지)
        w_lstm = np.clip(w_lstm, 0.2, 0.8)
        w_transformer = 1.0 - w_lstm

        logger.debug(f"Adaptive weights: LSTM={w_lstm:.3f}, Transformer={w_transformer:.3f}")

        return w_lstm, w_transformer

    def _get_meta_model_weights(self,
                                 regime: str,
                                 volatility: float,
                                 lstm_brier: float,
                                 transformer_brier: float,
                                 lstm_accuracy: Optional[float] = None,
                                 transformer_accuracy: Optional[float] = None) -> Tuple[float, float]:
        """
        메타모델 기반 가중치 예측
        """
        # 피처 생성
        features = self._create_features(
            regime, volatility,
            lstm_brier, transformer_brier,
            lstm_accuracy, transformer_accuracy
        )

        try:
            # 예측
            weights = self.meta_model.predict(features.reshape(1, -1))[0]

            # LSTM 가중치만 예측 (Transformer는 1-LSTM)
            w_lstm = float(np.clip(weights, 0.2, 0.8))
            w_transformer = 1.0 - w_lstm

            logger.debug(f"Meta model weights: LSTM={w_lstm:.3f}, Transformer={w_transformer:.3f}")

            return w_lstm, w_transformer

        except Exception as e:
            logger.error(f"Meta model prediction failed: {e}")
            return self._get_adaptive_weights(
                regime, volatility,
                lstm_brier, transformer_brier,
                lstm_accuracy, transformer_accuracy
            )

    def _create_features(self,
                         regime: str,
                         volatility: float,
                         lstm_brier: float,
                         transformer_brier: float,
                         lstm_accuracy: Optional[float] = None,
                         transformer_accuracy: Optional[float] = None) -> np.ndarray:
        """
        메타모델용 피처 생성
        """
        # 레짐 인코딩
        regime_map = {'bear': 0, 'neutral': 1, 'bull': 2}
        regime_encoded = regime_map.get(regime, 1)

        # 피처 배열
        features = [
            regime_encoded,
            volatility,
            lstm_brier,
            transformer_brier,
            lstm_brier - transformer_brier,  # 성능 차이
            lstm_accuracy if lstm_accuracy is not None else 0.5,
            transformer_accuracy if transformer_accuracy is not None else 0.5
        ]

        return np.array(features)

    def train_meta_model(self,
                         training_data: pd.DataFrame,
                         regime_col: str = 'regime',
                         volatility_col: str = 'volatility',
                         lstm_brier_col: str = 'lstm_brier',
                         transformer_brier_col: str = 'transformer_brier',
                         optimal_weight_col: str = 'optimal_lstm_weight'):
        """
        메타모델 학습

        Args:
            training_data: 학습 데이터
                - regime, volatility, lstm_brier, transformer_brier, optimal_lstm_weight
            regime_col: 레짐 컬럼명
            volatility_col: 변동성 컬럼명
            lstm_brier_col: LSTM Brier 컬럼명
            transformer_brier_col: Transformer Brier 컬럼명
            optimal_weight_col: 최적 LSTM 가중치 컬럼명
        """
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available, cannot train meta model")
            return

        # 피처 생성
        X_list = []
        for _, row in training_data.iterrows():
            features = self._create_features(
                row[regime_col],
                row[volatility_col],
                row[lstm_brier_col],
                row[transformer_brier_col],
                row.get('lstm_accuracy', None),
                row.get('transformer_accuracy', None)
            )
            X_list.append(features)

        X = np.array(X_list)
        y = training_data[optimal_weight_col].values

        # 학습/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        try:
            # LightGBM 회귀 모델
            self.meta_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1
            )

            self.meta_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                # early_stopping_rounds=10,  # Removed deprecated parameter
                # verbose=False  # Removed deprecated parameter
            )

            # 검증 성능
            val_pred = self.meta_model.predict(X_val)
            mae = np.mean(np.abs(val_pred - y_val))

            logger.info(f"Meta model trained: MAE={mae:.4f}")

            self.method = 'meta_model'  # 메타모델 사용으로 전환

        except Exception as e:
            logger.error(f"Meta model training failed: {e}")

    def optimize_weights_grid_search(self,
                                     predictions_lstm: np.ndarray,
                                     predictions_transformer: np.ndarray,
                                     actual: np.ndarray,
                                     metric: str = 'mse') -> float:
        """
        그리드 서치로 최적 가중치 찾기

        Args:
            predictions_lstm: LSTM 예측값
            predictions_transformer: Transformer 예측값
            actual: 실제값
            metric: 'mse' 또는 'sharpe'

        Returns:
            최적 LSTM 가중치
        """
        best_weight = 0.5
        best_score = float('inf') if metric == 'mse' else float('-inf')

        # 0.1 단위로 그리드 서치
        for w_lstm in np.arange(0.0, 1.01, 0.1):
            w_transformer = 1.0 - w_lstm

            # 앙상블 예측
            ensemble_pred = w_lstm * predictions_lstm + w_transformer * predictions_transformer

            # 평가
            if metric == 'mse':
                score = np.mean((ensemble_pred - actual) ** 2)
                if score < best_score:
                    best_score = score
                    best_weight = w_lstm

            elif metric == 'sharpe':
                # 수익률 계산
                returns = np.diff(ensemble_pred) / ensemble_pred[:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)

                if sharpe > best_score:
                    best_score = sharpe
                    best_weight = w_lstm

        logger.debug(f"Grid search optimal weight: {best_weight:.2f} (score: {best_score:.4f})")

        return best_weight

    def optimize_weights_scipy(self,
                               predictions_lstm: np.ndarray,
                               predictions_transformer: np.ndarray,
                               actual: np.ndarray) -> float:
        """
        Scipy 최적화로 최적 가중치 찾기

        Returns:
            최적 LSTM 가중치
        """
        def objective(w_lstm):
            w_transformer = 1.0 - w_lstm
            ensemble_pred = w_lstm * predictions_lstm + w_transformer * predictions_transformer
            mse = np.mean((ensemble_pred - actual) ** 2)
            return mse

        # 최적화
        result = minimize(
            objective,
            x0=0.5,
            bounds=[(0.2, 0.8)],  # 극단적 가중치 방지
            method='L-BFGS-B'
        )

        optimal_weight = float(result.x[0])

        logger.debug(f"Scipy optimal weight: {optimal_weight:.3f} (MSE: {result.fun:.4f})")

        return optimal_weight

    def update_performance_history(self,
                                    regime: str,
                                    volatility: float,
                                    w_lstm: float,
                                    w_transformer: float,
                                    ensemble_error: float):
        """
        성능 히스토리 업데이트 (메타모델 재학습용)

        Args:
            regime: 레짐
            volatility: 변동성
            w_lstm: 사용한 LSTM 가중치
            w_transformer: 사용한 Transformer 가중치
            ensemble_error: 앙상블 오류 (MAE 또는 MSE)
        """
        self.performance_history.append({
            'regime': regime,
            'volatility': volatility,
            'w_lstm': w_lstm,
            'w_transformer': w_transformer,
            'error': ensemble_error
        })

        # 최대 1000개만 유지
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]


class BrierScoreCalculator:
    """
    Brier Score 계산기 (확률 예측 평가)
    """

    @staticmethod
    def calculate_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Brier Score 계산

        Args:
            probabilities: 예측 확률 (0~1)
            outcomes: 실제 결과 (0 또는 1)

        Returns:
            Brier Score (낮을수록 좋음, 0~1)
        """
        if not SKLEARN_AVAILABLE:
            # 수동 계산
            return float(np.mean((probabilities - outcomes) ** 2))

        try:
            return brier_score_loss(outcomes, probabilities)
        except Exception as e:
            logger.error(f"Brier score calculation failed: {e}")
            return 0.5  # 기본값

    @staticmethod
    def calculate_rolling_brier(probabilities: np.ndarray,
                                outcomes: np.ndarray,
                                window: int = 20) -> np.ndarray:
        """
        롤링 Brier Score 계산

        Args:
            probabilities: 예측 확률 시계열
            outcomes: 실제 결과 시계열
            window: 윈도우 크기

        Returns:
            롤링 Brier Score 배열
        """
        rolling_scores = []

        for i in range(window, len(probabilities) + 1):
            window_probs = probabilities[i - window:i]
            window_outcomes = outcomes[i - window:i]

            score = BrierScoreCalculator.calculate_brier_score(window_probs, window_outcomes)
            rolling_scores.append(score)

        return np.array(rolling_scores)


# === 사용 예시 ===
if __name__ == '__main__':
    # 1. 적응형 가중치
    optimizer = EnsembleWeightOptimizer(method='adaptive')

    w_lstm, w_transformer = optimizer.get_weights(
        regime='bull',
        volatility=0.03,
        lstm_brier=0.15,
        transformer_brier=0.12,
        lstm_accuracy=0.58,
        transformer_accuracy=0.62
    )

    print(f"Adaptive weights: LSTM={w_lstm:.3f}, Transformer={w_transformer:.3f}")

    # 2. 그리드 서치 최적화
    predictions_lstm = np.random.randn(100).cumsum()
    predictions_transformer = np.random.randn(100).cumsum()
    actual = np.random.randn(100).cumsum()

    optimal_w = optimizer.optimize_weights_grid_search(
        predictions_lstm, predictions_transformer, actual, metric='mse'
    )

    print(f"Grid search optimal weight: {optimal_w:.3f}")

    # 3. Brier Score 계산
    probabilities = np.random.rand(100)
    outcomes = (np.random.rand(100) > 0.5).astype(int)

    brier = BrierScoreCalculator.calculate_brier_score(probabilities, outcomes)
    print(f"Brier Score: {brier:.4f}")
