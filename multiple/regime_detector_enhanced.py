"""
regime_detector_enhanced.py
강화된 레짐 감지기 - 다양한 피처와 ML 기반 시장 상태 분류

핵심 개선사항:
1. 추가 피처: 브레드스 지수, 섹터 로테이션, 모멘텀, 유동성
2. ML 기반 레짐 분류 (LightGBM/XGBoost)
3. 확률 기반 출력 (Bull/Neutral/Bear 확률)
4. 레짐 전환 감지 및 점진적 가중치 조정
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from logger_config import get_logger

logger = get_logger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from cache_manager import get_stock_data
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache manager not available")


class EnhancedRegimeDetector:
    """
    강화된 레짐 감지기

    Features:
    - 기본: 추세, 변동성, 상승/하락 일수
    - 추가: 브레드스, 섹터 로테이션, 모멘텀, 유동성
    - 외부: VIX, S&P500, 국채 수익률, 수익률 곡선
    """

    def __init__(self, use_ml=True, model_type='lightgbm'):
        """
        Args:
            use_ml: True면 ML 기반, False면 규칙 기반
            model_type: 'lightgbm' 또는 'xgboost'
        """
        self.use_ml = use_ml and (LIGHTGBM_AVAILABLE or XGBOOST_AVAILABLE)
        self.model_type = model_type
        self.model = None
        self.feature_names = []

        # 레짐 히스토리 (전환 감지용)
        self.regime_history = []
        self.regime_transition_date = None

        logger.info(f"EnhancedRegimeDetector initialized (ML: {self.use_ml}, Type: {model_type})")

    def extract_features(self,
                         prices: np.ndarray,
                         volumes: Optional[np.ndarray] = None,
                         market_data: Optional[Dict] = None,
                         window: int = 50) -> Dict[str, float]:
        """
        레짐 감지용 피처 추출

        Args:
            prices: 가격 배열
            volumes: 거래량 배열 (옵션)
            market_data: 시장 데이터 딕셔너리 (VIX, S&P500 등)
            window: 윈도우 크기

        Returns:
            피처 딕셔너리
        """
        features = {}

        if len(prices) < window:
            logger.warning(f"Insufficient data: {len(prices)} < {window}")
            return self._get_default_features()

        recent_prices = prices[-window:]

        # === 1. 기본 피처 ===

        # 1.1 추세 (선형 회귀 기울기)
        trend_coef = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        avg_price = np.mean(recent_prices)
        features['trend_pct'] = (trend_coef / avg_price) * 100 if avg_price > 0 else 0

        # 1.2 변동성 (CV)
        features['volatility'] = np.std(recent_prices) / avg_price if avg_price > 0 else 0

        # 1.3 상승/하락 일수 비율
        price_changes = np.diff(recent_prices)
        features['up_days_ratio'] = np.sum(price_changes > 0) / len(price_changes) if len(price_changes) > 0 else 0.5

        # 1.4 최근 수익률 (1주, 1개월, 3개월)
        if len(prices) >= 5:
            features['return_1w'] = (prices[-1] - prices[-5]) / prices[-5] * 100 if prices[-5] != 0 else 0
        if len(prices) >= 20:
            features['return_1m'] = (prices[-1] - prices[-20]) / prices[-20] * 100 if prices[-20] != 0 else 0
        if len(prices) >= 60:
            features['return_3m'] = (prices[-1] - prices[-60]) / prices[-60] * 100 if prices[-60] != 0 else 0

        # === 2. 추가 피처 ===

        # 2.1 모멘텀 지표
        features['momentum_10'] = self._calculate_momentum(recent_prices, 10)
        features['momentum_20'] = self._calculate_momentum(recent_prices, 20)

        # 2.2 변동성 비율 (최근 vs 과거)
        if len(prices) >= window * 2:
            past_vol = np.std(prices[-window*2:-window]) / np.mean(prices[-window*2:-window])
            current_vol = features['volatility']
            features['vol_ratio'] = current_vol / past_vol if past_vol > 0 else 1.0
        else:
            features['vol_ratio'] = 1.0

        # 2.3 드로다운 (최대 낙폭)
        features['drawdown'] = self._calculate_max_drawdown(recent_prices)

        # 2.4 고점/저점 갱신 빈도
        features['new_high_ratio'] = self._calculate_new_high_ratio(recent_prices, 20)
        features['new_low_ratio'] = self._calculate_new_low_ratio(recent_prices, 20)

        # 2.5 가격 대비 이동평균 위치
        if len(recent_prices) >= 20:
            ma20 = np.mean(recent_prices[-20:])
            features['price_vs_ma20'] = (recent_prices[-1] - ma20) / ma20 * 100 if ma20 > 0 else 0

        if len(recent_prices) >= 50:
            ma50 = np.mean(recent_prices[-50:])
            features['price_vs_ma50'] = (recent_prices[-1] - ma50) / ma50 * 100 if ma50 > 0 else 0

        # === 3. 거래량 피처 (있는 경우) ===

        if volumes is not None and len(volumes) >= window:
            recent_volumes = volumes[-window:]

            # 3.1 거래량 추세
            vol_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
            avg_volume = np.mean(recent_volumes)
            features['volume_trend'] = (vol_trend / avg_volume) * 100 if avg_volume > 0 else 0

            # 3.2 거래량 변동성
            features['volume_volatility'] = np.std(recent_volumes) / avg_volume if avg_volume > 0 else 0

            # 3.3 거래량 급증 빈도
            if len(recent_volumes) >= 20:
                vol_ma20 = np.mean(recent_volumes[-20:])
                volume_spikes = np.sum(recent_volumes[-20:] > vol_ma20 * 1.5) / 20
                features['volume_spike_ratio'] = volume_spikes

        # === 4. 외부 시장 지표 ===

        if market_data:
            # 4.1 VIX (변동성 지수)
            if 'vix' in market_data and market_data['vix'] is not None:
                features['vix_level'] = market_data['vix']
                features['vix_vs_avg'] = market_data.get('vix_vs_avg', 0)

            # 4.2 S&P 500 추세
            if 'sp500_return' in market_data:
                features['sp500_return_3m'] = market_data['sp500_return']

            # 4.3 국채 수익률
            if 'treasury_10y' in market_data:
                features['treasury_10y'] = market_data['treasury_10y']
                features['treasury_yield_change'] = market_data.get('treasury_yield_change', 0)

            # 4.4 수익률 곡선 (10Y-2Y)
            if 'yield_spread' in market_data:
                features['yield_spread'] = market_data['yield_spread']

            # 4.5 브레드스 지수 (상승 종목 비율) - 시장 강도
            if 'breadth_ratio' in market_data:
                features['breadth_ratio'] = market_data['breadth_ratio']

        return features

    def _calculate_momentum(self, prices: np.ndarray, period: int) -> float:
        """모멘텀 계산 (Rate of Change)"""
        if len(prices) <= period:
            return 0.0
        return (prices[-1] - prices[-period]) / prices[-period] * 100 if prices[-period] != 0 else 0.0

    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """최대 낙폭 계산 (%)"""
        if len(prices) == 0:
            return 0.0

        cummax = np.maximum.accumulate(prices)
        drawdown = (prices - cummax) / cummax * 100
        return float(np.min(drawdown))

    def _calculate_new_high_ratio(self, prices: np.ndarray, window: int) -> float:
        """고점 갱신 비율"""
        if len(prices) < window:
            return 0.0

        highs = [prices[i] >= np.max(prices[max(0, i-window):i+1]) for i in range(len(prices))]
        return np.sum(highs[-window:]) / window

    def _calculate_new_low_ratio(self, prices: np.ndarray, window: int) -> float:
        """저점 갱신 비율"""
        if len(prices) < window:
            return 0.0

        lows = [prices[i] <= np.min(prices[max(0, i-window):i+1]) for i in range(len(prices))]
        return np.sum(lows[-window:]) / window

    def _get_default_features(self) -> Dict[str, float]:
        """기본 피처 (데이터 부족 시)"""
        return {
            'trend_pct': 0.0,
            'volatility': 0.05,
            'up_days_ratio': 0.5,
            'return_1w': 0.0,
            'return_1m': 0.0,
            'return_3m': 0.0,
            'momentum_10': 0.0,
            'momentum_20': 0.0,
            'vol_ratio': 1.0,
            'drawdown': 0.0,
            'new_high_ratio': 0.0,
            'new_low_ratio': 0.0,
        }

    def detect_regime_rule_based(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        규칙 기반 레짐 감지

        Returns:
            (regime, probabilities)
            regime: 'bull', 'neutral', 'bear'
            probabilities: {'bull': 0.7, 'neutral': 0.2, 'bear': 0.1}
        """
        score = 0.0

        # 추세 점수
        if features.get('trend_pct', 0) > 0.5:
            score += 1.5
        elif features.get('trend_pct', 0) < -0.5:
            score -= 1.5

        # 상승일 비율
        if features.get('up_days_ratio', 0.5) > 0.55:
            score += 1.0
        elif features.get('up_days_ratio', 0.5) < 0.45:
            score -= 1.0

        # 모멘텀
        momentum = features.get('momentum_20', 0)
        if momentum > 5:
            score += 1.0
        elif momentum < -5:
            score -= 1.0

        # 변동성 (고변동성은 불확실성)
        volatility = features.get('volatility', 0.05)
        if volatility > 0.05:
            score -= 0.5  # 변동성 높으면 약간 부정적

        # VIX (있으면)
        if 'vix_vs_avg' in features:
            vix_vs_avg = features['vix_vs_avg']
            if vix_vs_avg < -0.1:  # VIX 낮음
                score += 0.5
            elif vix_vs_avg > 0.2:  # VIX 높음
                score -= 0.5

        # S&P 500
        if 'sp500_return_3m' in features:
            sp500_ret = features['sp500_return_3m']
            if sp500_ret > 5:
                score += 0.5
            elif sp500_ret < -5:
                score -= 0.5

        # 수익률 곡선
        if 'yield_spread' in features:
            spread = features['yield_spread']
            if spread < -0.1:  # 역전
                score -= 1.0
            elif spread > 1.5:  # 가파름
                score += 0.5

        # 브레드스 (시장 강도)
        if 'breadth_ratio' in features:
            breadth = features['breadth_ratio']
            if breadth > 0.6:  # 60% 이상 상승
                score += 0.5
            elif breadth < 0.4:  # 40% 미만 상승
                score -= 0.5

        # 점수를 확률로 변환 (softmax 스타일)
        if score >= 1.5:
            regime = 'bull'
            probs = {'bull': 0.7, 'neutral': 0.2, 'bear': 0.1}
        elif score <= -1.5:
            regime = 'bear'
            probs = {'bull': 0.1, 'neutral': 0.2, 'bear': 0.7}
        else:
            regime = 'neutral'
            probs = {'bull': 0.25, 'neutral': 0.5, 'bear': 0.25}

        # 변동성이 매우 높으면 neutral 확률 상승
        if volatility > 0.08:
            probs['neutral'] = min(0.6, probs['neutral'] + 0.2)
            probs['bull'] *= 0.8
            probs['bear'] *= 0.8

        return regime, probs

    def detect_regime(self,
                      prices: np.ndarray,
                      volumes: Optional[np.ndarray] = None,
                      market_data: Optional[Dict] = None,
                      window: int = 50) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        레짐 감지 (통합 인터페이스)

        Returns:
            (regime, probabilities, features)
        """
        # 피처 추출
        features = self.extract_features(prices, volumes, market_data, window)

        # ML 또는 규칙 기반
        if self.use_ml and self.model is not None:
            regime, probs = self.detect_regime_ml(features)
        else:
            regime, probs = self.detect_regime_rule_based(features)

        # 레짐 히스토리 업데이트
        self._update_regime_history(regime)

        return regime, probs, features

    def detect_regime_ml(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        ML 기반 레짐 감지
        (학습된 모델이 있어야 함)
        """
        if self.model is None:
            logger.warning("Model not trained, falling back to rule-based")
            return self.detect_regime_rule_based(features)

        # 피처 배열 생성
        X = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)

        try:
            # 확률 예측
            proba = self.model.predict_proba(X)[0]

            # 클래스: [bear, neutral, bull]
            probs = {
                'bear': float(proba[0]),
                'neutral': float(proba[1]),
                'bull': float(proba[2])
            }

            # 최대 확률 레짐
            regime = max(probs, key=probs.get)

            return regime, probs

        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return self.detect_regime_rule_based(features)

    def _update_regime_history(self, regime: str):
        """레짐 히스토리 업데이트 (전환 감지)"""
        if len(self.regime_history) == 0 or self.regime_history[-1] != regime:
            self.regime_transition_date = datetime.now()
            logger.info(f"Regime transition detected: {self.regime_history[-1] if self.regime_history else 'None'} -> {regime}")

        self.regime_history.append(regime)

        # 최대 100개만 유지
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]

    def get_days_since_transition(self) -> int:
        """레짐 전환 후 경과 일수"""
        if self.regime_transition_date is None:
            return 999  # 충분히 큰 값

        days = (datetime.now() - self.regime_transition_date).days
        return days

    def get_ensemble_weights_for_regime(self,
                                        regime: str,
                                        regime_probs: Dict[str, float],
                                        volatility: float,
                                        days_since_transition: Optional[int] = None) -> Dict[str, float]:
        """
        레짐별 앙상블 가중치 반환

        Args:
            regime: 현재 레짐
            regime_probs: 레짐 확률
            volatility: 변동성
            days_since_transition: 전환 후 경과 일수

        Returns:
            {'lstm': 0.4, 'transformer': 0.6}
        """
        # 기본 가중치 (레짐별)
        base_weights = {
            'bull': {'lstm': 0.40, 'transformer': 0.60},
            'neutral': {'lstm': 0.50, 'transformer': 0.50},
            'bear': {'lstm': 0.65, 'transformer': 0.35}
        }

        # 변동성 조정
        if volatility > 0.08:  # 고변동성
            # LSTM이 단기 패턴을 더 잘 포착
            base_weights[regime]['lstm'] = min(0.7, base_weights[regime]['lstm'] + 0.15)
            base_weights[regime]['transformer'] = 1.0 - base_weights[regime]['lstm']

        weights = base_weights.get(regime, {'lstm': 0.5, 'transformer': 0.5})

        # 레짐 확률이 불확실하면 균등 가중치로
        max_prob = max(regime_probs.values())
        if max_prob < 0.5:  # 불확실
            weights = {'lstm': 0.5, 'transformer': 0.5}

        # 전환 직후면 점진적 변경 (3일간 선형 보간)
        if days_since_transition is not None and days_since_transition < 3:
            if len(self.regime_history) >= 2:
                prev_regime = self.regime_history[-2]
                prev_weights = base_weights.get(prev_regime, {'lstm': 0.5, 'transformer': 0.5})

                alpha = days_since_transition / 3.0  # 0 -> 1
                weights['lstm'] = prev_weights['lstm'] * (1 - alpha) + weights['lstm'] * alpha
                weights['transformer'] = prev_weights['transformer'] * (1 - alpha) + weights['transformer'] * alpha

        return weights

    def train_ml_model(self,
                       training_data: pd.DataFrame,
                       target_column: str = 'regime',
                       feature_columns: Optional[List[str]] = None):
        """
        ML 모델 학습

        Args:
            training_data: 학습 데이터 (피처 + 레짐 라벨)
            target_column: 타겟 컬럼명
            feature_columns: 피처 컬럼 리스트 (None이면 자동)
        """
        if not self.use_ml:
            logger.warning("ML not enabled")
            return

        if feature_columns is None:
            feature_columns = [col for col in training_data.columns if col != target_column]

        X = training_data[feature_columns].values
        y = training_data[target_column].values

        # 레짐 라벨 인코딩: bear=0, neutral=1, bull=2
        label_map = {'bear': 0, 'neutral': 1, 'bull': 2}
        y_encoded = np.array([label_map.get(regime, 1) for regime in y])

        self.feature_names = feature_columns

        try:
            if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=-1
                )
                self.model.fit(X, y_encoded)
                logger.info(f"LightGBM model trained on {len(X)} samples")

            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    verbosity=0
                )
                self.model.fit(X, y_encoded)
                logger.info(f"XGBoost model trained on {len(X)} samples")

            else:
                logger.warning("No ML library available")
                self.use_ml = False

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            self.use_ml = False


def fetch_market_data() -> Dict:
    """
    외부 시장 데이터 가져오기

    Returns:
        시장 데이터 딕셔너리
    """
    market_data = {}

    if not CACHE_AVAILABLE:
        logger.warning("Cache not available, skipping market data fetch")
        return market_data

    try:
        # VIX
        vix_data = get_stock_data('^VIX', period='3mo')
        if vix_data is not None and len(vix_data) > 0:
            market_data['vix'] = float(vix_data['Close'].iloc[-1])
            avg_vix = float(vix_data['Close'].mean())
            market_data['vix_vs_avg'] = (market_data['vix'] - avg_vix) / avg_vix

        # S&P 500
        sp500_data = get_stock_data('^GSPC', period='3mo')
        if sp500_data is not None and len(sp500_data) > 0:
            sp500_return = (sp500_data['Close'].iloc[-1] - sp500_data['Close'].iloc[0]) / sp500_data['Close'].iloc[0] * 100
            market_data['sp500_return'] = float(sp500_return)

        # 10년 국채
        treasury_10y = get_stock_data('^TNX', period='3mo')
        if treasury_10y is not None and len(treasury_10y) > 0:
            market_data['treasury_10y'] = float(treasury_10y['Close'].iloc[-1])
            market_data['treasury_yield_change'] = float(treasury_10y['Close'].iloc[-1] - treasury_10y['Close'].iloc[0])

        # 2년 국채
        treasury_2y = get_stock_data('^IRX', period='1mo')
        if treasury_2y is not None and len(treasury_2y) > 0 and 'treasury_10y' in market_data:
            yield_2y = float(treasury_2y['Close'].iloc[-1])
            market_data['yield_spread'] = market_data['treasury_10y'] - yield_2y

        logger.debug(f"Market data fetched: {list(market_data.keys())}")

    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")

    return market_data


# === 사용 예시 ===
if __name__ == '__main__':
    # 규칙 기반
    detector = EnhancedRegimeDetector(use_ml=False)

    # 더미 데이터
    prices = np.random.randn(100).cumsum() + 100
    volumes = np.random.rand(100) * 1000000

    market_data = fetch_market_data()

    regime, probs, features = detector.detect_regime(prices, volumes, market_data)

    print(f"Regime: {regime}")
    print(f"Probabilities: {probs}")
    print(f"Key features: {features}")

    # 앙상블 가중치
    weights = detector.get_ensemble_weights_for_regime(
        regime, probs, features.get('volatility', 0.05),
        detector.get_days_since_transition()
    )
    print(f"Ensemble weights: {weights}")
