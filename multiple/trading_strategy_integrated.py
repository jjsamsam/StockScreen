"""
trading_strategy_integrated.py
일봉 기반 전업투자용 안정형 딥러닝 앙상블 시스템 - 통합 예시

이 파일은 4가지 핵심 모듈을 통합하여 실전 트레이딩 전략을 구현합니다:

1. EnhancedRegimeDetector - 강화된 레짐 감지
2. EnsembleWeightOptimizer - 동적 가중치 최적화
3. ExpectancyCalculator - 기대값 계산
4. WalkForwardBacktest - 워크포워드 백테스팅

전략 흐름:
1. 시장 데이터 수집
2. 레짐 감지 (Bull/Neutral/Bear)
3. LSTM/Transformer 예측
4. 동적 가중치로 앙상블
5. 기대값 계산 및 포지션 결정
6. 워크포워드 백테스팅으로 검증
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from logger_config import get_logger

# 새로 생성한 모듈들
from regime_detector_enhanced import EnhancedRegimeDetector, fetch_market_data
from ensemble_weight_optimizer import EnsembleWeightOptimizer, BrierScoreCalculator
from expectancy_calculator import ExpectancyCalculator
from walkforward_backtest import WalkForwardBacktest

# 기존 모듈
from cache_manager import get_stock_data

logger = get_logger(__name__)


class IntegratedTradingStrategy:
    """
    통합 트레이딩 전략

    핵심 특징:
    - 레짐별 동적 가중치 조정
    - 확률 기반 진입/청산
    - 기대값 최적화
    - 리스크 관리 (변동성 타깃팅, Kelly Criterion)
    """

    def __init__(self, ticker: str):
        """
        Args:
            ticker: 종목 코드
        """
        self.ticker = ticker

        # 모듈 초기화
        self.regime_detector = EnhancedRegimeDetector(use_ml=False)
        self.weight_optimizer = EnsembleWeightOptimizer(method='adaptive')
        self.expectancy_calc = ExpectancyCalculator()

        # 모델 (실제로는 stock_prediction.py에서 불러옴)
        self.lstm_model = None
        self.transformer_model = None

        # 성능 추적
        self.lstm_brier_scores = []
        self.transformer_brier_scores = []

        logger.info(f"IntegratedTradingStrategy initialized for {ticker}")

    def initialize_models(self):
        """
        딥러닝 모델 초기화 (실제로는 stock_prediction.py 사용)
        """
        try:
            from stock_prediction import LSTMPredictor, TransformerPredictor

            self.lstm_model = LSTMPredictor(ticker=self.ticker, auto_load=True)
            self.transformer_model = TransformerPredictor(ticker=self.ticker, auto_load=True)

            logger.info("Models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.lstm_model = None
            self.transformer_model = None

    def predict_with_ensemble(self,
                              prices: np.ndarray,
                              volumes: Optional[np.ndarray] = None,
                              forecast_days: int = 5) -> Dict:
        """
        앙상블 예측 실행

        Returns:
            {
                'regime': 'bull',
                'p_lstm': 0.55,
                'p_transformer': 0.65,
                'w_lstm': 0.4,
                'w_transformer': 0.6,
                'p_final': 0.61,
                'volatility': 0.03,
                'features': {...}
            }
        """
        # 1. 시장 데이터 가져오기
        market_data = fetch_market_data()

        # 2. 레짐 감지
        regime, regime_probs, features = self.regime_detector.detect_regime(
            prices, volumes, market_data, window=50
        )

        volatility = features.get('volatility', 0.05)

        # 3. LSTM/Transformer 예측 (더미 - 실제로는 모델 사용)
        if self.lstm_model and self.transformer_model:
            try:
                # LSTM 예측
                lstm_result = self.lstm_model.fit_predict(prices, forecast_days=forecast_days)
                p_lstm = lstm_result.get('confidence', 0.5)

                # Transformer 예측
                transformer_result = self.transformer_model.fit_predict(prices, forecast_days=forecast_days)
                p_transformer = transformer_result.get('confidence', 0.5)

            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                p_lstm = 0.5
                p_transformer = 0.5
        else:
            # 더미 예측
            p_lstm = 0.55
            p_transformer = 0.65

        # 4. 최근 성능 기반 Brier Score (더미)
        lstm_brier = np.mean(self.lstm_brier_scores[-20:]) if self.lstm_brier_scores else 0.15
        transformer_brier = np.mean(self.transformer_brier_scores[-20:]) if self.transformer_brier_scores else 0.12

        # 5. 동적 가중치 계산
        w_lstm, w_transformer = self.weight_optimizer.get_weights(
            regime=regime,
            volatility=volatility,
            lstm_brier=lstm_brier,
            transformer_brier=transformer_brier
        )

        # 6. 앙상블 확률
        p_final = w_lstm * p_lstm + w_transformer * p_transformer

        logger.info(f"Ensemble prediction: Regime={regime}, p_final={p_final:.3f} "
                   f"(LSTM={p_lstm:.3f}×{w_lstm:.2f}, Transformer={p_transformer:.3f}×{w_transformer:.2f})")

        return {
            'regime': regime,
            'regime_probs': regime_probs,
            'p_lstm': p_lstm,
            'p_transformer': p_transformer,
            'w_lstm': w_lstm,
            'w_transformer': w_transformer,
            'p_final': p_final,
            'volatility': volatility,
            'features': features
        }

    def make_trading_decision(self,
                              prediction: Dict,
                              current_price: float,
                              atr: float) -> Dict:
        """
        거래 의사결정

        Args:
            prediction: 예측 결과
            current_price: 현재가
            atr: ATR (14일)

        Returns:
            거래 결정 딕셔너리
        """
        p_final = prediction['p_final']
        regime = prediction['regime']
        volatility = prediction['volatility']

        # 레짐별 임계값
        thresholds = {
            'bull': 0.60,
            'neutral': 0.65,
            'bear': 0.70  # 하락장에서는 더 확실할 때만 진입
        }

        threshold = thresholds.get(regime, 0.65)

        # 진입 조건
        should_enter = p_final >= threshold

        if not should_enter:
            return {
                'action': 'hold',
                'reason': f'p_final ({p_final:.3f}) < threshold ({threshold:.3f})',
                'position_size': 0
            }

        # 손절/익절 설정
        stop_loss = current_price - (1.5 * atr)
        take_profit = current_price + (2.5 * atr)  # 1.5R 목표

        # 기대값 계산
        expectancy_result = self.expectancy_calc.calculate_position_expectancy(
            p_final=p_final,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=1.0  # 기본 단위
        )

        # 기대값이 음수면 진입하지 않음
        if expectancy_result['expectancy'] <= 0:
            return {
                'action': 'hold',
                'reason': f'Negative expectancy: {expectancy_result["expectancy"]:.2f}',
                'position_size': 0
            }

        # 포지션 사이징 (Kelly의 25%)
        kelly_pct = expectancy_result['kelly_pct']
        position_size = kelly_pct * 0.25  # 보수적

        # 변동성 타깃팅 추가 조정
        target_vol = 0.02  # 2% 목표 변동성
        vol_adjustment = target_vol / volatility if volatility > 0 else 1.0
        position_size *= vol_adjustment

        # 최대 비중 제한
        position_size = min(position_size, 0.20)  # 최대 20%

        return {
            'action': 'buy',
            'reason': f'Strong signal: p_final={p_final:.3f}, expectancy={expectancy_result["expectancy"]:.2f}',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'expectancy': expectancy_result['expectancy'],
            'expectancy_pct': expectancy_result['expectancy_pct'],
            'risk_reward_ratio': expectancy_result['risk_reward_ratio'],
            'kelly_pct': kelly_pct
        }

    def backtest_strategy(self,
                          start_date: str,
                          end_date: str,
                          initial_capital: float = 10000000) -> pd.DataFrame:
        """
        워크포워드 백테스팅

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일
            initial_capital: 초기 자본

        Returns:
            백테스팅 결과 데이터프레임
        """
        # 데이터 가져오기
        data = get_stock_data(self.ticker, period='max')
        data = data.loc[start_date:end_date]

        # 워크포워드 백테스트 설정
        wf_backtest = WalkForwardBacktest(
            train_period_days=180,  # 6개월 학습
            test_period_days=30,    # 1개월 테스트
            window_type='fixed'
        )

        # 학습 함수
        def train_func(train_data):
            """모델 재학습 또는 파라미터 최적화"""
            # 여기서는 단순화 - 실제로는 모델 재학습
            return {'trained': True}

        # 테스트 함수
        def test_func(model, test_data):
            """테스트 기간 거래 시뮬레이션"""
            prices = test_data['Close'].values
            volumes = test_data['Volume'].values if 'Volume' in test_data else None

            # ATR 계산 (단순화)
            high = test_data['High'].values
            low = test_data['Low'].values
            atr = np.mean(high - low)

            # 예측 및 거래
            prediction = self.predict_with_ensemble(prices, volumes)
            decision = self.make_trading_decision(prediction, prices[-1], atr)

            # 간단한 수익률 계산 (실제로는 더 정교하게)
            if decision['action'] == 'buy':
                # 다음 기간 수익률 추정 (더미)
                future_return = np.random.randn() * 0.05  # ±5%
                actual_return = future_return * decision['position_size']
            else:
                actual_return = 0.0

            # Sharpe Ratio (더미)
            sharpe = actual_return / 0.02 if actual_return != 0 else 0

            return {
                'return': actual_return,
                'sharpe': sharpe,
                'regime': prediction['regime'],
                'p_final': prediction['p_final'],
                'position_size': decision.get('position_size', 0)
            }

        # 백테스트 실행
        results = wf_backtest.run_backtest(
            data,
            train_func=train_func,
            test_func=test_func
        )

        # 리포트 출력
        print(wf_backtest.generate_report(results))

        return results


# === 메인 실행 예시 ===
def main():
    """통합 전략 실행 예시"""

    # 1. 전략 초기화
    strategy = IntegratedTradingStrategy(ticker='AAPL')

    # 2. 모델 초기화 (옵션)
    # strategy.initialize_models()

    # 3. 현재 시점 예측
    print("=" * 60)
    print("현재 시점 예측")
    print("=" * 60)

    # 더미 데이터
    prices = np.random.randn(200).cumsum() + 100
    volumes = np.random.rand(200) * 1000000

    prediction = strategy.predict_with_ensemble(prices, volumes)

    print(f"레짐: {prediction['regime']}")
    print(f"레짐 확률: {prediction['regime_probs']}")
    print(f"LSTM 예측: {prediction['p_lstm']:.3f} (가중치: {prediction['w_lstm']:.2f})")
    print(f"Transformer 예측: {prediction['p_transformer']:.3f} (가중치: {prediction['w_transformer']:.2f})")
    print(f"최종 앙상블 확률: {prediction['p_final']:.3f}")
    print(f"변동성: {prediction['volatility']:.2%}")

    # 4. 거래 의사결정
    print("\n" + "=" * 60)
    print("거래 의사결정")
    print("=" * 60)

    current_price = prices[-1]
    atr = np.std(prices[-20:]) * 1.5

    decision = strategy.make_trading_decision(prediction, current_price, atr)

    print(f"액션: {decision['action'].upper()}")
    print(f"이유: {decision['reason']}")

    if decision['action'] == 'buy':
        print(f"진입가: {decision['entry_price']:,.2f}")
        print(f"손절가: {decision['stop_loss']:,.2f}")
        print(f"목표가: {decision['take_profit']:,.2f}")
        print(f"포지션 크기: {decision['position_size']:.2%}")
        print(f"기대값: {decision['expectancy']:.2f}")
        print(f"리스크-리워드: {decision['risk_reward_ratio']:.2f}")

    # 5. 워크포워드 백테스팅 (시간이 오래 걸림 - 주석 처리)
    # print("\n" + "=" * 60)
    # print("워크포워드 백테스팅")
    # print("=" * 60)
    #
    # results = strategy.backtest_strategy(
    #     start_date='2022-01-01',
    #     end_date='2024-12-31',
    #     initial_capital=10000000
    # )


if __name__ == '__main__':
    main()
