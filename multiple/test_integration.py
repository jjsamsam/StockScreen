#!/usr/bin/env python3
"""
test_integration.py
Enhanced Trading System 통합 테스트 스크립트

이 스크립트는 새로운 모듈들이 올바르게 통합되었는지 테스트합니다.
"""

import sys
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)


def test_module_imports():
    """모듈 import 테스트"""
    print("=" * 60)
    print("1. 모듈 Import 테스트")
    print("=" * 60)

    results = {}

    # 1. Enhanced Regime Detector
    try:
        from regime_detector_enhanced import EnhancedRegimeDetector, fetch_market_data
        results['regime_detector'] = '✅ 성공'
        print("✅ EnhancedRegimeDetector import 성공")
    except ImportError as e:
        results['regime_detector'] = f'❌ 실패: {e}'
        print(f"❌ EnhancedRegimeDetector import 실패: {e}")

    # 2. Ensemble Weight Optimizer
    try:
        from ensemble_weight_optimizer import EnsembleWeightOptimizer, BrierScoreCalculator
        results['weight_optimizer'] = '✅ 성공'
        print("✅ EnsembleWeightOptimizer import 성공")
    except ImportError as e:
        results['weight_optimizer'] = f'❌ 실패: {e}'
        print(f"❌ EnsembleWeightOptimizer import 실패: {e}")

    # 3. Expectancy Calculator
    try:
        from expectancy_calculator import ExpectancyCalculator
        results['expectancy_calc'] = '✅ 성공'
        print("✅ ExpectancyCalculator import 성공")
    except ImportError as e:
        results['expectancy_calc'] = f'❌ 실패: {e}'
        print(f"❌ ExpectancyCalculator import 실패: {e}")

    # 4. Walk-Forward Backtest
    try:
        from walkforward_backtest import WalkForwardBacktest
        results['walkforward'] = '✅ 성공'
        print("✅ WalkForwardBacktest import 성공")
    except ImportError as e:
        results['walkforward'] = f'❌ 실패: {e}'
        print(f"❌ WalkForwardBacktest import 실패: {e}")

    # 5. Stock Prediction (통합 확인)
    try:
        from stock_prediction import EnsemblePredictor
        results['stock_prediction'] = '✅ 성공'
        print("✅ stock_prediction.py import 성공")
    except ImportError as e:
        results['stock_prediction'] = f'❌ 실패: {e}'
        print(f"❌ stock_prediction.py import 실패: {e}")

    return results


def test_regime_detector():
    """Enhanced Regime Detector 기능 테스트"""
    print("\n" + "=" * 60)
    print("2. Enhanced Regime Detector 기능 테스트")
    print("=" * 60)

    try:
        from regime_detector_enhanced import EnhancedRegimeDetector

        detector = EnhancedRegimeDetector(use_ml=False)

        # 더미 데이터
        prices = np.random.randn(100).cumsum() + 100
        volumes = np.random.rand(100) * 1000000

        # 레짐 감지
        regime, probs, features = detector.detect_regime(prices, volumes, market_data={})

        print(f"✅ 레짐 감지: {regime}")
        print(f"   확률: {probs}")
        print(f"   주요 피처: volatility={features.get('volatility', 0):.3f}")

        # 가중치 계산
        weights = detector.get_ensemble_weights_for_regime(
            regime, probs, features.get('volatility', 0.05), 0
        )
        print(f"   앙상블 가중치: {weights}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_weight_optimizer():
    """Ensemble Weight Optimizer 테스트"""
    print("\n" + "=" * 60)
    print("3. Ensemble Weight Optimizer 테스트")
    print("=" * 60)

    try:
        from ensemble_weight_optimizer import EnsembleWeightOptimizer

        optimizer = EnsembleWeightOptimizer(method='adaptive')

        # 가중치 계산
        w_lstm, w_transformer = optimizer.get_weights(
            regime='bull',
            volatility=0.03,
            lstm_brier=0.15,
            transformer_brier=0.12
        )

        print(f"✅ 가중치 계산 성공")
        print(f"   LSTM: {w_lstm:.3f}")
        print(f"   Transformer: {w_transformer:.3f}")

        # Grid Search 테스트
        pred_lstm = np.random.randn(100).cumsum()
        pred_transformer = np.random.randn(100).cumsum()
        actual = np.random.randn(100).cumsum()

        optimal_w = optimizer.optimize_weights_grid_search(
            pred_lstm, pred_transformer, actual
        )

        print(f"✅ Grid Search 최적 가중치: {optimal_w:.3f}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_expectancy_calculator():
    """Expectancy Calculator 테스트"""
    print("\n" + "=" * 60)
    print("4. Expectancy Calculator 테스트")
    print("=" * 60)

    try:
        import pandas as pd
        from expectancy_calculator import ExpectancyCalculator

        calc = ExpectancyCalculator()

        # 더미 거래 데이터
        trades = pd.DataFrame({
            'profit': [1000, -300, 800, -200, 1200, -400],
            'result': ['win', 'loss', 'win', 'loss', 'win', 'loss']
        })

        results = calc.calculate_expectancy(trades)

        print(f"✅ 기대값 계산 성공")
        print(f"   Expectancy: {results['expectancy']:.2f}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   SQN: {results['sqn']:.2f}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


def test_stock_prediction_integration():
    """Stock Prediction 통합 테스트"""
    print("\n" + "=" * 60)
    print("5. Stock Prediction 통합 테스트")
    print("=" * 60)

    try:
        from stock_prediction import EnsemblePredictor

        # EnsemblePredictor 생성 (딥러닝 없이)
        ensemble = EnsemblePredictor(use_deep_learning=False, ticker='TEST')

        # Enhanced 모듈 활성화 확인
        print(f"✅ EnsemblePredictor 생성 성공")
        print(f"   Enhanced Regime 사용: {ensemble.use_enhanced_regime}")

        if ensemble.use_enhanced_regime:
            print(f"   ✨ Enhanced Regime Detector 활성화됨")
            print(f"   ✨ Weight Optimizer 활성화됨")
        else:
            print(f"   ⚠️ Enhanced 모듈 비활성화 (기본 방식 사용)")

        # 간단한 예측 테스트 (오래 걸릴 수 있음)
        # prices = np.random.randn(200).cumsum() + 100
        # result = ensemble.fit_predict(prices, forecast_days=5)
        # print(f"   예측 결과: {result}")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 60)
    print("🚀 Enhanced Trading System 통합 테스트")
    print("=" * 60)

    test_results = {}

    # 1. Import 테스트
    import_results = test_module_imports()
    test_results['imports'] = all('✅' in v for v in import_results.values())

    # 2. 개별 모듈 테스트
    if '✅' in import_results.get('regime_detector', ''):
        test_results['regime_detector'] = test_regime_detector()
    else:
        print("\n⚠️ Enhanced Regime Detector 없음, 테스트 건너뜀")
        test_results['regime_detector'] = None

    if '✅' in import_results.get('weight_optimizer', ''):
        test_results['weight_optimizer'] = test_weight_optimizer()
    else:
        print("\n⚠️ Weight Optimizer 없음, 테스트 건너뜀")
        test_results['weight_optimizer'] = None

    if '✅' in import_results.get('expectancy_calc', ''):
        test_results['expectancy_calculator'] = test_expectancy_calculator()
    else:
        print("\n⚠️ Expectancy Calculator 없음, 테스트 건너뜀")
        test_results['expectancy_calculator'] = None

    # 3. 통합 테스트
    if '✅' in import_results.get('stock_prediction', ''):
        test_results['integration'] = test_stock_prediction_integration()
    else:
        print("\n❌ Stock Prediction import 실패, 통합 테스트 불가")
        test_results['integration'] = False

    # 최종 결과
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)

    for test_name, result in test_results.items():
        if result is True:
            print(f"✅ {test_name}: 통과")
        elif result is False:
            print(f"❌ {test_name}: 실패")
        else:
            print(f"⚠️ {test_name}: 건너뜀")

    # 전체 성공률
    passed = sum(1 for r in test_results.values() if r is True)
    total = len([r for r in test_results.values() if r is not None])

    print("\n" + "=" * 60)
    if total > 0:
        success_rate = passed / total * 100
        print(f"🎯 성공률: {passed}/{total} ({success_rate:.1f}%)")

        if success_rate == 100:
            print("🎉 모든 테스트 통과! Enhanced Trading System 정상 작동")
        elif success_rate >= 80:
            print("✅ 대부분의 테스트 통과! 일부 모듈 확인 필요")
        elif success_rate >= 50:
            print("⚠️ 일부 테스트 실패. 모듈 설치 확인 필요")
        else:
            print("❌ 많은 테스트 실패. 설치 및 통합 확인 필요")
    else:
        print("⚠️ 실행 가능한 테스트 없음")

    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
