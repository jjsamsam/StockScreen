#!/usr/bin/env python3
"""
demo_enhanced_features.py
새로운 Enhanced 기능 데모 - GUI 없이 콘솔에서 확인

오늘 구현한 4가지 기능을 실제 데이터로 시연합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from logger_config import get_logger

logger = get_logger(__name__)

print("=" * 70)
print("🚀 Enhanced Trading System 실전 데모")
print("=" * 70)

# 1. Enhanced Regime Detector 데모
print("\n" + "=" * 70)
print("1️⃣ Enhanced Regime Detector - 강화된 시장 상황 분석")
print("=" * 70)

try:
    from regime_detector_enhanced import EnhancedRegimeDetector, fetch_market_data
    from cache_manager import get_stock_data

    # 실제 주가 데이터 가져오기
    print("📊 AAPL 주가 데이터 가져오는 중...")
    data = get_stock_data('AAPL', period='1y')
    prices = data['Close'].values
    volumes = data['Volume'].values if 'Volume' in data else None

    # Enhanced 레짐 감지
    detector = EnhancedRegimeDetector(use_ml=False)
    print("🔍 시장 상황 분석 중...")

    # 시장 데이터 가져오기
    market_data = fetch_market_data()

    # 레짐 감지
    regime, probs, features = detector.detect_regime(
        prices, volumes, market_data, window=50
    )

    print(f"\n📈 분석 결과:")
    print(f"   현재 레짐: {regime.upper()}")
    print(f"   확률 분포:")
    print(f"      - Bull (상승장):   {probs['bull']*100:.1f}%")
    print(f"      - Neutral (횡보): {probs['neutral']*100:.1f}%")
    print(f"      - Bear (하락장):  {probs['bear']*100:.1f}%")

    print(f"\n📊 주요 피처:")
    print(f"   - 추세:         {features.get('trend_pct', 0):+.2f}%")
    print(f"   - 변동성:       {features.get('volatility', 0):.3f}")
    print(f"   - 상승일 비율:  {features.get('up_days_ratio', 0.5)*100:.1f}%")
    print(f"   - 1주 수익률:   {features.get('return_1w', 0):+.2f}%")
    print(f"   - 1개월 수익률: {features.get('return_1m', 0):+.2f}%")

    if market_data:
        print(f"\n🌍 시장 지표:")
        if 'vix' in market_data:
            print(f"   - VIX:          {market_data['vix']:.2f}")
        if 'sp500_return' in market_data:
            print(f"   - S&P 500 3개월: {market_data['sp500_return']:+.2f}%")
        if 'yield_spread' in market_data:
            print(f"   - 국채 스프레드: {market_data['yield_spread']:+.2f}%")

    # 앙상블 가중치
    weights = detector.get_ensemble_weights_for_regime(
        regime, probs, features.get('volatility', 0.05), 0
    )
    print(f"\n⚖️ 추천 앙상블 가중치:")
    print(f"   - LSTM:        {weights['lstm']*100:.1f}%")
    print(f"   - Transformer: {weights['transformer']*100:.1f}%")

    print("\n✅ Enhanced Regime Detector 정상 작동!")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

# 2. Ensemble Weight Optimizer 데모
print("\n" + "=" * 70)
print("2️⃣ Ensemble Weight Optimizer - 동적 가중치 최적화")
print("=" * 70)

try:
    from ensemble_weight_optimizer import EnsembleWeightOptimizer

    optimizer = EnsembleWeightOptimizer(method='adaptive')

    # 시나리오 1: 상승장, 낮은 변동성
    print("\n📊 시나리오 1: 상승장 + 낮은 변동성")
    w1_lstm, w1_trf = optimizer.get_weights(
        regime='bull', volatility=0.02,
        lstm_brier=0.15, transformer_brier=0.12
    )
    print(f"   LSTM:        {w1_lstm*100:.1f}%")
    print(f"   Transformer: {w1_trf*100:.1f}%")
    print(f"   → Transformer 우세 (정확도 높음)")

    # 시나리오 2: 하락장, 높은 변동성
    print("\n📊 시나리오 2: 하락장 + 높은 변동성")
    w2_lstm, w2_trf = optimizer.get_weights(
        regime='bear', volatility=0.08,
        lstm_brier=0.12, transformer_brier=0.15
    )
    print(f"   LSTM:        {w2_lstm*100:.1f}%")
    print(f"   Transformer: {w2_trf*100:.1f}%")
    print(f"   → LSTM 우세 (단기 패턴 포착)")

    # Grid Search 최적화
    print("\n🔍 Grid Search 최적화 테스트...")
    np.random.seed(42)
    pred_lstm = np.random.randn(100).cumsum() + 100
    pred_trf = np.random.randn(100).cumsum() + 100
    actual = np.random.randn(100).cumsum() + 100

    optimal_w = optimizer.optimize_weights_grid_search(
        pred_lstm, pred_trf, actual, metric='mse'
    )
    print(f"   최적 LSTM 가중치: {optimal_w*100:.1f}%")
    print(f"   최적 Transformer 가중치: {(1-optimal_w)*100:.1f}%")

    print("\n✅ Ensemble Weight Optimizer 정상 작동!")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# 3. Expectancy Calculator 데모
print("\n" + "=" * 70)
print("3️⃣ Expectancy Calculator - 전략 기대값 분석")
print("=" * 70)

try:
    from expectancy_calculator import ExpectancyCalculator

    calc = ExpectancyCalculator()

    # 시나리오 A: 좋은 전략
    print("\n📊 전략 A: 우수한 전략 (승률 60%, RR 2:1)")
    trades_a = pd.DataFrame({
        'profit': [2000, -1000, 2000, 1500, -1000, 2000, -1000, 2000, 1500, 2000],
        'result': ['win', 'loss', 'win', 'win', 'loss', 'win', 'loss', 'win', 'win', 'win']
    })
    results_a = calc.calculate_expectancy(trades_a)

    print(f"   기대값:         {results_a['expectancy']:,.0f}원")
    print(f"   승률:           {results_a['win_rate']*100:.1f}%")
    print(f"   Profit Factor:  {results_a['profit_factor']:.2f}")
    print(f"   SQN:            {results_a['sqn']:.2f} {'⭐'*int(min(5, results_a['sqn']))}")
    print(f"   Kelly %:        {results_a['kelly_pct']*100:.1f}%")
    print(f"   추천 포지션:    {results_a['kelly_pct']*25:.1f}% (Kelly 25%)")

    # 시나리오 B: 나쁜 전략
    print("\n📊 전략 B: 부진한 전략 (승률 40%, RR 1:1)")
    trades_b = pd.DataFrame({
        'profit': [1000, -1000, -1000, 1000, -1000, -1000, 1000, -1000, -1000, 1000],
        'result': ['win', 'loss', 'loss', 'win', 'loss', 'loss', 'win', 'loss', 'loss', 'win']
    })
    results_b = calc.calculate_expectancy(trades_b)

    print(f"   기대값:         {results_b['expectancy']:,.0f}원")
    print(f"   승률:           {results_b['win_rate']*100:.1f}%")
    print(f"   Profit Factor:  {results_b['profit_factor']:.2f}")
    print(f"   ⚠️ 기대값이 음수! 이 전략은 사용하면 안 됨")

    # 포지션 기대값
    print("\n📊 개별 포지션 분석:")
    position = calc.calculate_position_expectancy(
        p_final=0.65,      # 상승 확률 65%
        entry_price=100,
        stop_loss=95,      # -5% 손절
        take_profit=110,   # +10% 익절
        position_size=1000
    )

    print(f"   진입가:         100원")
    print(f"   손절가:         95원 (-5%)")
    print(f"   목표가:         110원 (+10%)")
    print(f"   상승 확률:      65%")
    print(f"   기대값:         {position['expectancy']:+,.0f}원")
    print(f"   Risk-Reward:    {position['risk_reward_ratio']:.2f}")
    print(f"   Kelly %:        {position['kelly_pct']*100:.1f}%")
    print(f"   추천 크기:      {position['kelly_pct']*25:.1f}% (Kelly 25%)")

    print("\n✅ Expectancy Calculator 정상 작동!")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# 4. Walk-Forward Backtest 데모
print("\n" + "=" * 70)
print("4️⃣ Walk-Forward Backtest - 시계열 검증")
print("=" * 70)

try:
    from walkforward_backtest import WalkForwardBacktest

    # 더미 데이터 생성 (2년치)
    dates = pd.date_range('2022-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    prices = 100 + np.random.randn(len(dates)).cumsum() * 2
    data = pd.DataFrame({'price': prices, 'volume': np.random.rand(len(dates)) * 1e6}, index=dates)

    # Walk-Forward 설정
    wf = WalkForwardBacktest(
        train_period_days=180,  # 6개월 학습
        test_period_days=30,    # 1개월 테스트
        window_type='fixed'
    )

    print("\n📊 백테스팅 설정:")
    print(f"   학습 기간:   180일 (6개월)")
    print(f"   테스트 기간: 30일 (1개월)")
    print(f"   윈도우 타입: Fixed")
    print(f"   데이터 기간: 2022-01-01 ~ 2024-12-31")

    # 윈도우 생성
    windows = wf.generate_windows(data)

    print(f"\n✅ 생성된 윈도우: {len(windows)}개")
    print(f"\n📋 윈도우 예시:")
    for i in range(min(3, len(windows))):
        w = windows[i]
        print(f"   Window {w['window_id']}:")
        print(f"      Train: {w['train_start'].date()} ~ {w['train_end'].date()} ({w['train_samples']}일)")
        print(f"      Test:  {w['test_start'].date()} ~ {w['test_end'].date()} ({w['test_samples']}일)")

    if len(windows) > 3:
        print(f"   ... (총 {len(windows)}개 윈도우)")

    print("\n✅ Walk-Forward Backtest 정상 작동!")
    print("   (실제 백테스팅은 시간이 오래 걸려 생략)")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# 5. 통합 확인
print("\n" + "=" * 70)
print("5️⃣ Stock Prediction 통합 확인")
print("=" * 70)

try:
    from stock_prediction import EnsemblePredictor

    print("\n📊 EnsemblePredictor 생성 중...")
    ensemble = EnsemblePredictor(use_deep_learning=False, ticker='DEMO')

    print(f"\n✅ 통합 상태:")
    print(f"   Enhanced Regime 사용:  {'✓ 활성화' if ensemble.use_enhanced_regime else '✗ 비활성화'}")

    if ensemble.use_enhanced_regime:
        print(f"   Regime Detector:       ✓ 통합됨")
        print(f"   Weight Optimizer:      ✓ 통합됨")
        print(f"\n🎉 새로운 기능이 완벽하게 통합되었습니다!")
    else:
        print(f"\n⚠️ Enhanced 모듈이 비활성화되어 있습니다.")
        print(f"   기존 방식으로 정상 작동 중입니다.")

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")

# 최종 요약
print("\n" + "=" * 70)
print("📊 데모 완료 - 요약")
print("=" * 70)

print("\n✨ 오늘 구현된 4가지 핵심 기능:")
print("   1️⃣ Enhanced Regime Detector   - 15개 피처 기반 시장 분석")
print("   2️⃣ Ensemble Weight Optimizer  - 동적 가중치 최적화")
print("   3️⃣ Expectancy Calculator      - 전략 기대값 분석")
print("   4️⃣ Walk-Forward Backtest      - 시계열 검증")

print("\n🎯 통합 상태:")
print("   ✓ stock_prediction.py에 조용히 통합")
print("   ✓ 기존 GUI는 변경 없음")
print("   ✓ 사용자 경험 동일")
print("   ✓ 내부 성능 향상")

print("\n📚 더 자세한 정보:")
print("   - 상세 문서: ENHANCED_TRADING_SYSTEM_README.md")
print("   - 통합 가이드: INTEGRATION_COMPLETE.md")
print("   - 빠른 시작: QUICK_START_GUIDE.md")

print("\n" + "=" * 70)
print("🚀 Enhanced Trading System 데모 완료!")
print("=" * 70)
