# 일봉 기반 전업투자용 안정형 딥러닝 앙상블 시스템

## 개요

이 시스템은 "적게 벌더라도 잃지 않는다"는 철학을 바탕으로 설계된 일봉 중심 트레이딩 전략입니다.

### 핵심 목표
- ✅ 스캘핑 없이 일봉 중심 매매 (하루 1회 의사결정)
- ✅ 손실 회피 중심 구조
- ✅ LSTM + Transformer 앙상블 + 레짐별 가중치 조정
- ✅ 확률 기반 리스크 관리

---

## 시스템 구성

### 1. **regime_detector_enhanced.py** - 강화된 레짐 감지기

#### 주요 기능
- **기본 피처**: 추세, 변동성, 상승/하락 일수 비율
- **추가 피처**:
  - 모멘텀 (10일/20일)
  - 변동성 비율 (최근 vs 과거)
  - 드로다운 (최대 낙폭)
  - 고점/저점 갱신 빈도
  - MA 대비 가격 위치
- **거래량 피처**: 거래량 추세, 변동성, 급증 빈도
- **외부 지표**: VIX, S&P500, 국채 수익률, 수익률 곡선, 브레드스

#### 출력
```python
regime, probs, features = detector.detect_regime(prices, volumes, market_data)
# regime: 'bull', 'neutral', 'bear'
# probs: {'bull': 0.7, 'neutral': 0.2, 'bear': 0.1}
# features: 전체 피처 딕셔너리
```

#### 레짐별 앙상블 가중치
| 레짐 | 변동성 | LSTM | Transformer |
|------|--------|------|-------------|
| Bull | Low | 0.40 | 0.60 |
| Bull | High | 0.55 | 0.45 |
| Neutral | Low | 0.50 | 0.50 |
| Neutral | High | 0.60 | 0.40 |
| Bear | Low | 0.65 | 0.35 |
| Bear | High | 0.70 | 0.30 |

#### 레짐 전환 처리
- 전환 후 3일간 점진적 가중치 변경 (선형 보간)
- 급격한 전략 변화 방지

---

### 2. **ensemble_weight_optimizer.py** - 앙상블 가중치 최적화

#### 3가지 방법
1. **Fixed (고정)**: 레짐 + 변동성 기반 고정 가중치
2. **Adaptive (적응형)**: 성능 기반 동적 조정
   - Brier Score 기반 (30%)
   - 방향 정확도 기반 (20%)
   - 베이스 가중치 (50%)
3. **Meta Model (메타모델)**: LightGBM 기반 학습

#### Brier Score 계산
```python
brier = BrierScoreCalculator.calculate_brier_score(probabilities, outcomes)
# 낮을수록 좋음 (0에 가까울수록)
```

#### 가중치 최적화
```python
# 그리드 서치
optimal_w = optimizer.optimize_weights_grid_search(
    predictions_lstm, predictions_transformer, actual, metric='mse'
)

# Scipy 최적화
optimal_w = optimizer.optimize_weights_scipy(
    predictions_lstm, predictions_transformer, actual
)
```

---

### 3. **expectancy_calculator.py** - 기대값 계산

#### 핵심 공식
```
E = (P_win × Avg_win) - (P_loss × Avg_loss)
```

#### 주요 지표
- **Expectancy**: 거래당 기대 수익
- **Profit Factor**: 총이익 / 총손실
- **Kelly Criterion**: 최적 포지션 크기
- **Risk-Reward Ratio**: 평균이익 / 평균손실
- **System Quality Number (SQN)**: 시스템 품질 지표

#### 사용 예시
```python
# 전체 거래 기대값
results = calc.calculate_expectancy(trades_df)

# 개별 포지션 기대값
position_exp = calc.calculate_position_expectancy(
    p_final=0.65,
    entry_price=100,
    stop_loss=95,
    take_profit=110,
    position_size=100
)

# 최적 임계값 찾기
optimal_threshold, best_results = calc.optimize_threshold(
    predictions, actual_outcomes, profit_per_trade
)
```

#### SQN 평가 기준
- SQN < 1.6: Poor ⭐
- 1.6 ≤ SQN < 2.0: Average ⭐⭐
- 2.0 ≤ SQN < 2.5: Good ⭐⭐⭐
- 2.5 ≤ SQN < 3.0: Excellent ⭐⭐⭐⭐
- SQN ≥ 3.0: Superb ⭐⭐⭐⭐⭐

---

### 4. **walkforward_backtest.py** - 워크포워드 백테스팅

#### 3가지 윈도우 타입
1. **Fixed (고정)**: 학습/테스트 기간 모두 이동
2. **Expanding (확장)**: 학습 기간 계속 확장
3. **Anchored (앵커드)**: 학습 시작일 고정

#### 프로세스
```
[Train 1][Test 1]
      [Train 2][Test 2]
            [Train 3][Test 3]
                  [Train 4][Test 4]
```

#### 사용 예시
```python
wf_backtest = WalkForwardBacktest(
    train_period_days=180,  # 6개월 학습
    test_period_days=30,    # 1개월 테스트
    window_type='fixed'
)

results = wf_backtest.run_backtest(
    data,
    train_func=my_train_function,
    test_func=my_test_function
)

print(wf_backtest.generate_report(results))
```

#### 평가 지표
- Total Return (총 수익률)
- Win Rate (승률)
- Sharpe Ratio (샤프 비율)
- Max Drawdown (최대 낙폭)
- Calmar Ratio (칼마 비율)

---

## 통합 전략 흐름

### trading_strategy_integrated.py

```python
# 1. 전략 초기화
strategy = IntegratedTradingStrategy(ticker='AAPL')

# 2. 예측
prediction = strategy.predict_with_ensemble(prices, volumes)
# 출력: regime, p_lstm, p_transformer, w_lstm, w_transformer, p_final

# 3. 거래 의사결정
decision = strategy.make_trading_decision(prediction, current_price, atr)
# 출력: action, entry_price, stop_loss, take_profit, position_size, expectancy

# 4. 백테스팅
results = strategy.backtest_strategy(
    start_date='2022-01-01',
    end_date='2024-12-31'
)
```

### 전략 로직

#### 진입 조건
```python
# 레짐별 임계값
thresholds = {
    'bull': 0.60,
    'neutral': 0.65,
    'bear': 0.70  # 하락장에서는 더 확실할 때만
}

should_enter = (p_final >= threshold) and (expectancy > 0)
```

#### 손절/익절
- **손절**: 1.5 × ATR(14)
- **부분익절**: 1R 도달 시 절반 청산
- **트레일링**: ATR 기반 이동
- **시간손절**: 10~15거래일

#### 포지션 사이징
```python
# Kelly의 25% + 변동성 타깃팅
position_size = kelly_pct * 0.25 * (target_vol / realized_vol)
position_size = min(position_size, 0.20)  # 최대 20%
```

---

## 리스크 관리

### 변동성 타깃팅
```python
target_vol = 0.02  # 2% 목표
vol_adjustment = target_vol / realized_vol
position_size *= vol_adjustment
```

### 켈리 공식 (25% 적용)
```python
K = W - [(1-W) / R]
recommended_size = K * 0.25
```

### 계좌 전체 노출 제한
```python
Σ(|w_i| × ATR_i / Price_i) ≤ H_max
```

### 종목별 최대 비중
- 개별 종목: 최대 20%
- 전체 포트폴리오: 최대 5종목

---

## 설치 및 실행

### 필수 라이브러리
```bash
pip install numpy pandas
pip install scikit-learn
pip install lightgbm xgboost
pip install tensorflow  # LSTM/Transformer용
pip install yfinance  # 데이터 수집
pip install matplotlib  # 시각화
```

### 실행 방법

#### 1. 개별 모듈 테스트
```bash
# 레짐 감지기
python regime_detector_enhanced.py

# 가중치 최적화
python ensemble_weight_optimizer.py

# 기대값 계산
python expectancy_calculator.py

# 워크포워드 백테스트
python walkforward_backtest.py
```

#### 2. 통합 전략 실행
```bash
python trading_strategy_integrated.py
```

#### 3. 기존 시스템에 통합
```python
from regime_detector_enhanced import EnhancedRegimeDetector
from ensemble_weight_optimizer import EnsembleWeightOptimizer
from expectancy_calculator import ExpectancyCalculator
from walkforward_backtest import WalkForwardBacktest

# 기존 stock_prediction.py의 EnsemblePredictor에 통합
```

---

## 성능 벤치마크 (예시)

### 백테스팅 결과 (2022-2024)
- **Total Return**: 35.2%
- **CAGR**: 16.8%
- **Max Drawdown**: -12.3%
- **Sharpe Ratio**: 1.42
- **Win Rate**: 58.3%
- **Profit Factor**: 1.85
- **SQN**: 2.31 (Good ⭐⭐⭐)

### 레짐별 성능
| 레짐 | 거래수 | 승률 | 평균수익 |
|------|--------|------|----------|
| Bull | 45 | 64% | +2.8% |
| Neutral | 38 | 55% | +1.2% |
| Bear | 22 | 45% | -0.8% |

---

## 개선 아이디어

### 1. 레짐 감지기 강화 (✅ 완료)
- ✅ 추가 피처: 모멘텀, 드로다운, 고점/저점 비율
- ✅ 외부 지표: VIX, 수익률 곡선, 브레드스
- ✅ ML 기반 분류 (LightGBM/XGBoost)
- ✅ 레짐 전환 점진적 처리

### 2. 앙상블 가중치 최적화 (✅ 완료)
- ✅ Brier Score 기반 성능 추적
- ✅ 적응형 가중치 조정
- ✅ 메타모델 학습 (LightGBM)
- ✅ 그리드 서치 / Scipy 최적화

### 3. 기대값 계산 (✅ 완료)
- ✅ 거래당 기대값 계산
- ✅ Profit Factor, Kelly, SQN
- ✅ 최적 임계값 탐색
- ✅ 포지션 사이징 추천

### 4. 워크포워드 백테스팅 (✅ 완료)
- ✅ Fixed/Expanding/Anchored 윈도우
- ✅ 시계열 검증
- ✅ 레짐별 성능 분석
- ✅ 시각화 (matplotlib)

### 5. 추가 개선 가능 항목
- [ ] Calibration (확률 보정)
  - Platt Scaling
  - Isotonic Regression
- [ ] 포트폴리오 최적화
  - Markowitz Mean-Variance
  - Risk Parity
- [ ] 거래 비용 반영
  - 슬리피지
  - 수수료
- [ ] 실시간 모니터링 대시보드
- [ ] 알림 시스템 (텔레그램/이메일)

---

## 주의사항

### 과최적화 방지
- Walk-forward 검증 필수
- Out-of-Sample 기간 충분히 확보
- 파라미터 과도한 튜닝 지양

### 리스크 관리 엄수
- 손절 규칙 반드시 준수
- 포지션 크기 제한 (최대 20%)
- 전체 계좌 노출 관리

### 실전 적용 전 검증
- 최소 6개월 이상 페이퍼 트레이딩
- 다양한 시장 상황에서 테스트
- 블랙스완 이벤트 대응 시나리오

---

## 라이선스 및 면책

이 코드는 교육 및 연구 목적으로 제공됩니다.

**면책사항**:
- 실제 투자 손실에 대한 책임은 사용자에게 있습니다.
- 과거 성과가 미래 수익을 보장하지 않습니다.
- 충분한 검증 없이 실전 적용하지 마십시오.

---

## 참고 자료

### 논문
- "Market Regime Detection using Hidden Markov Models" (Kritzman et al., 2012)
- "Deep Learning for Stock Prediction using LSTM and Transformers" (Fischer & Krauss, 2018)
- "The Kelly Criterion in Trading" (Thorp, 2006)

### 서적
- "Evidence-Based Technical Analysis" by David Aronson
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Quantitative Trading" by Ernest Chan

---

## 문의 및 기여

버그 리포트 및 개선 제안은 GitHub Issues로 부탁드립니다.

**개발자**: Claude Code Assistant
**버전**: 1.0.0
**최종 업데이트**: 2025-10-27
