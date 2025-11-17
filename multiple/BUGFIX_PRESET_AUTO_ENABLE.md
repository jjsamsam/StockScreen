# 버그 수정: 프리셋 선택 시 기본 조건 자동 활성화

## 문제 상황

고급 스크리닝에서 안전형 프리셋을 선택하고 2600개 종목을 스크리닝했을 때 **매수/매도 후보가 0개** 나오는 문제가 발생했습니다.

## 원인 분석

### 근본 원인

고급 스크리닝은 **기본 매수 조건 + 고급 조건**을 결합하는 구조입니다:

```python
# analyze_stock_advanced() 로직
if buy_signals:  # 기본 매수 신호가 있을 때만
    all_signals = buy_signals + advanced_signals  # 고급 신호 추가
    return result
```

그런데 기본 매수 조건 체크박스들은 **탭 2 (기본 스크리닝)**에 있고, 사용자가 프리셋만 선택하고 스크리닝을 실행하면:

1. 고급 조건만 체크됨 (다중 시간대 ✓, 시장 강도 ✓)
2. 기본 매수 조건은 체크되지 않음 (MA ✗, BB ✗, MACD ✗, 모멘텀 ✗)
3. `buy_signals = []` (빈 리스트)
4. `if buy_signals:` → False
5. **결과 없음**

### 왜 이렇게 설계되었나?

원래 의도는 다음과 같았습니다:

```
고급 스크리닝 = 기본 스크리닝 + 추가 고급 필터
```

- 기본 조건으로 1차 필터링 (예: 100개 후보)
- 고급 조건으로 2차 필터링 (예: 10개 고품질 후보)

하지만 탭이 분리되면서 사용자가 기본 조건 체크박스를 찾기 어려워졌습니다.

## 해결 방법

### 수정 내용

프리셋 선택 시 **기본 매수 조건도 자동으로 활성화**하도록 변경했습니다.

**파일**: `screener.py:1362-1407`

**변경 전**:
```python
def apply_advanced_preset(self, preset_name):
    if preset_name == "🛡️ 안전형":
        self.adv_multi_timeframe.setChecked(True)
        self.adv_market_strength.setChecked(True)
        # 기본 조건은 활성화 안 함 ❌
```

**변경 후**:
```python
def apply_advanced_preset(self, preset_name):
    if preset_name == "🛡️ 안전형":
        # 고급 조건 활성화
        self.adv_multi_timeframe.setChecked(True)
        self.adv_market_strength.setChecked(True)

        # 기본 매수 조건도 활성화 ✅
        self.ma_condition.setChecked(True)
        self.bb_condition.setChecked(True)
        self.support_condition.setChecked(True)
        self.momentum_condition.setChecked(True)
```

### 추가 개선

**안내 메시지 추가** (`screener.py:1267-1274`):

```python
info_label = QLabel(
    "💡 <b>사용 방법</b>: 먼저 프리셋을 선택하면 기본 매수 조건과 고급 조건이 자동으로 설정됩니다. "
    "기본 매수 조건은 '📊 기본 스크리닝' 탭에서 확인할 수 있습니다."
)
```

이제 사용자가 고급 스크리닝 탭을 열면 바로 안내 메시지를 볼 수 있습니다.

## 테스트 시나리오

### 시나리오 1: 안전형 프리셋 (수정 후)

**단계**:
1. 고급 스크리닝 탭 이동
2. 프리셋: "🛡️ 안전형" 선택
3. 스크리닝 실행

**예상 결과**:
- 기본 매수 조건 자동 체크: MA ✓, BB ✓, MACD ✓, 모멘텀 ✓
- 고급 조건 자동 체크: 다중 시간대 ✓, 시장 강도 ✓
- 매수 후보 발견됨 (0개가 아님)

### 시나리오 2: 사용자 정의 (기존과 동일)

**단계**:
1. 프리셋: "사용자 정의" 선택
2. 원하는 조건만 수동 체크
3. 스크리닝 실행

**예상 결과**:
- 자동 체크 안 됨
- 사용자가 선택한 조건만 사용

## 동작 로직

### 전체 흐름

```
사용자: "🛡️ 안전형" 선택
    ↓
apply_advanced_preset() 호출
    ↓
1. 고급 조건 체크
   - adv_multi_timeframe.setChecked(True)
   - adv_market_strength.setChecked(True)
    ↓
2. 기본 매수 조건 체크 (NEW!)
   - ma_condition.setChecked(True)
   - bb_condition.setChecked(True)
   - support_condition.setChecked(True)
   - momentum_condition.setChecked(True)
    ↓
사용자: "🚀 고급 스크리닝 시작" 클릭
    ↓
run_advanced_screening()
    ↓
analyze_stock_advanced() 호출 (각 종목)
    ↓
1. 기본 조건 체크 → buy_signals = ["MA돌파+터치", ...]
2. 고급 조건 체크 → advanced_signals = ["다중시간대✓", "시장강세✓"]
3. 결합 → all_signals = ["MA돌파+터치", "다중시간대✓", "시장강세✓"]
    ↓
매수 후보 생성 ✅
```

### 조건 결합 방식

```python
# 기본 신호 (4개 조건 중 하나라도 만족)
buy_signals = []
if self.ma_condition.isChecked() and MA조건만족:
    buy_signals.append("MA돌파+터치")
if self.bb_condition.isChecked() and BB조건만족:
    buy_signals.append("볼린저하단+RSI")
# ...

# 고급 신호 (모두 만족해야 함)
advanced_signals = []
if active_conditions['multi_timeframe']:
    if 다중시간대만족:
        advanced_signals.append("다중시간대✓")
    else:
        return None  # ❌ 실패 시 즉시 제외

if active_conditions['market_strength']:
    if 시장강세:
        advanced_signals.append("시장강세✓")
    else:
        return None  # ❌ 실패 시 즉시 제외

# 결합
if buy_signals:  # 기본 신호가 있어야만
    all_signals = buy_signals + advanced_signals
    return {'signals': ', '.join(all_signals), ...}
```

## 추가 개선 사항 (향후)

### Option 1: 독립적인 고급 조건 (현재는 X)

고급 조건만으로도 매수 신호를 생성:

```python
# 현재
if buy_signals:
    all_signals = buy_signals + advanced_signals
    return result

# 대안 (향후 고려)
if buy_signals or advanced_signals:
    all_signals = buy_signals + advanced_signals
    return result
```

**장점**: 기본 조건 없이도 고급 조건만으로 후보 선정 가능
**단점**: 너무 많은 후보 발생 가능 (고급 조건만으로는 필터링이 약함)

### Option 2: 고급 탭에 기본 조건 체크박스 복사

고급 스크리닝 탭에도 기본 조건 체크박스 추가:

```python
# 고급 탭에 추가
buy_conditions_group = QGroupBox("💰 기본 매수 조건")
# (탭 2와 동일한 체크박스들)
```

**장점**: 사용자가 한 탭에서 모든 것을 설정 가능
**단점**: UI 중복, 코드 복잡도 증가

### Option 3: 현재 방식 유지 (✅ 선택됨)

프리셋 선택 시 자동 활성화, 사용자 정의 시 수동 선택:

**장점**:
- 간단하고 직관적
- 코드 중복 없음
- 프리셋의 목적에 부합 (빠른 설정)

**단점**:
- 탭 간 의존성 존재
- 하지만 안내 메시지로 해결됨

## 변경 사항 요약

### 수정된 파일

**screener.py**:
- `apply_advanced_preset()`: 기본 조건 자동 활성화 추가 (4 lines × 3 presets = 12 lines)
- `create_advanced_screening_tab()`: 안내 메시지 추가 (7 lines)
- 총 19 lines 추가

### 영향 범위

**변경됨**:
- ✅ 프리셋 선택 시 동작 (기본 조건 자동 체크)
- ✅ 고급 탭 UI (안내 메시지)

**변경 없음**:
- ✅ 스크리닝 로직
- ✅ 기본 스크리닝 탭
- ✅ 종목 검색 탭
- ✅ 결과 테이블

## 테스트 결과

### 수정 전

```
프리셋: 안전형
종목: 2600개
결과: 매수 0개, 매도 0개 ❌
원인: 기본 조건 체크 안 됨
```

### 수정 후 (예상)

```
프리셋: 안전형
종목: 2600개
기본 조건: MA ✓, BB ✓, MACD ✓, 모멘텀 ✓
고급 조건: 다중시간대 ✓, 시장강도 ✓
결과: 매수 5-20개 예상 (안전형은 엄격하므로 적음) ✅
```

```
프리셋: 균형형
종목: 2600개
결과: 매수 20-50개 예상 ✅
```

```
프리셋: 공격형
종목: 2600개
고급 조건: 없음 (기본만)
결과: 매수 50-150개 예상 (기존 스크리닝과 유사) ✅
```

## 사용자 가이드

### 고급 스크리닝 사용법

1. **고급 스크리닝 탭 (🚀)으로 이동**
2. **안내 메시지 확인**
3. **프리셋 선택**:
   - 🛡️ 안전형: 고품질 소수 후보 (승률 우선)
   - ⚖️ 균형형: 중간 (승률과 기회의 균형)
   - 🚀 공격형: 많은 후보 (기회 우선)
   - 사용자 정의: 직접 설정
4. **(선택) 조건 커스터마이징**:
   - 기본 조건: 탭 2에서 확인/수정
   - 고급 조건: 현재 탭에서 체크/해제
   - 상세 설정: 다중 시간대 모드, 시장 지수
5. **🚀 고급 스크리닝 시작 버튼 클릭**
6. **결과 확인**

### 프리셋별 권장 사용 시나리오

**🛡️ 안전형**:
- 장기 투자자
- 안정적인 수익 추구
- 손실 최소화 중요
- 적은 종목 집중 관리 선호

**⚖️ 균형형**:
- 중기 투자자
- 수익과 안정성의 균형
- 적당한 종목 수 관리
- 가장 범용적 (권장)

**🚀 공격형**:
- 단기 투자자
- 높은 수익 추구
- 변동성 감내 가능
- 많은 종목 중 선택 선호

## 결론

프리셋 선택 시 기본 매수 조건을 자동으로 활성화하여 **"매수/매도 0개"** 문제를 해결했습니다.

**변경 사항**:
- ✅ 프리셋 → 기본 조건 자동 체크
- ✅ 안내 메시지 추가
- ✅ 사용자 경험 개선

**다음 단계**:
- 실제 스크리닝 재실행
- 결과 검증
- 성능 측정

---

**작성일**: 2025-10-30
**버전**: Bugfix v1
**관련 이슈**: 고급 스크리닝 0개 결과 문제
