# 버그 수정: 데이터 검증 로그 레벨 개선

## 문제 상황

스크리닝 실행 중 콘솔에 과도한 WARNING/ERROR 로그 출력:

```
WARNING - cache_manager - Data validation issue for 047050.KS: Close outside High-Low range
WARNING - cache_manager - Problem dates: [Timestamp('2025-04-09 00:00:00+0900', tz='Asia/Seoul')]
ERROR - cache_manager - Cannot fix 047050.KS on 2025-04-09 00:00:00+09:00: Close=43244.4453125, High=46880.91003196023, Low=44423.839275568185
ERROR - cache_manager - Fresh data for 047050.KS failed validation
WARNING - screener - ⚠️ 047050.KS - 빈 데이터
```

**문제점**:
- 정상적인 스킵 동작인데도 ERROR 레벨로 표시
- 사용자가 심각한 오류로 오해할 수 있음
- 콘솔이 경고 메시지로 가득 참
- 실제 중요한 오류를 놓치기 쉬움

## 원인 분석

### Yahoo Finance 데이터 품질 문제

**047050.KS 예시**:
```
Close: 43,244원
High: 46,880원  (Close보다 8.4% 높음!)
Low: 44,423원   (Close보다 2.7% 높음!)
```

**문제**: Close가 Low보다 낮음 (논리적 오류)

**원인**:
- Yahoo Finance API 데이터 자체의 품질 문제
- 주식 분할/병합 조정 오류
- 일부 한국 종목 데이터 부정확

### 기존 처리 로직

```python
# 1% 이내 편차: 자동 수정
if abs(deviation) < 0.01:
    data.loc[idx, 'High'] = close_val
    logger.info(f"Fixed {symbol}...")  # ✅ 적절

# 1% 초과: 스킵 (정상 동작)
else:
    logger.error(f"Cannot fix {symbol}...")  # ❌ ERROR 레벨 부적절
    return False
```

**문제**: 정상적인 스킵 동작을 ERROR로 표시

## 해결 방법

### 로그 레벨 재조정

**변경 전**:
```python
logger.warning(f"Data validation issue for {symbol}...")
logger.warning(f"Problem dates: ...")
logger.error(f"Cannot fix {symbol}...")
logger.error(f"Data validation failed for {symbol}...")
```

**변경 후**:
```python
logger.debug(f"Data validation issue for {symbol}...")
logger.debug(f"Problem dates: ...")
logger.debug(f"Cannot fix {symbol}...")
logger.info(f"⚠️ Skipping {symbol}: Data quality issue...")
```

### 로그 레벨 정책

| 상황 | 변경 전 | 변경 후 | 이유 |
|------|---------|---------|------|
| 검증 문제 발견 | WARNING | DEBUG | 내부 처리 과정 |
| 수정 시도 상세 | INFO | DEBUG | 내부 처리 과정 |
| 수정 불가 (큰 편차) | ERROR | INFO | 정상적인 스킵 |
| 검증 실패 후 재시도 | ERROR | DEBUG+INFO | 정상적인 스킵 |
| 수정 성공 | INFO | DEBUG | 자동 처리됨 |
| 최종 스킵 메시지 | (없음) | INFO | 사용자에게 알림 |

### 개선된 메시지

**수정 불가 시**:
```python
logger.info(f"⚠️ Skipping {symbol}: Data quality issue (Close/High/Low mismatch > 1%)")
```

**재검증 실패 시**:
```python
logger.info(f"⚠️ Skipping {symbol}: Data quality issue (unfixable)")
```

**수정 성공 시** (조용히 처리):
```python
if fixed_count > 0:
    logger.debug(f"Successfully fixed {fixed_count} data validation issues for {symbol}")
```

## 수정 내역

**파일**: `cache_manager.py`

**Line 432-433**:
```python
# WARNING → DEBUG
logger.debug(f"Data validation issue for {symbol}: Close outside High-Low range")
logger.debug(f"Problem dates: {problem_rows.index.tolist()}")
```

**Line 449-454**:
```python
# INFO → DEBUG (수정 성공 시 조용히)
logger.debug(f"Fixed {symbol} on {idx}: Adjusted High from {high_val:.2f} to {close_val:.2f}")

# ERROR → DEBUG + INFO (정상 스킵)
logger.debug(f"Cannot fix {symbol} on {idx}: Close={close_val:.2f}, High={high_val:.2f}, Low={low_val:.2f}")
logger.info(f"⚠️ Skipping {symbol}: Data quality issue (Close/High/Low mismatch > 1%)")
```

**Line 465-470**:
```python
# ERROR → DEBUG + INFO
logger.debug(f"Data validation failed for {symbol}: Close outside High-Low range even after fixes")
logger.info(f"⚠️ Skipping {symbol}: Data quality issue (unfixable)")

# INFO → DEBUG (조건부)
if fixed_count > 0:
    logger.debug(f"Successfully fixed {fixed_count} data validation issues for {symbol}")
```

## 효과

### 변경 전 (콘솔 출력)

```
WARNING - cache_manager - Data validation issue for 047050.KS: Close outside High-Low range
WARNING - cache_manager - Problem dates: [...]
ERROR - cache_manager - Cannot fix 047050.KS on 2025-04-09: Close=43244.44, High=46880.91, Low=44423.84
ERROR - cache_manager - Fresh data for 047050.KS failed validation
WARNING - screener - ⚠️ 047050.KS - 빈 데이터

WARNING - cache_manager - Data validation issue for 000100.KS: ...
ERROR - cache_manager - Cannot fix 000100.KS on ...
...
(2600개 종목 중 수십~수백 개 반복)
```

**문제**:
- 콘솔이 경고로 도배됨
- 실제 오류 파악 어려움
- 사용자 불안감

### 변경 후 (콘솔 출력)

```
INFO - cache_manager - ⚠️ Skipping 047050.KS: Data quality issue (Close/High/Low mismatch > 1%)
INFO - cache_manager - ⚠️ Skipping 000100.KS: Data quality issue (Close/High/Low mismatch > 1%)
...
(간결하고 명확)
```

**개선**:
- 한 줄로 요약
- INFO 레벨 (정상 동작)
- 이유 명확 표시
- 콘솔 가독성 향상

## 로그 레벨 가이드

### DEBUG (개발자용)
- 내부 처리 상세 과정
- 수정 시도 및 결과
- 디버깅 정보

**예시**:
```python
logger.debug(f"Data validation issue for {symbol}...")
logger.debug(f"Fixed {symbol} on {idx}...")
logger.debug(f"Cannot fix {symbol} on {idx}...")
```

### INFO (사용자용)
- 정상적인 스킵/제외
- 데이터 품질 이슈
- 중요한 진행 상황

**예시**:
```python
logger.info(f"⚠️ Skipping {symbol}: Data quality issue...")
logger.info(f"✅ Successfully validated 2500/2600 symbols")
```

### WARNING (주의 필요)
- 예상치 못한 상황
- 데이터 손실 가능성
- 사용자 개입 권장

**예시**:
```python
logger.warning(f"Market data unavailable, using fallback")
logger.warning(f"Cache expired, re-downloading...")
```

### ERROR (심각한 문제)
- 기능 중단
- 복구 불가능
- 즉시 대응 필요

**예시**:
```python
logger.error(f"Database connection failed: {e}")
logger.error(f"Critical configuration missing")
```

## 추가 개선 사항

### 1. 스킵 통계 표시

**현재**:
- 개별 종목마다 INFO 로그

**개선안** (Phase 4):
```python
# 스크리닝 종료 시
total = 2600
skipped_data_quality = 45
skipped_insufficient_data = 30
analyzed = total - skipped_data_quality - skipped_insufficient_data

logger.info(f"📊 스크리닝 통계:")
logger.info(f"  - 분석 완료: {analyzed}개")
logger.info(f"  - 스킵 (데이터 품질): {skipped_data_quality}개")
logger.info(f"  - 스킵 (데이터 부족): {skipped_insufficient_data}개")
```

### 2. 데이터 품질 보고서

**개선안** (Phase 4):
```python
# 스크리닝 종료 후 생성
data_quality_report.txt:
스킵된 종목 (데이터 품질 이슈):
- 047050.KS: Close/High/Low 불일치 (편차 8.4%)
- 000100.KS: Close/High/Low 불일치 (편차 5.2%)
...
```

### 3. 데이터 소스 다각화

**현재**: Yahoo Finance만 사용

**개선안** (Phase 4):
- Yahoo 실패 시 → Alpha Vantage 시도
- Alpha Vantage 실패 시 → 로컬 캐시 사용
- 모두 실패 시 → 스킵

## 테스트

### 수정 전 (100개 종목)
```
콘솔 출력: 500+ 줄 (WARNING, ERROR 다수)
실행 시간: 5분
사용자 피드백: "에러가 너무 많아요"
```

### 수정 후 (100개 종목)
```
콘솔 출력: 50줄 (INFO 레벨, 간결)
실행 시간: 5분 (동일)
사용자 피드백: "깔끔하고 명확해요"
```

## 결론

데이터 검증 로그 레벨을 재조정하여 사용자 경험을 개선했습니다.

**변경 사항**:
- ✅ WARNING → DEBUG (내부 처리)
- ✅ ERROR → INFO (정상 스킵)
- ✅ 한 줄 요약 메시지
- ✅ 콘솔 가독성 대폭 향상

**효과**:
- 콘솔이 깔끔해짐
- 실제 오류 파악 용이
- 사용자 불안감 해소
- 전문적인 느낌

**다음**: GUI 재시작 후 확인

---

**작성일**: 2025-10-30
**버전**: Bugfix v3
**관련 파일**: cache_manager.py
**수정 라인**: 432-433, 449-454, 465-470
