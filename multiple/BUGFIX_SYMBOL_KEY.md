# 버그 수정: 'symbol' KeyError 해결

## 문제 상황

Phase 2/3 고급 스크리닝 실행 중 다음 에러 발생:

```python
ERROR - screener - 고급 스크리닝 오류: 'symbol'
Traceback (most recent call last):
  File "C:\StockScreen\multiple\screener.py", line 1542, in run_advanced_screening
    self.update_buy_table(buy_candidates)
  File "C:\StockScreen\multiple\screener.py", line 3260, in update_buy_table
    symbol_item = QTableWidgetItem(candidate['symbol'])
KeyError: 'symbol'
```

## 원인 분석

### 불일치 발견

**`analyze_stock_advanced()` 반환값** (line 1739, 1763):
```python
result = {
    'ticker': symbol,  # ❌ 'ticker' 사용
    'name': stock_info.get('name', symbol),
    'market': market,
    'action': 'BUY',
    ...
}
```

**`update_buy_table()` 접근** (line 3260):
```python
symbol_item = QTableWidgetItem(candidate['symbol'])  # ❌ 'symbol' 접근
```

**결과**: KeyError 발생

### 왜 발생했나?

1. **기존 `analyze_stock()` 메서드**는 `'symbol'` 키 사용
2. **새로 만든 `analyze_stock_advanced()`**는 `'ticker'` 키 사용
3. **테이블 업데이트 함수**는 `'symbol'` 키 기대
4. **키 불일치** → KeyError

## 해결 방법

### 수정 내용

**파일**: `screener.py`

**Line 1740** (매수 결과):
```python
# 수정 전
result = {
    'ticker': symbol,  # ❌
    ...
}

# 수정 후
result = {
    'symbol': symbol,  # ✅
    ...
}
```

**Line 1764** (매도 결과):
```python
# 수정 전
result = {
    'ticker': symbol,  # ❌
    ...
}

# 수정 후
result = {
    'symbol': symbol,  # ✅
    ...
}
```

### 대안 방법 (선택하지 않음)

**Option 1**: 테이블 함수 수정
```python
# update_buy_table()에서
symbol_item = QTableWidgetItem(candidate.get('ticker') or candidate.get('symbol'))
```
- 장점: 하위 호환성 유지
- 단점: 복잡도 증가, 일관성 부족

**Option 2**: 키 통일 ('ticker'로)
```python
# 모든 곳에서 'ticker' 사용
```
- 장점: 명확한 명명
- 단점: 기존 코드 대량 수정 필요

**선택한 방법**: 기존 코드 기준 (`'symbol'`)에 맞춤
- 장점: 최소 변경, 일관성 유지
- 단점: None

## 영향 범위

### 수정된 부분
- ✅ `analyze_stock_advanced()` - 매수 결과 (line 1740)
- ✅ `analyze_stock_advanced()` - 매도 결과 (line 1764)

### 영향받지 않는 부분
- ✅ `analyze_stock()` - 기존 기본 스크리닝
- ✅ `update_buy_table()` - 변경 없음
- ✅ `update_sell_table()` - 변경 없음
- ✅ 엑셀 저장 기능 - 변경 없음

## 테스트

### 수정 전
```python
# 실행
run_advanced_screening()

# 결과
KeyError: 'symbol'
```

### 수정 후 (예상)
```python
# 실행
run_advanced_screening()

# 결과
✅ 고급 스크리닝 완료 - 매수후보: 5개, 매도후보: 2개
테이블에 정상 표시
```

## 추가 개선 사항

### 일관성 검증

전체 코드베이스에서 종목 심볼 키 사용 현황:

```python
# 검색 결과 확인 필요
grep -r "'ticker'" screener.py
grep -r "'symbol'" screener.py
```

**권장**: 향후 리팩토링 시 하나로 통일 고려

### 타입 힌팅 추가 (향후)

```python
from typing import Dict, Any

def analyze_stock_advanced(...) -> Dict[str, Any]:
    """
    Returns:
        Dict with keys: 'symbol', 'name', 'market', 'action', ...
    """
    result = {
        'symbol': symbol,  # Type: str
        'name': stock_info.get('name', symbol),  # Type: str
        ...
    }
    return result
```

### 단위 테스트 추가 (향후)

```python
def test_analyze_stock_advanced():
    result = screener.analyze_stock_advanced(stock_info, ...)

    # 키 존재 확인
    assert 'symbol' in result
    assert 'name' in result
    assert 'action' in result

    # 타입 확인
    assert isinstance(result['symbol'], str)
    assert result['action'] in ['BUY', 'SELL']
```

## 교훈

### 1. 일관성의 중요성

- 동일한 개념에 대해 일관된 키/변수명 사용
- 'ticker' vs 'symbol' - 하나로 통일 필요

### 2. 코드 리뷰

- 새 함수 추가 시 기존 코드 패턴 확인
- 인터페이스 일치 여부 검증

### 3. 통합 테스트

- 종단간(end-to-end) 테스트 중요
- 실제 데이터로 전체 흐름 검증

## 결론

`'ticker'` → `'symbol'`로 키를 수정하여 KeyError를 해결했습니다.

**변경 사항**:
- ✅ 2줄 수정 (line 1740, 1764)
- ✅ 최소 변경으로 문제 해결
- ✅ 기존 코드와 일관성 유지

**다음**: GUI 재시작 후 재테스트

---

**작성일**: 2025-10-30
**버전**: Bugfix v2
**관련 파일**: screener.py
**수정 라인**: 1740, 1764
