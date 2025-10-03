"""
Vectorized Operations Guide and Utilities
DataFrame.iterrows() 최적화 - 벡터화 연산으로 10-50배 성능 향상
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable


# ============================================================================
# 1. iterrows() vs 벡터화 성능 비교
# ============================================================================

def example_slow_iterrows(df: pd.DataFrame) -> pd.Series:
    """❌ SLOW: iterrows() 사용 (비추천)"""
    results = []
    for idx, row in df.iterrows():
        # 각 행마다 개별 연산
        result = row['price'] * row['quantity'] * 1.1  # 10% 할인
        results.append(result)
    return pd.Series(results, index=df.index)


def example_fast_vectorized(df: pd.DataFrame) -> pd.Series:
    """✅ FAST: 벡터화 연산 (추천) - 10-50배 빠름"""
    # 전체 컬럼에 대해 한 번에 연산
    return df['price'] * df['quantity'] * 1.1


# ============================================================================
# 2. 검색 최적화: iterrows() → 벡터화
# ============================================================================

def search_stocks_slow(df: pd.DataFrame, search_term: str) -> List[Dict]:
    """❌ SLOW: iterrows()로 검색"""
    results = []
    search_upper = search_term.upper()

    for _, row in df.iterrows():
        ticker = str(row.get('ticker', '')).upper()
        name = str(row.get('name', '')).upper()

        if search_upper in ticker or search_upper in name:
            results.append({
                'ticker': row['ticker'],
                'name': row['name'],
                'sector': row.get('sector', ''),
            })

    return results


def search_stocks_fast(df: pd.DataFrame, search_term: str) -> List[Dict]:
    """✅ FAST: 벡터화 검색 - 10-20배 빠름"""
    search_upper = search_term.upper()

    # 벡터화된 문자열 검색
    ticker_match = df['ticker'].astype(str).str.upper().str.contains(search_upper, na=False)
    name_match = df['name'].astype(str).str.upper().str.contains(search_upper, na=False)

    # 조건을 만족하는 행만 필터링
    matched_df = df[ticker_match | name_match]

    # to_dict('records')로 한 번에 변환
    return matched_df[['ticker', 'name', 'sector']].to_dict('records')


# ============================================================================
# 3. 조건부 연산 최적화
# ============================================================================

def calculate_signals_slow(df: pd.DataFrame) -> pd.Series:
    """❌ SLOW: iterrows()로 조건 체크"""
    signals = []

    for _, row in df.iterrows():
        if row['rsi'] < 30 and row['price'] < row['ma_50']:
            signal = 'BUY'
        elif row['rsi'] > 70 and row['price'] > row['ma_50']:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        signals.append(signal)

    return pd.Series(signals, index=df.index)


def calculate_signals_fast(df: pd.DataFrame) -> pd.Series:
    """✅ FAST: 벡터화 조건 - 20-30배 빠름"""
    # numpy.where 또는 pandas 조건 활용
    signals = pd.Series('HOLD', index=df.index)  # 기본값

    # 조건을 벡터로 체크
    buy_condition = (df['rsi'] < 30) & (df['price'] < df['ma_50'])
    sell_condition = (df['rsi'] > 70) & (df['price'] > df['ma_50'])

    signals[buy_condition] = 'BUY'
    signals[sell_condition] = 'SELL'

    return signals


# ============================================================================
# 4. 복잡한 계산 최적화
# ============================================================================

def calculate_returns_slow(df: pd.DataFrame) -> pd.DataFrame:
    """❌ SLOW: iterrows()로 수익률 계산"""
    returns = []

    for i, row in df.iterrows():
        if i == 0:
            returns.append(0)
        else:
            prev_price = df.iloc[i - 1]['Close']
            current_price = row['Close']
            ret = (current_price - prev_price) / prev_price * 100
            returns.append(ret)

    df['returns'] = returns
    return df


def calculate_returns_fast(df: pd.DataFrame) -> pd.DataFrame:
    """✅ FAST: 벡터화 수익률 계산 - 50배 빠름"""
    # pct_change()를 사용한 한 줄 계산
    df['returns'] = df['Close'].pct_change() * 100
    return df


# ============================================================================
# 5. 누적 계산 최적화
# ============================================================================

def calculate_cumulative_slow(df: pd.DataFrame) -> pd.Series:
    """❌ SLOW: iterrows()로 누적합"""
    cumsum = []
    total = 0

    for _, row in df.iterrows():
        total += row['value']
        cumsum.append(total)

    return pd.Series(cumsum, index=df.index)


def calculate_cumulative_fast(df: pd.DataFrame) -> pd.Series:
    """✅ FAST: 벡터화 누적합 - 100배 빠름"""
    return df['value'].cumsum()


# ============================================================================
# 6. 그룹별 연산 최적화
# ============================================================================

def group_analysis_slow(df: pd.DataFrame) -> Dict[str, float]:
    """❌ SLOW: iterrows()로 그룹 분석"""
    sector_totals = {}

    for _, row in df.iterrows():
        sector = row['sector']
        value = row['market_cap']

        if sector not in sector_totals:
            sector_totals[sector] = 0
        sector_totals[sector] += value

    return sector_totals


def group_analysis_fast(df: pd.DataFrame) -> Dict[str, float]:
    """✅ FAST: groupby 사용 - 30-40배 빠름"""
    return df.groupby('sector')['market_cap'].sum().to_dict()


# ============================================================================
# 7. 실전 예제: 주식 스크리닝 최적화
# ============================================================================

class OptimizedScreener:
    """최적화된 주식 스크리너"""

    @staticmethod
    def screen_stocks_slow(df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
        """❌ SLOW: iterrows() 사용"""
        filtered_stocks = []

        for _, row in df.iterrows():
            # 여러 조건 체크
            if (row['market_cap'] > criteria.get('min_market_cap', 0) and
                row['pe_ratio'] < criteria.get('max_pe', 100) and
                row['dividend_yield'] > criteria.get('min_dividend', 0)):

                filtered_stocks.append(row.to_dict())

        return pd.DataFrame(filtered_stocks)

    @staticmethod
    def screen_stocks_fast(df: pd.DataFrame, criteria: Dict) -> pd.DataFrame:
        """✅ FAST: 벡터화 필터링 - 15-25배 빠름"""
        # 모든 조건을 벡터로 체크
        mask = (
            (df['market_cap'] > criteria.get('min_market_cap', 0)) &
            (df['pe_ratio'] < criteria.get('max_pe', 100)) &
            (df['dividend_yield'] > criteria.get('min_dividend', 0))
        )

        return df[mask].copy()


# ============================================================================
# 8. apply() vs 벡터화 (apply()도 느림!)
# ============================================================================

def using_apply_slow(df: pd.DataFrame) -> pd.Series:
    """⚠️ SLOW: apply()도 느림 (iterrows()보다는 빠름)"""
    return df.apply(lambda row: row['price'] * row['quantity'], axis=1)


def using_vectorized_fast(df: pd.DataFrame) -> pd.Series:
    """✅ FAST: 순수 벡터화가 가장 빠름"""
    return df['price'] * df['quantity']


# ============================================================================
# 9. 성능 측정 유틸리티
# ============================================================================

def benchmark_performance():
    """벡터화 vs iterrows() 성능 비교"""
    import time

    # 테스트 데이터 생성
    df = pd.DataFrame({
        'ticker': [f'STOCK{i}' for i in range(10000)],
        'name': [f'Company {i}' for i in range(10000)],
        'price': np.random.uniform(10, 1000, 10000),
        'quantity': np.random.randint(1, 1000, 10000),
        'sector': np.random.choice(['Tech', 'Finance', 'Healthcare'], 10000),
        'market_cap': np.random.uniform(1e6, 1e12, 10000),
    })

    print("=" * 60)
    print("성능 비교: iterrows() vs 벡터화")
    print("=" * 60)

    # 1. 곱셈 연산
    start = time.time()
    result_slow = example_slow_iterrows(df)
    time_slow = time.time() - start

    start = time.time()
    result_fast = example_fast_vectorized(df)
    time_fast = time.time() - start

    print(f"\n1. 곱셈 연산 (10,000 rows)")
    print(f"   iterrows(): {time_slow:.4f}s")
    print(f"   벡터화:     {time_fast:.4f}s")
    print(f"   속도 향상:  {time_slow/time_fast:.1f}x")

    # 2. 검색
    start = time.time()
    result_slow = search_stocks_slow(df, 'STOCK1')
    time_slow = time.time() - start

    start = time.time()
    result_fast = search_stocks_fast(df, 'STOCK1')
    time_fast = time.time() - start

    print(f"\n2. 검색 (10,000 rows)")
    print(f"   iterrows(): {time_slow:.4f}s")
    print(f"   벡터화:     {time_fast:.4f}s")
    print(f"   속도 향상:  {time_slow/time_fast:.1f}x")

    # 3. 그룹 분석
    start = time.time()
    result_slow = group_analysis_slow(df)
    time_slow = time.time() - start

    start = time.time()
    result_fast = group_analysis_fast(df)
    time_fast = time.time() - start

    print(f"\n3. 그룹 분석 (10,000 rows)")
    print(f"   iterrows(): {time_slow:.4f}s")
    print(f"   벡터화:     {time_fast:.4f}s")
    print(f"   속도 향상:  {time_slow/time_fast:.1f}x")
    print("=" * 60)


# ============================================================================
# 10. 최적화 체크리스트
# ============================================================================

"""
DataFrame 최적화 체크리스트:

□ iterrows() 제거하고 벡터화 연산 사용
□ apply(axis=1) 최소화 (가능하면 벡터화로 대체)
□ 조건문은 boolean indexing 사용
□ 누적 연산은 cumsum(), cumprod() 등 내장 함수 사용
□ 그룹 연산은 groupby() 사용
□ 문자열 검색은 .str.contains() 사용
□ to_dict('records')로 리스트 변환 최적화
□ 큰 DataFrame은 청크 단위로 처리
"""


if __name__ == "__main__":
    # 성능 비교 실행
    benchmark_performance()
