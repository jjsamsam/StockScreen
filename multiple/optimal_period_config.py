#!/usr/bin/env python3
"""
optimal_period_config.py
종목 특성별 최적 훈련 기간 자동 설정
"""

def get_optimal_training_period(symbol: str, forecast_days: int = 5) -> str:
    """
    종목과 예측 기간에 따라 최적 훈련 기간 자동 결정

    Args:
        symbol: 주식 티커
        forecast_days: 예측 일수

    Returns:
        period: yfinance 형식 기간 ("1y", "2y", "3y", etc.)
    """

    # 1. 한국 주식
    if symbol.endswith('.KS') or symbol.endswith('.KQ'):
        return "3y"  # 한국: 3년 권장

    # 2. 테크 주식 (빠른 변화)
    tech_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA',
        'TSLA', 'AMD', 'NFLX', 'INTC', 'CRM', 'ADBE', 'ORCL'
    ]
    if symbol in tech_stocks:
        return "3y"  # 테크: 3년 (최신 중시)

    # 3. 금융 주식 (주기성 강함)
    financial_stocks = [
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW',
        'USB', 'PNC', 'TFC', 'COF'
    ]
    if symbol in financial_stocks:
        return "5y"  # 금융: 5년 (경기 사이클)

    # 4. 에너지/원자재 (강한 주기성)
    energy_stocks = [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX',
        'VLO', 'OXY'
    ]
    if symbol in energy_stocks:
        return "5y"  # 에너지: 5년

    # 5. 바이오/제약 (높은 변동성)
    biotech_symbols = ['MRNA', 'BNTX', 'REGN', 'VRTX', 'ILMN', 'BIIB']
    if symbol in biotech_symbols:
        return "2y"  # 바이오: 2년 (최신 중시)

    # 6. 암호화폐 관련 (매우 높은 변동성)
    crypto_related = ['COIN', 'MSTR', 'RIOT', 'MARA']
    if symbol in crypto_related:
        return "1y"  # 암호화폐: 1년 (과거 무의미)

    # 7. 예측 기간 기반
    if forecast_days <= 5:
        return "2y"  # 단기 예측: 2년
    elif forecast_days <= 30:
        return "3y"  # 중기 예측: 3년
    else:
        return "5y"  # 장기 예측: 5년

    # 8. 기본값
    return "3y"  # ⭐ 일반적으로 3년이 최적


def get_training_period_by_volatility(symbol: str, default_period: str = "3y") -> str:
    """
    종목의 변동성을 분석하여 최적 기간 결정

    높은 변동성 → 짧은 기간 (최신 데이터 중시)
    낮은 변동성 → 긴 기간 (안정적 패턴)

    Args:
        symbol: 주식 티커
        default_period: 기본 기간

    Returns:
        period: 최적 훈련 기간
    """
    try:
        import yfinance as yf
        import numpy as np

        # 최근 6개월 데이터로 변동성 계산
        data = yf.download(symbol, period="6mo", progress=False)

        if data is None or len(data) < 30:
            return default_period

        # 일일 수익률의 표준편차 (변동성)
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 연간 변동성

        # 변동성 기반 기간 결정
        if volatility > 0.5:  # 50% 이상 (매우 높음)
            return "1y"
        elif volatility > 0.35:  # 35-50% (높음)
            return "2y"
        elif volatility > 0.25:  # 25-35% (중간)
            return "3y"
        else:  # 25% 미만 (낮음)
            return "5y"

    except Exception as e:
        # 오류 시 기본값 반환
        return default_period


# ========== 사용 예제 ==========

if __name__ == "__main__":

    # 테스트
    test_symbols = [
        'AAPL',      # 테크
        'JPM',       # 금융
        'XOM',       # 에너지
        'COIN',      # 암호화폐
        '005930.KS', # 삼성전자
    ]

    print("=== 종목별 최적 훈련 기간 ===\n")

    for symbol in test_symbols:
        period = get_optimal_training_period(symbol)
        print(f"{symbol:12s} → {period} (최적)")

    print("\n=== 변동성 기반 훈련 기간 ===\n")

    for symbol in ['AAPL', 'JPM', 'TSLA']:
        period = get_training_period_by_volatility(symbol)
        print(f"{symbol:12s} → {period} (변동성 기반)")
