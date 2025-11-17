"""
sector_mapping.py
섹터 분류 및 매핑 데이터
"""

# 미국 주요 섹터 ETF
SECTOR_ETFS = {
    'Technology': 'XLK',           # Technology Select Sector SPDR Fund
    'Healthcare': 'XLV',           # Health Care Select Sector SPDR Fund
    'Financials': 'XLF',           # Financial Select Sector SPDR Fund
    'Consumer Discretionary': 'XLY',  # Consumer Discretionary Select Sector SPDR Fund
    'Consumer Staples': 'XLP',     # Consumer Staples Select Sector SPDR Fund
    'Energy': 'XLE',               # Energy Select Sector SPDR Fund
    'Industrials': 'XLI',          # Industrial Select Sector SPDR Fund
    'Materials': 'XLB',            # Materials Select Sector SPDR Fund
    'Real Estate': 'XLRE',         # Real Estate Select Sector SPDR Fund
    'Communication Services': 'XLC',  # Communication Services Select Sector SPDR Fund
    'Utilities': 'XLU',            # Utilities Select Sector SPDR Fund
}

# 주요 종목별 섹터 매핑 (예시)
STOCK_SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology',
    'MSFT': 'Technology',
    'GOOGL': 'Technology',
    'GOOG': 'Technology',
    'NVDA': 'Technology',
    'META': 'Technology',
    'TSLA': 'Technology',
    'AVGO': 'Technology',
    'ADBE': 'Technology',
    'CSCO': 'Technology',
    'CRM': 'Technology',
    'ORCL': 'Technology',
    'INTC': 'Technology',
    'AMD': 'Technology',
    'QCOM': 'Technology',

    # Healthcare
    'UNH': 'Healthcare',
    'JNJ': 'Healthcare',
    'LLY': 'Healthcare',
    'ABBV': 'Healthcare',
    'MRK': 'Healthcare',
    'PFE': 'Healthcare',
    'TMO': 'Healthcare',
    'ABT': 'Healthcare',
    'DHR': 'Healthcare',
    'BMY': 'Healthcare',

    # Financials
    'BRK.B': 'Financials',
    'JPM': 'Financials',
    'V': 'Financials',
    'MA': 'Financials',
    'BAC': 'Financials',
    'WFC': 'Financials',
    'MS': 'Financials',
    'GS': 'Financials',
    'SPGI': 'Financials',
    'BLK': 'Financials',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary',
    'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary',
    'TJX': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples',
    'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples',
    'WMT': 'Consumer Staples',
    'PM': 'Consumer Staples',
    'MO': 'Consumer Staples',

    # Energy
    'XOM': 'Energy',
    'CVX': 'Energy',
    'COP': 'Energy',
    'SLB': 'Energy',
    'EOG': 'Energy',
    'PXD': 'Energy',

    # Industrials
    'BA': 'Industrials',
    'HON': 'Industrials',
    'UNP': 'Industrials',
    'CAT': 'Industrials',
    'GE': 'Industrials',
    'MMM': 'Industrials',
    'LMT': 'Industrials',

    # Materials
    'LIN': 'Materials',
    'APD': 'Materials',
    'SHW': 'Materials',
    'FCX': 'Materials',
    'NEM': 'Materials',

    # Real Estate
    'AMT': 'Real Estate',
    'PLD': 'Real Estate',
    'CCI': 'Real Estate',
    'EQIX': 'Real Estate',
    'PSA': 'Real Estate',

    # Communication Services
    'NFLX': 'Communication Services',
    'DIS': 'Communication Services',
    'CMCSA': 'Communication Services',
    'T': 'Communication Services',
    'VZ': 'Communication Services',

    # Utilities
    'NEE': 'Utilities',
    'DUK': 'Utilities',
    'SO': 'Utilities',
    'D': 'Utilities',
    'AEP': 'Utilities',
}

# 한국 섹터 (임시 - ETF 대신 대표 종목 사용)
KOREA_SECTOR_STOCKS = {
    'Technology': ['005930.KS', '000660.KS', '035420.KS'],  # 삼성전자, SK하이닉스, NAVER
    'Automotive': ['005380.KS', '000270.KS'],  # 현대차, 기아
    'Chemical': ['051910.KS', '009830.KS'],  # LG화학, 한화솔루션
    'Finance': ['055550.KS', '086790.KS'],  # 신한지주, 하나금융지주
    'Energy': ['006400.KS', '010950.KS'],  # 삼성SDI, S-Oil
}

def get_sector(symbol):
    """
    종목의 섹터를 반환

    Args:
        symbol: 종목 심볼 (예: 'AAPL', '005930.KS')

    Returns:
        섹터 이름 또는 'Unknown'
    """
    # 미국 종목
    if symbol in STOCK_SECTOR_MAP:
        return STOCK_SECTOR_MAP[symbol]

    # 한국 종목 (추가 로직 필요)
    if '.KS' in symbol:
        for sector, stocks in KOREA_SECTOR_STOCKS.items():
            if symbol in stocks:
                return sector
        return 'Unknown'

    return 'Unknown'

def get_sector_etf(sector):
    """
    섹터에 해당하는 ETF 심볼 반환

    Args:
        sector: 섹터 이름

    Returns:
        ETF 심볼 또는 None
    """
    return SECTOR_ETFS.get(sector)

def get_sector_peers(symbol, max_count=10):
    """
    같은 섹터의 동료 종목 리스트 반환

    Args:
        symbol: 종목 심볼
        max_count: 최대 반환 개수

    Returns:
        동료 종목 리스트 (ETF 또는 종목들)
    """
    sector = get_sector(symbol)

    if sector == 'Unknown':
        return []

    # 미국 종목: 섹터 ETF 사용
    etf = get_sector_etf(sector)
    if etf:
        return [etf]

    # 한국 종목: 대표 종목들 사용
    if '.KS' in symbol:
        peers = KOREA_SECTOR_STOCKS.get(sector, [])
        # 자신 제외
        peers = [s for s in peers if s != symbol]
        return peers[:max_count]

    # 동일 섹터의 다른 종목들
    peers = [s for s, sec in STOCK_SECTOR_MAP.items() if sec == sector and s != symbol]
    return peers[:max_count]

def add_stock_sector(symbol, sector):
    """
    종목의 섹터를 매핑에 추가 (런타임)

    Args:
        symbol: 종목 심볼
        sector: 섹터 이름
    """
    STOCK_SECTOR_MAP[symbol] = sector

# 테스트 코드
if __name__ == "__main__":
    print("=== 섹터 매핑 테스트 ===\n")

    # 테스트 종목들
    test_symbols = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'XOM', '005930.KS', 'UNKNOWN']

    for symbol in test_symbols:
        sector = get_sector(symbol)
        etf = get_sector_etf(sector)
        peers = get_sector_peers(symbol)

        print(f"{symbol}:")
        print(f"  Sector: {sector}")
        print(f"  ETF: {etf}")
        print(f"  Peers: {peers[:3]}")  # 처음 3개만
        print()

    print("✅ 테스트 완료")
