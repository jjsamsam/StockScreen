"""
Enhanced Search Functions for Stock Search Integration
prediction_window.py와 screener.py에 통합할 향상된 검색 기능
"""

import urllib.parse
import requests
import pandas as pd
import os
from typing import List, Dict, Any, Optional

def search_stocks_with_api(search_term: str) -> List[Dict[str, Any]]:
    """
    API를 사용한 실시간 주식 검색 + 기존 CSV 백업
    
    Args:
        search_term (str): 검색어 (종목코드 또는 회사명)
        
    Returns:
        List[Dict]: 검색된 종목 정보 리스트 (CSV 포맷 준비)
        
    Example:
        results = search_stocks_with_api("삼성")
        for stock in results:
            print(f"{stock['ticker']}: {stock['name']} - {stock['market_cap']}")
    """
    
    print(f"🔍 API로 '{search_term}' 검색 시작...")
    api_results = []
    csv_results = []
    
    # 1. 먼저 API로 검색 시도
    try:
        query = urllib.parse.quote(search_term)
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        res = requests.get(url, headers=headers, timeout=10)
        print(f"API 응답 상태코드: {res.status_code}")

        if res.ok:
            data = res.json()
            quotes = data.get('quotes', [])
            print(f"📊 API에서 {len(quotes)}개 종목 발견")
            
            # JSON을 CSV 포맷으로 변환
            api_results = convert_api_response_to_csv_format(quotes)
            
        else:
            print(f"API 요청 실패: {res.status_code} - {res.text[:200]}")

    except Exception as e:
        print(f"❌ API 검색 실패: {e}")

    # 2. CSV에서도 검색 (백업용)
    try:
        csv_results = search_master_csv_backup(search_term)
        print(f"📁 CSV에서 {len(csv_results)}개 종목 발견")
    except Exception as e:
        print(f"❌ CSV 검색 실패: {e}")

    # 3. 결과 병합 및 중복 제거
    all_results = merge_and_deduplicate_results(api_results, csv_results)
    
    print(f"✅ 총 {len(all_results)}개 종목 반환 (API: {len(api_results)}, CSV: {len(csv_results)})")
    
    return all_results


def convert_api_response_to_csv_format(quotes: List[Dict]) -> List[Dict[str, Any]]:
    """
    Yahoo Finance API 응답을 CSV 포맷으로 변환
    
    Example API Response:
    {
        "symbol": "AAPL",
        "shortname": "Apple Inc.",
        "longname": "Apple Inc.",
        "exchDisp": "NASDAQ",
        "typeDisp": "Equity",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3000000000000
    }
    """
    csv_format_results = []
    
    for quote in quotes:
        try:
            # 기본 정보 추출
            ticker = quote.get('symbol', '').strip()
            if not ticker:
                continue
                
            # 회사명 추출 (우선순위: longname > shortname)
            name = quote.get('longname') or quote.get('shortname', ticker)
            
            # 섹터/산업 정보
            sector = quote.get('sector', quote.get('industry', '미분류'))
            
            # 시가총액 포맷팅
            market_cap_raw = quote.get('marketCap', 0)
            market_cap_str = format_market_cap(market_cap_raw)
            
            # 거래소 정보
            exchange = quote.get('exchDisp') or quote.get('exchange', 'Unknown')
            
            # CSV 포맷으로 구성
            stock_info = {
                'ticker': ticker,
                'name': name,
                'sector': sector,
                'market_cap': market_cap_str,
                'market': exchange,
                'raw_market_cap': market_cap_raw,
                'match_score': calculate_api_match_score(quote, ticker, name),
                'source': 'API'
            }
            
            csv_format_results.append(stock_info)
            
        except Exception as e:
            print(f"⚠️ API 데이터 변환 오류: {e}")
            continue
    
    return csv_format_results


def search_master_csv_backup(search_term: str) -> List[Dict[str, Any]]:
    """
    기존 마스터 CSV 파일에서 검색 (백업용)
    prediction_window.py와 screener.py의 기존 함수와 동일한 로직
    """
    found_stocks = []
    seen_tickers = set()
    search_term_upper = search_term.strip().upper()
    
    # 마스터 CSV 파일 경로들 (두 위치 모두 확인)
    possible_locations = [
        # 첫 번째 우선순위: master_csv 폴더
        [
            'master_csv/korea_stocks_master.csv',
            'master_csv/usa_stocks_master.csv', 
            'master_csv/sweden_stocks_master.csv'
        ],
        # 두 번째 우선순위: stock_data 폴더
        [
            'stock_data/korea_stocks_master.csv',
            'stock_data/usa_stocks_master.csv', 
            'stock_data/sweden_stocks_master.csv'
        ]
    ]
    
    # 첫 번째로 찾은 위치 사용
    master_files = []
    for location_set in possible_locations:
        if any(os.path.exists(f) for f in location_set):
            master_files = location_set
            break
    
    if not master_files:
        print("⚠️ 마스터 CSV 파일을 찾을 수 없습니다")
        return []
    
    # 각 마스터 CSV 파일에서 검색
    for file_path in master_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            market_name = get_market_name_from_filename(file_path)
            
            for _, row in df.iterrows():
                ticker = str(row.get('ticker', '')).strip()
                name = str(row.get('name', '')).strip()
                sector = str(row.get('sector', '')).strip()
                market_cap = row.get('market_cap', 0)
                
                if not ticker or ticker in seen_tickers:
                    continue
                
                # 매칭 로직
                match_score = calculate_csv_match_score(
                    search_term_upper, ticker, name, sector
                )
                
                if match_score > 0:
                    # 시가총액 포맷팅
                    market_cap_str = format_market_cap(market_cap)
                    
                    stock_info = {
                        'ticker': ticker,
                        'name': name,
                        'sector': sector,
                        'market_cap': market_cap_str,
                        'market': market_name,
                        'raw_market_cap': market_cap if pd.notna(market_cap) else 0,
                        'match_score': match_score,
                        'source': 'CSV'
                    }
                    
                    found_stocks.append(stock_info)
                    seen_tickers.add(ticker)
                    
        except Exception as e:
            print(f"⚠️ {file_path} 읽기 오류: {e}")
            continue
    
    return found_stocks


def merge_and_deduplicate_results(
    api_results: List[Dict], 
    csv_results: List[Dict]
) -> List[Dict[str, Any]]:
    """
    API 결과와 CSV 결과를 병합하고 중복 제거
    API 결과를 우선시하되, CSV에만 있는 결과도 포함
    """
    combined_results = {}
    
    # API 결과 우선 추가
    for stock in api_results:
        ticker = stock['ticker']
        stock['match_score'] += 10  # API 결과에 보너스 점수
        combined_results[ticker] = stock
    
    # CSV 결과 추가 (이미 있는 ticker는 건너뛰기)
    for stock in csv_results:
        ticker = stock['ticker']
        if ticker not in combined_results:
            combined_results[ticker] = stock
    
    # 매치 점수와 시가총액으로 정렬
    sorted_results = sorted(
        combined_results.values(), 
        key=lambda x: (-x['match_score'], -x.get('raw_market_cap', 0))
    )
    
    return sorted_results


# ==================== 헬퍼 함수들 ====================

def format_market_cap(market_cap_value) -> str:
    """시가총액을 사람이 읽기 쉬운 형태로 포맷팅"""
    try:
        if pd.isna(market_cap_value) or market_cap_value == 0:
            return "N/A"
        
        mcap = float(market_cap_value)
        
        if mcap >= 1e12:
            return f"{mcap/1e12:.1f}T"
        elif mcap >= 1e9:
            return f"{mcap/1e9:.1f}B"
        elif mcap >= 1e6:
            return f"{mcap/1e6:.1f}M"
        else:
            return f"{mcap:,.0f}"
            
    except (ValueError, TypeError):
        return "N/A"


def calculate_api_match_score(quote: Dict, ticker: str, name: str) -> int:
    """API 결과의 매치 점수 계산"""
    # API 결과는 이미 검색어와 관련성이 높으므로 기본 점수를 높게 설정
    base_score = 90
    
    # 추가 보너스 (예: 정확한 타입인지 확인)
    if quote.get('typeDisp') == 'Equity':
        base_score += 5
    
    return base_score


def calculate_csv_match_score(search_term_upper: str, ticker: str, name: str, sector: str) -> int:
    """CSV 결과의 매치 점수 계산 (기존 로직과 동일)"""
    ticker_upper = ticker.upper()
    name_upper = name.upper()
    sector_upper = sector.upper()
    
    # 1. 티커 완전 매치 (최고 점수)
    if ticker_upper == search_term_upper:
        return 100
    # 2. 회사명 완전 매치
    elif name_upper == search_term_upper:
        return 95
    # 3. 티커 부분 매치
    elif search_term_upper in ticker_upper:
        return 85
    # 4. 회사명 부분 매치
    elif search_term_upper in name_upper:
        return 75
    # 5. 섹터 매치
    elif search_term_upper in sector_upper:
        return 50
    
    return 0


def get_market_name_from_filename(file_path: str) -> str:
    """파일 경로에서 시장 이름 추출"""
    filename = os.path.basename(file_path).lower()
    
    if 'korea' in filename:
        return 'KRX'
    elif 'usa' in filename:
        return 'NYSE/NASDAQ'
    elif 'sweden' in filename:
        return 'OMX Stockholm'
    else:
        return 'Unknown'


def display_search_results_as_csv(results: List[Dict[str, Any]]) -> str:
    """
    검색 결과를 CSV 형태의 문자열로 포맷팅하여 반환
    
    Returns:
        str: CSV 포맷의 검색 결과 문자열
        
    Example:
        results = search_stocks_with_api("삼성")
        csv_string = display_search_results_as_csv(results)
        print(csv_string)
    """
    if not results:
        return "ticker,name,sector,market_cap,market,source\n(검색 결과가 없습니다)"
    
    # CSV 헤더
    csv_lines = ["ticker,name,sector,market_cap,market,source"]
    
    # 데이터 행들
    for stock in results:
        # CSV 포맷에 맞게 데이터 정리 (쉼표나 특수문자가 있으면 따옴표로 감싸기)
        ticker = clean_csv_value(stock.get('ticker', ''))
        name = clean_csv_value(stock.get('name', ''))
        sector = clean_csv_value(stock.get('sector', ''))
        market_cap = clean_csv_value(stock.get('market_cap', 'N/A'))
        market = clean_csv_value(stock.get('market', ''))
        source = clean_csv_value(stock.get('source', ''))
        
        csv_line = f"{ticker},{name},{sector},{market_cap},{market},{source}"
        csv_lines.append(csv_line)
    
    return "\n".join(csv_lines)


def clean_csv_value(value: str) -> str:
    """CSV 값에서 특수문자 처리"""
    if not isinstance(value, str):
        value = str(value)
    
    # 쉼표나 따옴표가 있으면 따옴표로 감싸고 내부 따옴표는 이스케이프
    if ',' in value or '"' in value or '\n' in value:
        value = value.replace('"', '""')  # 따옴표 이스케이프
        return f'"{value}"'
    
    return value


# ==================== 통합을 위한 함수들 ====================

def integrate_enhanced_search_into_prediction_window():
    """
    prediction_window.py에 통합할 때 사용할 함수
    기존 search_master_csv 함수를 대체하거나 보완
    """
    example_code = '''
# prediction_window.py의 StockSearchDialog 클래스에 추가할 메서드

def search_stocks_enhanced(self):
    """향상된 종목 검색 (API + CSV)"""
    query = self.search_input.text().strip()
    if len(query) < 1:
        self.show_popular_stocks()
        return
    
    try:
        self.status_label.setText(f"'{query}' 검색 중... (API + CSV)")
        QApplication.processEvents()
        
        # 향상된 검색 함수 사용
        from enhanced_search_functions import search_stocks_with_api
        results = search_stocks_with_api(query)
        
        self.display_results(results)
        
        if results:
            api_count = len([r for r in results if r.get('source') == 'API'])
            csv_count = len([r for r in results if r.get('source') == 'CSV'])
            self.status_label.setText(
                f"🔍 {len(results)}개 종목 발견 (API: {api_count}, CSV: {csv_count})"
            )
        else:
            self.status_label.setText("❌ 검색 결과가 없습니다")
            
    except Exception as e:
        self.status_label.setText(f"❌ 검색 오류: {str(e)}")
        print(f"검색 오류: {e}")

# 기존 search_btn.clicked.connect를 다음과 같이 변경:
# self.search_btn.clicked.connect(self.search_stocks_enhanced)
'''
    return example_code


def integrate_enhanced_search_into_screener():
    """
    screener.py에 통합할 때 사용할 함수
    기존 search_master_csv 함수를 대체하거나 보완
    """
    example_code = '''
# screener.py의 StockScreener 클래스에 추가할 메서드

def search_and_show_chart_enhanced(self):
    """향상된 검색으로 종목을 찾아서 차트 표시"""
    query = self.search_input.text().strip()
    if not query:
        QMessageBox.warning(self, "검색어 필요", "검색할 종목코드나 회사명을 입력해주세요.")
        return
    
    try:
        self.search_result_label.setText("검색 중...")
        QApplication.processEvents()
        
        # 향상된 검색 함수 사용
        from enhanced_search_functions import search_stocks_with_api, display_search_results_as_csv
        results = search_stocks_with_api(query)
        
        if results:
            # 검색 결과를 CSV 포맷으로 표시
            csv_results = display_search_results_as_csv(results)
            
            # 첫 번째 결과로 차트 표시
            first_result = results[0]
            ticker = first_result['ticker']
            
            self.search_result_label.setText(
                f"✅ {len(results)}개 발견 (API+CSV)"
            )
            
            # CSV 결과를 다이얼로그로 표시
            dialog = QDialog(self)
            dialog.setWindowTitle(f"검색 결과: {query}")
            dialog.resize(800, 400)
            
            layout = QVBoxLayout()
            
            # 결과 정보
            info_label = QLabel(f"총 {len(results)}개 종목 발견")
            layout.addWidget(info_label)
            
            # CSV 형태로 결과 표시
            text_edit = QTextEdit()
            text_edit.setPlainText(csv_results)
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)
            
            # 선택된 종목으로 차트 보기 버튼
            chart_btn = QPushButton(f"{ticker} 차트 보기")
            chart_btn.clicked.connect(lambda: self.show_stock_detail(ticker))
            layout.addWidget(chart_btn)
            
            dialog.setLayout(layout)
            dialog.show()
            
        else:
            self.search_result_label.setText("❌ 결과 없음")
            QMessageBox.information(self, "검색 결과", f"'{query}'에 대한 검색 결과가 없습니다.")
            
    except Exception as e:
        self.search_result_label.setText(f"❌ 오류: {str(e)}")
        QMessageBox.critical(self, "검색 오류", f"검색 중 오류가 발생했습니다:\\n{str(e)}")

# 기존 search_btn.clicked.connect를 다음과 같이 변경:
# self.search_btn.clicked.connect(self.search_and_show_chart_enhanced)
'''
    return example_code


if __name__ == "__main__":
    # 테스트 코드
    print("🧪 Enhanced Search Functions 테스트")
    
    # 예시 1: 삼성 검색
    print("\n=== 삼성 검색 테스트 ===")
    samsung_results = search_stocks_with_api("삼성")
    csv_output = display_search_results_as_csv(samsung_results)
    print(csv_output)
    
    # 예시 2: Apple 검색
    print("\n=== Apple 검색 테스트 ===")
    apple_results = search_stocks_with_api("AAPL")
    csv_output = display_search_results_as_csv(apple_results)
    print(csv_output)
    
    print("\n✅ 테스트 완료")
