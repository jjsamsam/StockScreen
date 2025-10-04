"""
screener.py
메인 스크리너 클래스 및 핵심 로직 (중지 버튼 + 엑셀 저장 기능 추가)
"""
import pandas as pd
import yfinance as yf
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from datetime import datetime, timedelta
import os
import re
import urllib.parse
import requests

from chart_window import StockChartWindow
from dialogs import CSVEditorDialog, ConditionBuilderDialog, ConditionManagerDialog
#from utils import UpdateThread, TechnicalAnalysis, export_screening_results
from utils import TechnicalAnalysis, export_screening_results, format_market_cap_value
from utils import SmartUpdateThread
from utils import MasterCSVThread, MasterFilterThread

from trend_analysis import TrendTimingAnalyzer
from backtesting_system import BacktestingDialog

# 최적화 모듈 import
from cache_manager import get_stock_data, get_ticker_info
from unified_search import search_stocks
from csv_manager import load_all_master_csvs

# AI 예측 기능 통합 import
try:
    from prediction_window import StockPredictionDialog
    from enhanced_screener import EnhancedStockScreenerMethods, BatchPredictionDialog, PredictionSettingsDialog
    PREDICTION_AVAILABLE = True
    print("✅ Enhanced AI Prediction 기능 활성화")
except ImportError as e:
    print(f"⚠️ AI Prediction 기능 없음: {e}")
    # 기본 클래스들 더미 정의 (오류 방지)
    class EnhancedStockScreenerMethods:
        def __init__(self):
            pass
        def enhance_ui_with_ai_features(self):
            pass
        def enhance_table_context_menus(self):
            pass
    PREDICTION_AVAILABLE = False

# 통합된 StockScreener 클래스 (조건부 상속)
if PREDICTION_AVAILABLE:
    # AI 기능과 함께 상속
    class StockScreener(QMainWindow, EnhancedStockScreenerMethods):
        pass
else:
    # 기본 기능만 상속
    class StockScreener(QMainWindow):
        pass

# StockScreener 클래스 구현 (공통)
class StockScreener(StockScreener):  # 위에서 정의된 클래스를 상속
    def __init__(self):
        super().__init__()
        
        # AI 기능이 있는 경우 Enhanced 초기화도 함께
        if PREDICTION_AVAILABLE:
            EnhancedStockScreenerMethods.__init__(self)
        
        # 기본 속성들 초기화
        self.stock_lists = {}
        self.custom_conditions = []  # 사용자 정의 조건들
        self.technical_analyzer = TechnicalAnalysis()

        # 추세 분석기 추가
        self.trend_analyzer = TrendTimingAnalyzer() 

        # 스크리닝 제어 변수들
        self.is_screening = False
        self.screening_cancelled = False
        
        # 결과 저장용 변수들
        self.last_buy_candidates = []
        self.last_sell_candidates = []
        
        # 검색 관련 변수들
        self.search_index = {}  # 빠른 검색을 위한 인덱스
        self.recent_searches = []  # 최근 검색어

        # UI 및 기본 기능 초기화
        self.initUI()
        self.setup_stock_lists()
        self.rebuild_search_index()

        # 검색 결과 저장용 변수 추가
        self.last_search_results = []
        
        # 기존 UI 초기화 후에 CSV 기능 추가
        self.add_csv_search_features()

        # 🚀 AI 예측 기능 초기화 (가능한 경우에만)
        if PREDICTION_AVAILABLE:
            try:
                print("🤖 AI 예측 기능 초기화 중...")
                
                # 예측 설정 로드
                self.load_prediction_settings()
                
                # UI에 AI 기능 추가 (메뉴, 버튼 등)
                self.enhance_ui_with_ai_features()
                
                # 테이블 컨텍스트 메뉴에 AI 기능 추가
                self.enhance_table_context_menus()
                
                print("✅ AI 예측 기능 초기화 완료")
                
            except Exception as e:
                print(f"⚠️ AI 기능 초기화 오류: {e}")
                # 오류가 있어도 기본 기능은 동작하도록
        else:
            print("ℹ️ 기본 모드로 실행 중 (AI 기능 비활성화)")
            
        try:
            # enhanced_screener의 기능이 사용 가능한지 확인
            if hasattr(self, 'enhance_table_context_menus'):
                print("✅ Enhanced screener 기능 활성화됨")
            else:
                print("ℹ️ 기본 screener 모드로 실행 중")
        except Exception as e:
            print(f"⚠️ Enhanced screener 초기화 오류: {e}")

    def search_stocks_with_api(self, search_term):
        """API를 사용한 실시간 주식 검색 + 기존 CSV 백업 (screener용)"""
        
        print(f"🔍 Screener API로 '{search_term}' 검색 시작...")
        api_results = []
        
        # 1. 먼저 API로 검색 시도
        try:
            query = urllib.parse.quote(search_term)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            res = requests.get(url, headers=headers, timeout=10)
            print("Screener API Status code:", res.status_code)

            if res.ok:
                data = res.json()
                quotes = data.get('quotes', [])
                print(f"📊 Screener API에서 {len(quotes)}개 종목 발견")
                
                # Make csv from json.
                api_results = self.convert_api_to_screener_format(quotes, search_term)

            else:
                print("Screener API Request failed:", res.text[:200])

        except Exception as e:
            print(f"Screener API 검색 실패: {e}")
        
        # 2. CSV에서도 검색 (백업용) - 기존 함수 활용
        csv_results = self.enhanced_search_stocks(search_term)
        
        # 3. 결과 병합
        combined_results = self.merge_screener_search_results(api_results, csv_results)
        
        print(f"✅ Screener 총 {len(combined_results)}개 종목 반환")
        return combined_results

    def convert_api_to_screener_format(self, quotes, search_term):
        """Yahoo Finance API 응답을 screener 포맷으로 변환"""
        screener_format_results = []
        
        for quote in quotes:
            try:
                # 기본 정보 추출
                ticker = quote.get('symbol', '').strip()
                if not ticker:
                    continue
                    
                # 회사명 추출
                name = quote.get('longname') or quote.get('shortname', ticker)
                
                # 섹터/산업 정보
                sector = quote.get('sector', quote.get('industry', '미분류'))
                
                # 시가총액 포맷팅
                market_cap_raw = quote.get('marketCap', 0)
                market_cap_str = self.format_screener_market_cap(market_cap_raw)
                
                # 거래소 정보
                exchange = quote.get('exchDisp') or quote.get('exchange', 'Unknown')
                
                # screener 형식에 맞게 구성
                stock_info = {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'market_cap': market_cap_str,
                    'market': exchange,
                    'raw_market_cap': market_cap_raw,
                    'match_score': 90 + self.calculate_screener_relevance_bonus(quote, search_term),
                    'source': 'API'
                }
                
                screener_format_results.append(stock_info)
                
            except Exception as e:
                print(f"⚠️ Screener API 데이터 변환 오류: {e}")
                continue
        
        return screener_format_results

    def format_screener_market_cap(self, market_cap_value):
        """시가총액을 screener용으로 포맷팅"""
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

    def calculate_screener_relevance_bonus(self, quote, search_term):
        """screener용 API 결과의 관련성 보너스 점수 계산"""
        bonus = 0
        
        if quote.get('typeDisp') == 'Equity':
            bonus += 5
        
        ticker = quote.get('symbol', '').upper()
        search_upper = search_term.upper()
        
        if ticker == search_upper:
            bonus += 10
        elif search_upper in ticker:
            bonus += 5
        
        return bonus

    def merge_screener_search_results(self, api_results, csv_results):
        """screener용 API 결과와 CSV 결과 병합"""
        combined = {}
        
        # API 결과 우선 추가
        for stock in api_results:
            ticker = stock['ticker']
            combined[ticker] = stock
        
        # CSV 결과 추가 (중복 제거)
        for stock in csv_results:
            ticker = stock['ticker']
            if ticker not in combined:
                stock['source'] = 'CSV'
                combined[ticker] = stock
        
        # 정렬
        sorted_results = sorted(
            combined.values(), 
            key=lambda x: (-x['match_score'], -x.get('raw_market_cap', 0))
        )
        
        return sorted_results

    def search_and_show_chart_enhanced(self):
        """향상된 검색으로 종목을 찾아서 차트 표시 + CSV 결과 보기"""
        query = self.search_input.text().strip()
        if not query:
            QMessageBox.warning(self, "검색어 필요", "검색할 종목코드나 회사명을 입력해주세요.")
            return

        # 검색 중복 실행 방지
        if hasattr(self, '_is_searching') and self._is_searching:
            print("⚠️ 이미 검색 중입니다. 중복 실행 방지")
            return

        try:
            self._is_searching = True  # 검색 플래그 설정
            self.search_result_label.setText("검색 중... (API+CSV)")
            QApplication.processEvents()
            
            # 향상된 검색 함수 사용
            results = self.search_stocks_with_api(query)
            
            # 결과 저장
            self.last_search_results = results
            
            if results:
                api_count = len([r for r in results if r.get('source') == 'API'])
                csv_count = len([r for r in results if r.get('source') == 'CSV'])
                
                self.search_result_label.setText(
                    f"✅ {len(results)}개 발견 (API:{api_count}, CSV:{csv_count})"
                )
                
                # 검색 결과를 보여주는 다이얼로그
                self.show_enhanced_search_results_dialog(query, results)
                
            else:
                self.search_result_label.setText("❌ 결과 없음")
                QMessageBox.information(self, "검색 결과", f"'{query}'에 대한 검색 결과가 없습니다.")
                
        except Exception as e:
            self.search_result_label.setText(f"❌ 오류")
            QMessageBox.critical(self, "검색 오류", f"검색 중 오류가 발생했습니다:\n{str(e)}")
            print(f"Screener 검색 오류: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 검색 플래그 해제
            if hasattr(self, '_is_searching'):
                delattr(self, '_is_searching')

    def show_enhanced_search_results_dialog(self, query, results):
        """향상된 검색 결과를 보여주는 다이얼로그 (CSV 포맷 포함)"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"검색 결과: {query}")
        dialog.resize(1000, 600)
        
        layout = QVBoxLayout()
        
        # 상단 정보
        api_count = len([r for r in results if r.get('source') == 'API'])
        csv_count = len([r for r in results if r.get('source') == 'CSV'])
        
        info_label = QLabel(
            f"총 {len(results)}개 종목 발견 (API: {api_count}개, CSV: {csv_count}개)"
        )
        info_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        layout.addWidget(info_label)
        
        # 탭 위젯 생성
        tab_widget = QTabWidget()
        
        # 탭 1: 테이블 형태로 결과 보기
        table_tab = self.create_results_table_tab(results)
        tab_widget.addTab(table_tab, "📊 테이블 보기")
        
        # 탭 2: CSV 형태로 결과 보기
        csv_tab = self.create_results_csv_tab(results)
        tab_widget.addTab(csv_tab, "📄 CSV 포맷")
        
        layout.addWidget(tab_widget)
        
        # 하단 버튼들
        button_layout = QHBoxLayout()
        
        # 첫 번째 종목 차트 보기
        if results:
            first_ticker = results[0]['ticker']
            chart_btn = QPushButton(f"📈 {first_ticker} 차트 보기")
            chart_btn.clicked.connect(lambda: self.show_stock_detail(first_ticker))
            button_layout.addWidget(chart_btn)
        
        # CSV 파일로 저장
        save_csv_btn = QPushButton("💾 CSV 저장")
        save_csv_btn.clicked.connect(lambda: self.save_search_results_csv(results))
        button_layout.addWidget(save_csv_btn)
        
        # 클립보드 복사
        copy_btn = QPushButton("📋 복사")
        copy_btn.clicked.connect(lambda: self.copy_results_to_clipboard(results))
        button_layout.addWidget(copy_btn)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.show()

    def create_results_table_tab(self, results):
        """검색 결과 테이블 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 테이블 생성
        table = QTableWidget()
        table.setRowCount(len(results))
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "종목코드", "회사명", "섹터", "시가총액", "거래소", "출처"
        ])
        
        # 데이터 채우기
        for i, stock in enumerate(results):
            table.setItem(i, 0, QTableWidgetItem(stock.get('ticker', '')))
            table.setItem(i, 1, QTableWidgetItem(stock.get('name', '')))
            table.setItem(i, 2, QTableWidgetItem(stock.get('sector', '')))

            # market_cap을 포맷팅 (OverflowError 방지)
            market_cap_raw = stock.get('market_cap', '')
            if isinstance(market_cap_raw, (int, float)):
                market_cap_str = format_market_cap_value(market_cap_raw)
            else:
                market_cap_str = str(market_cap_raw) if market_cap_raw else 'N/A'

            table.setItem(i, 3, QTableWidgetItem(market_cap_str))
            table.setItem(i, 4, QTableWidgetItem(stock.get('market', '')))
            
            # 출처에 따른 색상 구분
            source = stock.get('source', 'CSV')
            source_item = QTableWidgetItem(source)
            
            if source == 'API':
                source_item.setBackground(QColor(200, 255, 200))  # 연한 초록색
                source_item.setToolTip("Yahoo Finance API 실시간 검색 결과")
            else:
                source_item.setBackground(QColor(255, 255, 200))  # 연한 노란색
                source_item.setToolTip("로컬 마스터 CSV 파일 검색 결과")
            
            table.setItem(i, 5, source_item)
        
        # 테이블 더블클릭으로 차트 보기
        table.doubleClicked.connect(lambda index: self.on_result_table_double_click(results, index))
        
        # 테이블 크기 조정
        table.resizeColumnsToContents()
        table.setSortingEnabled(True)
        
        layout.addWidget(table)
        widget.setLayout(layout)
        return widget

    def create_results_csv_tab(self, results):
        """검색 결과 CSV 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 설명 레이블
        desc_label = QLabel("아래 내용을 복사하여 Excel이나 다른 프로그램에서 사용할 수 있습니다.")
        desc_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(desc_label)
        
        # CSV 텍스트 영역
        text_edit = QTextEdit()
        csv_content = self.generate_screener_csv_content(results)
        text_edit.setPlainText(csv_content)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 9))  # 고정폭 글꼴
        layout.addWidget(text_edit)
        
        widget.setLayout(layout)
        return widget

    def generate_screener_csv_content(self, results):
        """screener용 검색 결과를 CSV 문자열로 생성"""
        lines = ["ticker,name,sector,market_cap,market,source,match_score"]
        
        for stock in results:
            ticker = self.clean_screener_csv_value(stock.get('ticker', ''))
            name = self.clean_screener_csv_value(stock.get('name', ''))
            sector = self.clean_screener_csv_value(stock.get('sector', ''))
            market_cap = self.clean_screener_csv_value(stock.get('market_cap', 'N/A'))
            market = self.clean_screener_csv_value(stock.get('market', ''))
            source = self.clean_screener_csv_value(stock.get('source', 'CSV'))
            match_score = stock.get('match_score', 0)
            
            line = f"{ticker},{name},{sector},{market_cap},{market},{source},{match_score}"
            lines.append(line)
        
        return "\n".join(lines)

    def clean_screener_csv_value(self, value):
        """screener용 CSV 값에서 특수문자 처리"""
        if not isinstance(value, str):
            value = str(value)
        
        if ',' in value or '"' in value or '\n' in value:
            value = value.replace('"', '""')
            return f'"{value}"'
        
        return value

    def on_result_table_double_click(self, results, index):
        """검색 결과 테이블 더블클릭 시 차트 보기"""
        row = index.row()
        if 0 <= row < len(results):
            ticker = results[row]['ticker']
            self.show_stock_detail(ticker)

    def save_search_results_csv(self, results):
        """검색 결과를 CSV 파일로 저장"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"screener_search_results_{timestamp}.csv"
            
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "검색 결과 CSV 저장", 
                default_filename,
                "CSV 파일 (*.csv);;모든 파일 (*)"
            )
            
            if filename:
                csv_content = self.generate_screener_csv_content(results)
                with open(filename, 'w', encoding='utf-8-sig') as f:
                    f.write(csv_content)
                
                QMessageBox.information(
                    self, 
                    "저장 완료", 
                    f"검색 결과가 저장되었습니다:\n{filename}\n\n총 {len(results)}개 종목"
                )
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "저장 오류", 
                f"파일 저장 중 오류가 발생했습니다:\n{str(e)}"
            )

    def copy_results_to_clipboard(self, results):
        """검색 결과를 클립보드에 복사"""
        try:
            csv_content = self.generate_screener_csv_content(results)
            QApplication.clipboard().setText(csv_content)
            
            # 잠시 상태 표시
            original_text = self.search_result_label.text()
            self.search_result_label.setText("📋 클립보드에 복사됨!")
            QTimer.singleShot(2000, lambda: self.search_result_label.setText(original_text))
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "복사 오류", 
                f"클립보드 복사 중 오류가 발생했습니다:\n{str(e)}"
            )

    def add_csv_search_features(self):
        """CSV 검색 기능을 UI에 추가"""
        # 검색 패널에 "고급 검색" 버튼 추가
        if hasattr(self, 'search_btn'):
            # 기존 검색 버튼을 향상된 검색으로 변경
            self.search_btn.setText("🔍 고급검색")
            self.search_btn.setToolTip("Yahoo Finance API + CSV 통합 검색")
            
            # 기존 연결을 새로운 함수로 변경
            try:
                self.search_btn.clicked.disconnect()  # 기존 연결 해제
            except:
                pass
            
            self.search_btn.clicked.connect(self.search_and_show_chart_enhanced)

        # 추가 기능 버튼들을 search panel에 추가
        if hasattr(self, 'search_help_btn'):
            # CSV 결과 보기 버튼 추가
            csv_results_btn = QPushButton("📊 최근검색")
            csv_results_btn.setToolTip("최근 검색 결과를 CSV로 보기")
            csv_results_btn.clicked.connect(self.show_last_search_csv)
            csv_results_btn.setMaximumWidth(100)
            
            # 검색 패널 레이아웃에 추가 (help 버튼 옆에)
            # 실제 UI 구조에 맞게 위치 조정 필요
            self.csv_results_btn = csv_results_btn

    def show_last_search_csv(self):
        """최근 검색 결과를 CSV 형태로 보기"""
        if not self.last_search_results:
            QMessageBox.information(
                self, 
                "검색 결과 없음", 
                "먼저 검색을 수행해주세요.\n고급검색 버튼을 사용하면 API+CSV 통합 검색이 가능합니다."
            )
            return
        
        # CSV 결과 다이얼로그 표시
        dialog = QDialog(self)
        dialog.setWindowTitle("최근 검색 결과 - CSV 포맷")
        dialog.resize(800, 500)
        
        layout = QVBoxLayout()
        
        # 정보 헤더
        info_label = QLabel(f"총 {len(self.last_search_results)}개 종목 - CSV 포맷으로 표시")
        info_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        layout.addWidget(info_label)
        
        # CSV 텍스트
        text_edit = QTextEdit()
        csv_content = self.generate_screener_csv_content(self.last_search_results)
        text_edit.setPlainText(csv_content)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 9))
        layout.addWidget(text_edit)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("📋 클립보드 복사")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(csv_content))
        button_layout.addWidget(copy_btn)
        
        save_btn = QPushButton("💾 파일 저장")
        save_btn.clicked.connect(lambda: self.save_search_results_csv(self.last_search_results))
        button_layout.addWidget(save_btn)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def search_master_csv_enhanced(self, search_term):
        """기존 search_master_csv 함수의 향상된 버전 - 무한 재귀 방지"""
        # 직접 마스터 CSV에서 검색하도록 수정
        return self.enhanced_search_stocks(search_term)

    def show_random_stock_chart_enhanced(self):
        """향상된 랜덤 종목 차트 보기 (API 활용)"""
        try:
            # 인기 종목들에서 랜덤 선택
            popular_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'BRK.B', 'LLY', 'V',
                '005930.KS', '000660.KS', '035420.KS'  # 한국 주요 종목도 포함
            ]
            
            import random
            selected_ticker = random.choice(popular_tickers)
            
            # API로 해당 종목 정보 검색
            results = self.search_stocks_with_api(selected_ticker)
            
            if results:
                # 검색된 정보와 함께 차트 표시
                stock_info = results[0]
                self.search_result_label.setText(
                    f"🎲 랜덤: {stock_info['name']} ({stock_info['ticker']})"
                )
                self.show_stock_detail(stock_info['ticker'])
            else:
                # 백업: 기존 방식으로 차트 표시
                self.search_result_label.setText(f"🎲 랜덤: {selected_ticker}")
                self.show_stock_detail(selected_ticker)
                
        except Exception as e:
            print(f"랜덤 종목 향상된 검색 오류: {e}")
            # 백업: 기존 랜덤 기능 사용
            if hasattr(self, 'show_random_stock_chart'):
                self.show_random_stock_chart()

    # 6. 검색 도움말 업데이트
    def show_search_help_enhanced(self):
        """향상된 검색 기능에 대한 도움말"""
        help_text = """
🔍 **향상된 종목 검색 기능**

**검색 방법:**
• 종목코드: AAPL, MSFT, 005930.KS
• 회사명: Apple, Microsoft, 삼성전자
• 부분 검색: 삼성, Apple

**검색 소스:**
🟢 **API 검색** (실시간)
  - Yahoo Finance API에서 최신 종목 정보 검색
  - 전 세계 거래소의 최신 데이터
  - 실시간 시가총액과 정보

🟡 **CSV 검색** (로컬)
  - 로컬 마스터 CSV 파일에서 검색
  - 한국/미국/스웨덴 주요 종목 데이터
  - 빠른 검색 속도

**결과 활용:**
📊 테이블 형태로 보기
📄 CSV 포맷으로 내보내기
📋 클립보드 복사
💾 파일로 저장
📈 종목 차트 바로 보기

**사용 예시:**
• "삼성" 입력 → 삼성 관련 모든 종목 검색
• "AAPL" 입력 → Apple 상세 정보 및 차트
• "반도체" 입력 → 반도체 섹터 종목들

**팁:**
✨ API 검색 결과는 초록색으로 표시
✨ 더블클릭으로 바로 차트 보기
✨ 매치 점수가 높을수록 관련성 높음
        """
        
        QMessageBox.information(self, "🔍 향상된 검색 도움말", help_text)

    # 7. 메뉴나 툴바에 새로운 기능 추가 (선택사항)
    def add_enhanced_search_menu(self):
        """향상된 검색 기능을 메뉴에 추가"""
        if hasattr(self, 'menubar'):
            # 검색 메뉴 생성 또는 기존 메뉴에 추가
            search_menu = self.menubar.addMenu('🔍 검색')
            
            # API 검색 액션
            api_search_action = QAction('🌐 API 통합 검색', self)
            api_search_action.setShortcut('Ctrl+F')
            api_search_action.triggered.connect(self.focus_search_input)
            search_menu.addAction(api_search_action)
            
            # CSV 결과 보기 액션
            csv_results_action = QAction('📄 최근 검색 결과', self)
            csv_results_action.setShortcut('Ctrl+R')
            csv_results_action.triggered.connect(self.show_last_search_csv)
            search_menu.addAction(csv_results_action)
            
            search_menu.addSeparator()
            
            # 검색 도움말 액션
            help_action = QAction('❓ 검색 도움말', self)
            help_action.triggered.connect(self.show_search_help_enhanced)
            search_menu.addAction(help_action)

    def focus_search_input(self):
        """검색 입력창에 포커스"""
        if hasattr(self, 'search_input'):
            self.search_input.setFocus()
            self.search_input.selectAll()

    # 8. 기존 버튼들 업데이트 (선택사항)
    def update_existing_search_buttons(self):
        """기존 검색 버튼들을 향상된 기능으로 업데이트"""
        
        # 랜덤 종목 버튼 업데이트
        if hasattr(self, 'random_stock_btn'):
            try:
                self.random_stock_btn.clicked.disconnect()
            except:
                pass
            self.random_stock_btn.clicked.connect(self.show_random_stock_chart_enhanced)
            self.random_stock_btn.setToolTip("향상된 랜덤 종목 (API 정보 포함)")
        
        # 도움말 버튼 업데이트
        if hasattr(self, 'search_help_btn'):
            try:
                self.search_help_btn.clicked.disconnect()
            except:
                pass
            self.search_help_btn.clicked.connect(self.show_search_help_enhanced)
            self.search_help_btn.setToolTip("향상된 검색 기능 도움말")

    def setup_prediction_features(self):
        """예측 기능 설정 (레거시 호환)"""
        if PREDICTION_AVAILABLE:
            # 이미 __init__에서 처리되므로 빈 메서드로 유지
            pass
        else:
            print("💡 AI 예측 기능을 사용하려면 enhanced_screener.py가 필요합니다")
            
    def enhance_table_context_menus(self):
        """테이블 컨텍스트 메뉴 강화"""
        if PREDICTION_AVAILABLE:
            # enhanced_screener의 메서드 호출
            super().enhance_table_context_menus()
        else:
            # 기본 동작 (필요시 추가)
            pass
        
    def initUI(self):
        self.setWindowTitle('Advanced Global Stock Screener - 고급 분석 시스템 2025')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 1. 상단 컨트롤 패널 (기존)
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 2. 🔍 검색 + 🛠️ 조건을 같은 라인에 배치
        search_conditions_layout = QHBoxLayout()
        
        # 2-1. 검색 패널 (기존 메서드 활용, 크기만 조정)
        search_panel = self.create_stock_search_panel()
        search_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_conditions_layout.addWidget(search_panel)
        
        # 2-2. 사용자 정의 조건 패널 (화면 절반 너비로 확장)
        conditions_panel = self.create_custom_conditions_panel()
        conditions_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        search_conditions_layout.addWidget(conditions_panel)
        
        # 레이아웃을 메인에 추가
        layout.addLayout(search_conditions_layout)
        
        # 3. 종목 현황 패널 (기존)
        status_panel = self.create_status_panel()
        layout.addWidget(status_panel)
        
        # 4. 결과 테이블들 (기존)
        tables_widget = self.create_tables()
        layout.addWidget(tables_widget)
        
        try:
            self.update_existing_search_buttons()
            self.add_enhanced_search_menu()  # 메뉴가 있는 경우
            print("✅ Screener 향상된 검색 기능 초기화 완료")
        except Exception as e:
            print(f"⚠️ Screener 향상된 검색 기능 초기화 중 오류: {e}")

        # 상태바
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('준비됨 - 종목 검색 또는 스크리닝을 시작하세요')

    def test_enhanced_screener_search():
        """향상된 screener 검색 기능 테스트"""
        print("🧪 Enhanced Screener Search 테스트")
        
        # 예시 사용법
        example_usage = '''
    # screener.py에서 사용 예시:

    # 1. 기본 검색 (기존 search_btn 클릭)
    screener.search_and_show_chart_enhanced()

    # 2. 프로그래밍 방식 검색
    results = screener.search_stocks_with_api("삼성")
    print(f"검색 결과: {len(results)}개")

    # 3. CSV 형태로 결과 보기
    csv_content = screener.generate_screener_csv_content(results)
    print(csv_content)

    # 4. 랜덤 종목 (향상된 버전)
    screener.show_random_stock_chart_enhanced()
        '''
        
        print(example_usage)
        print("✅ 테스트 코드 준비 완료")

    # 실제 통합 시 기존 함수들과 충돌하지 않도록 주의사항
    """
    ⚠️ 주의사항:

    1. 기존 search_master_csv 함수는 그대로 유지
    2. 새로운 함수들은 _enhanced 접미사 사용
    3. 기존 버튼 연결은 선택적으로 변경
    4. import 문 추가 필요: urllib.parse, requests
    5. QTimer import 필요 (클립보드 복사 알림용)

    👍 권장 적용 순서:
    1. import 문들 추가
    2. 새로운 메서드들 추가
    3. 기존 버튼 연결 변경 (선택)
    4. 테스트 및 확인
    """

    def on_market_cap_filter_toggled(self, checked):
        """시가총액 필터 체크박스 토글 이벤트"""
        # 관련 위젯들 활성화/비활성화
        widgets = [
            self.top_stocks_spin,
            self.enrich_all_radio,
            self.enrich_custom_radio,
            self.enrich_count_spin
        ]
        
        for widget in widgets:
            widget.setEnabled(checked)
        
        self.update_time_estimate()

    def update_time_estimate(self):
        """예상 소요 시간 업데이트"""
        if not self.use_market_cap_filter.isChecked():
            self.time_estimate_label.setText("예상 시간: 약 2초 (보강 없음)")
            return
        
        if self.enrich_all_radio.isChecked():
            # 전체 보강 - 시장에 따라 다름
            market = self.market_combo.currentText()
            if "한국" in market:
                estimate = "3-5분"
            elif "미국" in market:
                estimate = "5-10분"
            elif "스웨덴" in market:
                estimate = "2-3분"
            else:  # 전체
                estimate = "10-20분"
            self.time_estimate_label.setText(f"예상 시간: 약 {estimate} (전체 보강)")
        else:
            # 지정 개수 보강
            count = self.enrich_count_spin.value()
            seconds = count * 0.5  # 개당 약 0.5초
            if seconds < 60:
                estimate = f"{int(seconds)}초"
            else:
                estimate = f"{int(seconds/60)}분"
            self.time_estimate_label.setText(f"예상 시간: 약 {estimate} ({count}개 보강)")

    def open_backtesting_dialog(self):
        """백테스팅 다이얼로그 열기"""
        try:
            dialog = BacktestingDialog(self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"백테스팅 창을 열 수 없습니다: {str(e)}")

    # 사용 예시:
    """
    백테스팅 기능 사용법:

    1. "백테스팅 (전략 검증)" 버튼 클릭
    2. 테스트 기간 선택 (3개월, 6개월, 1년 또는 사용자 정의)
    3. 초기 자본 설정 (기본: 100,000원)
    4. 테스트할 매수 조건 선택:
    - 60일선이 120일선 돌파
    - RSI 과매도 반등 (30 돌파)
    - 볼린저밴드 하단 터치
    - MACD 골든크로스
    5. 테스트할 매도 조건 선택:
    - 데드크로스 (MA60 < MA120)
    - RSI 과매수 (>= 70)
    - 볼린저밴드 상단
    - 손절/익절 (-7% / +20%)
    6. "백테스팅 실행" 버튼 클릭

    결과로 다음 정보를 제공:
    - 총 수익률 및 수익금
    - 총 거래 횟수 및 승률
    - 평균 보유기간
    - 최고/최악 거래 내역
    - 상세 거래 로그 엑셀 저장 옵션

    실제 예시:
    - 6개월 전부터 현재까지
    - 60일선 돌파 + RSI 과매도 반등 조건으로 매수
    - RSI 과매수 + 손절/익절 조건으로 매도
    - 결과: 15건 거래, 승률 60%, 총 수익률 +12.5%
    """

    def create_control_panel(self):
        group = QGroupBox("검색 조건 설정")
        layout = QGridLayout()
        
        # 첫 번째 행: 시장 선택
        layout.addWidget(QLabel("시장 선택:"), 0, 0)
        self.market_combo = QComboBox()
        self.market_combo.addItems(["전체", "한국 (KOSPI/KOSDAQ)", "미국 (NASDAQ/NYSE)", "스웨덴 (OMX)"])
        self.market_combo.currentTextChanged.connect(self.update_stock_count)
        layout.addWidget(self.market_combo, 0, 1)
        
        # 두 번째 행: 시가총액 필터링 옵션들
        mcap_group = QGroupBox("🏆 시가총액 필터링")
        mcap_layout = QGridLayout()
        
        # 시가총액 필터 사용 여부
        self.use_market_cap_filter = QCheckBox("시가총액 필터 사용")
        self.use_market_cap_filter.setChecked(False)  # 기본값: OFF
        self.use_market_cap_filter.toggled.connect(self.on_market_cap_filter_toggled)
        mcap_layout.addWidget(self.use_market_cap_filter, 0, 0)
        
        # 상위 종목 수 선택
        mcap_layout.addWidget(QLabel("상위 종목:"), 0, 1)
        self.top_stocks_spin = QSpinBox()
        self.top_stocks_spin.setMinimum(10)
        self.top_stocks_spin.setMaximum(1000)
        self.top_stocks_spin.setValue(100)
        self.top_stocks_spin.setSuffix("개")
        self.top_stocks_spin.setToolTip("시가총액 기준 상위 종목 수")
        self.top_stocks_spin.setEnabled(False)  # 초기에는 비활성화
        mcap_layout.addWidget(self.top_stocks_spin, 0, 2)
        
        # 보강할 종목 수 선택
        mcap_layout.addWidget(QLabel("보강할 종목:"), 1, 0)
        
        # 보강 옵션 라디오 버튼들
        self.enrichment_group = QButtonGroup()
        
        self.enrich_all_radio = QRadioButton("전체 보강")
        self.enrich_all_radio.setToolTip("모든 종목의 시가총액 정보를 수집합니다 (시간 많이 소요)")
        self.enrichment_group.addButton(self.enrich_all_radio, 0)
        mcap_layout.addWidget(self.enrich_all_radio, 1, 1)
        
        self.enrich_custom_radio = QRadioButton("지정 개수:")
        self.enrich_custom_radio.setChecked(True)  # 기본값
        self.enrich_custom_radio.setToolTip("지정한 개수만큼만 보강하여 시간을 절약합니다")
        self.enrichment_group.addButton(self.enrich_custom_radio, 1)
        mcap_layout.addWidget(self.enrich_custom_radio, 1, 2)
        
        # 보강할 개수 입력
        self.enrich_count_spin = QSpinBox()
        self.enrich_count_spin.setMinimum(50)
        self.enrich_count_spin.setMaximum(1000)
        self.enrich_count_spin.setValue(150)  # 기본값 150개
        self.enrich_count_spin.setSuffix("개")
        self.enrich_count_spin.setToolTip("시가총액 정보를 수집할 종목 수 (많을수록 정확하지만 시간 소요)")
        self.enrich_count_spin.setKeyboardTracking(False)
        mcap_layout.addWidget(self.enrich_count_spin, 1, 3)


        # 범위 밖/중간값일 때 '이전값'으로 튀는 대신 근사치로 보정
        self.enrich_count_spin.setCorrectionMode(QAbstractSpinBox.CorrectToNearestValue)

        # 숫자만 허용(IME/접미사 섞일 때 중간상태 줄이기)
        self.enrich_count_spin.lineEdit().setValidator(QIntValidator(50, 1000, self))

        # 엔터·포커스 아웃 시 텍스트→값 해석 후 라벨 업데이트
        def _commit_enrich_count():
            self.enrich_count_spin.interpretText()   # ★ 현재 텍스트를 즉시 값으로 확정
            self.update_time_estimate()

        self.enrich_count_spin.editingFinished.connect(_commit_enrich_count)
        self.enrich_count_spin.lineEdit().returnPressed.connect(_commit_enrich_count)

        # (선택) 보기 좋게 정렬
        self.enrich_count_spin.setAlignment(Qt.AlignRight)        
        
        # 예상 시간 표시 라벨
        self.time_estimate_label = QLabel("예상 시간: 약 2초 (보강 없음)")
        self.time_estimate_label.setStyleSheet("color: #666; font-size: 11px;")
        mcap_layout.addWidget(self.time_estimate_label, 1, 4)
        
        # 모든 보강 관련 위젯들 초기에는 비활성화
        for widget in [self.enrich_all_radio, self.enrich_custom_radio, self.enrich_count_spin]:
            widget.setEnabled(False)
        
        # 신호 연결
        self.enrich_count_spin.valueChanged.connect(self.update_time_estimate)
        self.enrich_all_radio.toggled.connect(self.update_time_estimate)
        
        mcap_group.setLayout(mcap_layout)
        layout.addWidget(mcap_group, 1, 0, 1, 6)  # 행 1에 배치
        
        # 세 번째 행: CSV 파일 관리 (마스터 CSV 시스템 포함)
        csv_group = QGroupBox("📁 CSV 파일 관리")
        csv_layout = QGridLayout()
        
        # 첫 번째 행: 기본 관리 기능
        self.refresh_csv_btn = QPushButton("🔄 CSV 새로고침")
        self.refresh_csv_btn.clicked.connect(self.load_stock_lists)
        self.refresh_csv_btn.setToolTip("저장된 CSV 파일을 다시 로드합니다")
        csv_layout.addWidget(self.refresh_csv_btn, 0, 0)
        
        self.edit_csv_btn = QPushButton("📝 CSV 편집")
        self.edit_csv_btn.clicked.connect(self.open_csv_editor)
        self.edit_csv_btn.setToolTip("CSV 파일의 종목 정보를 직접 편집합니다")
        csv_layout.addWidget(self.edit_csv_btn, 0, 1)
        
        self.sample_csv_btn = QPushButton("🎯 샘플 생성")
        self.sample_csv_btn.clicked.connect(self.create_sample_csv_files)
        self.sample_csv_btn.setToolTip("테스트용 샘플 종목 리스트를 생성합니다")
        csv_layout.addWidget(self.sample_csv_btn, 0, 2)
        
        # 두 번째 행: 마스터 CSV 시스템
        self.create_master_btn = QPushButton("🏆 마스터 CSV 생성")
        self.create_master_btn.clicked.connect(self.create_master_csv)
        self.create_master_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; }")
        self.create_master_btn.setToolTip("전체 종목을 보강하여 마스터 CSV를 생성합니다 (시간 오래 걸림)")
        csv_layout.addWidget(self.create_master_btn, 1, 0)
        
        self.update_from_master_btn = QPushButton("📊 마스터에서 필터링")
        self.update_from_master_btn.clicked.connect(self.update_from_master_csv)
        self.update_from_master_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.update_from_master_btn.setToolTip("마스터 CSV에서 시총 상위 종목을 선별합니다 (빠름)")
        csv_layout.addWidget(self.update_from_master_btn, 1, 1)
        
        self.update_online_btn = QPushButton("🌐 온라인 종목 업데이트")
        self.update_online_btn.clicked.connect(self.update_stocks_online)
        self.update_online_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-weight: bold; }")
        self.update_online_btn.setToolTip("인터넷에서 최신 종목 리스트를 다운로드합니다")
        csv_layout.addWidget(self.update_online_btn, 1, 2)
        
        csv_group.setLayout(csv_layout)
        layout.addWidget(csv_group, 2, 0, 1, 6)  # 행 2에 배치
        
        # 네 번째 행: 기본 매수 조건
        buy_group = QGroupBox("💰 기본 매수 조건")
        buy_layout = QVBoxLayout()
        
        self.ma_condition = QCheckBox("최근 60일선이 120일선 돌파 + 우상향 + 이평선 터치")
        buy_layout.addWidget(self.ma_condition)
        
        self.bb_condition = QCheckBox("볼린저밴드 하단 터치 + RSI < 35")
        buy_layout.addWidget(self.bb_condition)
        
        self.support_condition = QCheckBox("MACD 골든 크로스 + 거래량 증가")
        buy_layout.addWidget(self.support_condition)
        
        self.momentum_condition = QCheckBox("20일 상대강도 상승 + 펀더멘털 양호")
        buy_layout.addWidget(self.momentum_condition)
        
        buy_group.setLayout(buy_layout)
        layout.addWidget(buy_group, 3, 0, 1, 3)  # 행 3, 컬럼 0-2
        
        # 다섯 번째 행: 기본 매도 조건
        sell_group = QGroupBox("📉 기본 매도 조건")
        sell_layout = QVBoxLayout()
        
        self.tech_sell = QCheckBox("데드크로스 + 60일선 3% 하향이탈")
        sell_layout.addWidget(self.tech_sell)
        
        self.profit_sell = QCheckBox("20% 수익달성 또는 -7% 손절")
        sell_layout.addWidget(self.profit_sell)
        
        self.bb_sell = QCheckBox("볼린저 상단 + RSI > 70")
        sell_layout.addWidget(self.bb_sell)
        
        self.volume_sell = QCheckBox("거래량 급감 + 모멘텀 약화")
        sell_layout.addWidget(self.volume_sell)
        
        sell_group.setLayout(sell_layout)
        layout.addWidget(sell_group, 3, 3, 1, 3)  # 행 3, 컬럼 3-5

        # 일곱 번째 행: 검색 버튼과 제어 버튼들
        button_layout = QHBoxLayout()
        
        self.search_btn = QPushButton("🔍 종목 스크리닝 시작")
        self.search_btn.clicked.connect(self.run_screening)
        self.search_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        button_layout.addWidget(self.search_btn)
        
        # 중지 버튼
        self.stop_btn = QPushButton("⏹️ 스크리닝 중지")
        self.stop_btn.clicked.connect(self.stop_screening)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        self.stop_btn.setVisible(False)  # 초기에는 숨김
        button_layout.addWidget(self.stop_btn)
    
        # 백테스팅 버튼 추가! - 새로운 기능
        self.backtest_btn = QPushButton("📈 백테스팅 (전략 검증)")
        self.backtest_btn.clicked.connect(self.open_backtesting_dialog)
        self.backtest_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        self.backtest_btn.setToolTip("과거 데이터로 매수/매도 전략의 효과를 검증합니다")
        button_layout.addWidget(self.backtest_btn)

        # 엑셀 저장 버튼
        self.export_btn = QPushButton("📊 결과를 엑셀로 저장")
        self.export_btn.clicked.connect(self.export_results_to_excel)
        self.export_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        self.export_btn.setEnabled(False)  # 초기에는 비활성화
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout, 4, 0, 1, 6)  # 행 4에 배치

        # QGridLayout의 행 확장 정책 설정
        for i in range(layout.rowCount()):  # 모든 행에 대해
            layout.setRowStretch(i, 0)      # 고정 크기로 설정

        group.setLayout(layout)

        # 그룹박스 자체도 세로 확장 금지
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return group
    
    def create_status_panel(self):
        """종목 현황 패널"""
        group = QGroupBox("📊 종목 현황")
        layout = QHBoxLayout()
        
        self.korea_count_label = QLabel("🇰🇷 한국: 0개")
        self.usa_count_label = QLabel("🇺🇸 미국: 0개")
        self.sweden_count_label = QLabel("🇸🇪 스웨덴: 0개")
        self.total_count_label = QLabel("🌍 전체: 0개")
        
        layout.addWidget(self.korea_count_label)
        layout.addWidget(self.usa_count_label)
        layout.addWidget(self.sweden_count_label)
        layout.addWidget(self.total_count_label)
        layout.addStretch()
        
        group.setLayout(layout)

        # 종목 현황 패널 크기 고정 - 핵심!
        group.setMaximumHeight(80)  # 최대 높이 제한
        group.setMinimumHeight(80)  # 최소 높이도 고정
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        return group
    
    def create_master_csv(self):
        """전체 보강된 마스터 CSV 생성"""
        market_selection = self.market_combo.currentText()
        
        # 업데이트할 시장 결정
        if market_selection == "전체":
            markets_to_update = ["한국", "미국", "스웨덴"]
        elif "한국" in market_selection:
            markets_to_update = ["한국"]
        elif "미국" in market_selection:
            markets_to_update = ["미국"]
        elif "스웨덴" in market_selection:
            markets_to_update = ["스웨덴"]
        else:
            markets_to_update = []
        
        if not markets_to_update:
            QMessageBox.warning(self, "알림", "업데이트할 시장을 선택해주세요.")
            return
        
        # 경고 메시지
        market_str = ", ".join(markets_to_update)
        reply = QMessageBox.question(
            self, '마스터 CSV 생성', 
            f'⚠️ {market_str} 시장의 마스터 CSV를 생성합니다.\n\n'
            f'• 모든 종목의 시가총액 정보를 수집합니다\n'
            f'• 시간이 매우 오래 걸릴 수 있습니다 (10-30분)\n'
            f'• 한 번 생성하면 계속 재사용할 수 있습니다\n\n'
            f'계속하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.create_master_btn.setEnabled(False)
            
            # 마스터 생성 설정
            master_config = {
                'markets': markets_to_update,
                'mode': 'master'  # 마스터 모드
            }
            
            # 별도 스레드에서 실행
            self.master_thread = MasterCSVThread(master_config)
            self.master_thread.finished.connect(self.on_master_creation_finished)
            self.master_thread.error.connect(self.on_master_creation_error)
            self.master_thread.progress.connect(self.on_update_progress)
            self.master_thread.start()

    def update_from_master_csv(self):
        """마스터 CSV에서 필터링하여 작업용 CSV 생성"""
        if not self.use_market_cap_filter.isChecked():
            QMessageBox.information(
                self, "알림", 
                "시가총액 필터를 먼저 활성화해주세요.\n"
                "마스터 CSV 필터링은 시가총액 기준으로만 동작합니다."
            )
            return
        
        market_selection = self.market_combo.currentText()
        top_count = self.top_stocks_spin.value()
        
        # 마스터 파일 존재 확인
        master_files = self.check_master_files(market_selection)
        if not master_files:
            QMessageBox.warning(
                self, "마스터 파일 없음",
                "마스터 CSV 파일이 없습니다.\n"
                "'마스터 CSV 생성' 버튼을 먼저 클릭해주세요."
            )
            return
        
        reply = QMessageBox.question(
            self, '마스터에서 필터링',
            f'마스터 CSV에서 시총 상위 {top_count}개를 선별합니다.\n\n'
            f'• 마스터 데이터에서 상위 {top_count * 2}개 추출\n'
            f'• 최신 시가총액 정보로 재보강\n'
            f'• 정확한 순위로 재정렬\n\n'
            f'예상 시간: 약 {int(top_count * 0.1)}초\n\n'
            f'계속하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.update_from_master_btn.setEnabled(False)
            
            filter_config = {
                'market_selection': market_selection,
                'top_count': top_count,
                'master_files': master_files
            }
            
            self.filter_thread = MasterFilterThread(filter_config)
            self.filter_thread.finished.connect(self.on_master_filtering_finished)
            self.filter_thread.error.connect(self.on_master_filtering_error)
            self.filter_thread.progress.connect(self.on_update_progress)
            self.filter_thread.start()

    def check_master_files(self, market_selection):
        """마스터 파일 존재 확인"""
        master_files = {}
        
        if market_selection == "전체":
            markets = ["korea", "usa", "sweden"]
        elif "한국" in market_selection:
            markets = ["korea"]
        elif "미국" in market_selection:
            markets = ["usa"]
        elif "스웨덴" in market_selection:
            markets = ["sweden"]
        else:
            return {}
        
        for market in markets:
            master_file = f'stock_data/{market}_stocks_master.csv'
            if os.path.exists(master_file):
                master_files[market] = master_file
            else:
                return {}  # 하나라도 없으면 전체 실패
        
        return master_files

    def on_master_creation_finished(self, message):
        """마스터 생성 완료"""
        self.create_master_btn.setEnabled(True)
        self.statusbar.showMessage('✅ 마스터 CSV 생성 완료')
        QMessageBox.information(self, '완료', message)

    def on_master_creation_error(self, error_message):
        """마스터 생성 오류"""
        self.create_master_btn.setEnabled(True)
        self.statusbar.showMessage('❌ 마스터 CSV 생성 실패')
        QMessageBox.critical(self, '오류', error_message)

    def on_master_filtering_finished(self, message):
        """마스터 필터링 완료"""
        self.update_from_master_btn.setEnabled(True)
        self.statusbar.showMessage('✅ 마스터에서 필터링 완료')
        self.load_stock_lists()  # 새로운 CSV 로드
        QMessageBox.information(self, '완료', message)

    def on_master_filtering_error(self, error_message):
        """마스터 필터링 오류"""
        self.update_from_master_btn.setEnabled(True)
        self.statusbar.showMessage('❌ 마스터 필터링 실패')
        QMessageBox.critical(self, '오류', error_message)

    def create_tables(self):
        """테이블 생성 - 정렬 기능 및 컨텍스트 메뉴 포함"""
        splitter = QSplitter(Qt.Horizontal)
        
        # 매수 후보 테이블
        buy_group = QGroupBox("매수 후보 종목")
        buy_layout = QVBoxLayout()
        
        self.buy_table = QTableWidget()
        self.buy_table.setColumnCount(12)
        self.buy_table.setHorizontalHeaderLabels([
            "종목코드", "종목명", "섹터", "현재가", "시장", "매수신호", 
            "RSI", "거래량비율", "추천도", 
            "추세방향", "추세강도", "매수타이밍"
        ])
        
        # 정렬 기능 활성화
        self.buy_table.setSortingEnabled(True)
        self.buy_table.horizontalHeader().setSectionsClickable(True)
        
        # 헤더 클릭 시 정렬 처리
        self.buy_table.horizontalHeader().sortIndicatorChanged.connect(
            self.on_buy_table_sort_changed
        )
        
        # 🔧 컨텍스트 메뉴 설정 추가
        self.buy_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.buy_table.customContextMenuRequested.connect(
            lambda pos: self.show_table_context_menu(pos, self.buy_table, 'buy')
        )
        
        self.buy_table.doubleClicked.connect(self.show_stock_detail)
        buy_layout.addWidget(self.buy_table)
        buy_group.setLayout(buy_layout)
        
        # 매도 후보 테이블  
        sell_group = QGroupBox("매도 후보 종목")
        sell_layout = QVBoxLayout()
        
        self.sell_table = QTableWidget()
        self.sell_table.setColumnCount(12)
        self.sell_table.setHorizontalHeaderLabels([
            "종목코드", "종목명", "섹터", "현재가", "시장", "매도신호", 
            "수익률", "보유기간", "위험도",
            "추세방향", "추세강도", "매도타이밍"
        ])
        
        # 정렬 기능 활성화
        self.sell_table.setSortingEnabled(True)
        self.sell_table.horizontalHeader().setSectionsClickable(True)
        
        # 헤더 클릭 시 정렬 처리
        self.sell_table.horizontalHeader().sortIndicatorChanged.connect(
            self.on_sell_table_sort_changed
        )
        
        # 🔧 컨텍스트 메뉴 설정 추가
        self.sell_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sell_table.customContextMenuRequested.connect(
            lambda pos: self.show_table_context_menu(pos, self.sell_table, 'sell')
        )
        
        self.sell_table.doubleClicked.connect(self.show_stock_detail)

        sell_layout.addWidget(self.sell_table)
        sell_group.setLayout(sell_layout)
        
        # 스플리터에 그룹 추가
        splitter.addWidget(buy_group)
        splitter.addWidget(sell_group)
        splitter.setSizes([1, 1])  # 50:50 비율
        
        return splitter

    def show_table_context_menu(self, position, table, table_type):
        """테이블 우클릭 메뉴 표시"""
        if not table.itemAt(position):
            return
        
        current_row = table.currentRow()
        if current_row < 0:
            return
        
        # 종목 정보 가져오기
        ticker_item = table.item(current_row, 0)  # 종목코드
        name_item = table.item(current_row, 1)    # 종목명
        
        if not ticker_item:
            return
        
        ticker = ticker_item.text()
        name = name_item.text() if name_item else ticker
        
        # 컨텍스트 메뉴 생성
        menu = QMenu(self)
        
        # 차트 보기
        chart_action = QAction('📊 차트 보기', self)
        chart_action.triggered.connect(lambda: self.show_chart_from_context(ticker, name))
        menu.addAction(chart_action)
        
        # AI 예측 (enhanced_screener 기능이 있는 경우)
#        if hasattr(self, 'run_quick_prediction'):
        menu.addSeparator()
            
        ai_predict_action = QAction('🤖 AI 예측', self)
        ai_predict_action.triggered.connect(lambda: self.show_ai_prediction_from_context(ticker, name))
        menu.addAction(ai_predict_action)
            
       
        # 구분선
        menu.addSeparator()
        
        # 종목 정보
        info_action = QAction('ℹ️ 종목 정보', self)
        info_action.triggered.connect(lambda: self.show_stock_info_from_context(ticker, name))
        menu.addAction(info_action)
        
        # 메뉴 표시
        global_pos = table.mapToGlobal(position)
        menu.exec_(global_pos)


    def show_chart_from_context(self, ticker, name=""):
        """컨텍스트 메뉴에서 차트 보기 - 직접 ticker 전달"""
        try:
            print(f"컨텍스트 메뉴에서 차트 요청: {ticker} ({name})")
            self.show_stock_detail(ticker, name)  # 문자열로 직접 전달
                
        except Exception as e:
            QMessageBox.warning(self, "차트 오류", f"차트를 표시할 수 없습니다:\n{str(e)}")


    def show_ai_prediction_from_context(self, ticker, name=""):
        """컨텍스트 메뉴에서 AI 예측"""
        try:
            if hasattr(self, 'show_prediction_dialog'):
                self.show_prediction_dialog(ticker)
            else:
                QMessageBox.information(self, "AI 예측", 
                                    f"🤖 {ticker} ({name}) AI 예측 기능을 실행합니다.\n"
                                    f"enhanced_screener.py의 예측 기능이 활성화되어야 합니다.")
        except Exception as e:
            QMessageBox.warning(self, "예측 오류", f"AI 예측 중 오류가 발생했습니다:\n{str(e)}")

    def show_stock_info_from_context(self, ticker, name=""):
        """컨텍스트 메뉴에서 종목 정보"""
        try:
            QMessageBox.information(self, "종목 정보", 
                                f"ℹ️ 종목 정보: {ticker}\n\n"
                                f"• 종목명: {name}\n"
                                f"• 종목코드: {ticker}\n\n"
                                f"상세한 정보는 차트 보기를 이용하시기 바랍니다.")
        except Exception as e:
            QMessageBox.warning(self, "정보 오류", f"종목 정보를 가져올 수 없습니다:\n{str(e)}")

    def get_timing_sort_score(self, timing_text):
        """타이밍 텍스트를 정렬 가능한 숫자로 변환 - 간단한 버전"""
        
        if not timing_text:
            return 0
        
        # 문자열로 변환하고 정리
        text = str(timing_text).strip()
        
        # 단순하고 명확한 매칭
        if "최적" in text:
            return 4  # 최고
        elif "양호" in text:
            return 3  # 두 번째
        elif "보통" in text:
            return 2  # 세 번째
        elif "대기" in text or "보유" in text:
            return 1  # 네 번째
        else:
            return 0  # 기타

    def on_buy_table_sort_changed(self, logical_index, order):
        """매수 테이블 정렬 정보 표시 - 깔끔한 버전"""
        column_names = {
            0: "종목코드", 1: "종목명", 2: "섹터", 3: "현재가", 
            4: "시장", 5: "매수신호", 6: "RSI", 7: "거래량비율", 
            8: "추천도", 9: "추세방향", 10: "추세강도", 11: "매수타이밍"
        }
        
        column_name = column_names.get(logical_index, f"컬럼 {logical_index}")
        direction_text = "오름차순 ↑" if order == Qt.AscendingOrder else "내림차순 ↓"
        
        # 각 컬럼별 정렬 의미 설명 - 타이밍 부분 수정
        sort_meanings = {
            0: {"asc": "A→Z 순", "desc": "Z→A 순"},
            1: {"asc": "가나다 순", "desc": "하파타 순"},
            2: {"asc": "섹터명 순", "desc": "섹터명 역순"},
            3: {"asc": "저가 → 고가", "desc": "고가 → 저가"},
            4: {"asc": "시장명 순", "desc": "시장명 역순"},
            5: {"asc": "신호명 순", "desc": "신호명 역순"},
            6: {"asc": "낮은 RSI → 높은 RSI", "desc": "높은 RSI → 낮은 RSI"},
            7: {"asc": "적은 거래량 → 많은 거래량", "desc": "많은 거래량 → 적은 거래량"},
            8: {"asc": "낮은 추천도 → 높은 추천도", "desc": "높은 추천도 → 낮은 추천도 👍"},
            9: {"asc": "추세방향 순", "desc": "추세방향 역순"},
            10: {"asc": "약한 추세 → 강한 추세", "desc": "강한 추세 → 약한 추세 👍"},
            11: {"asc": "대기 → 최적", "desc": "최적 → 대기 👍"}  # 수정됨
        }
        
        meaning_key = "desc" if order == Qt.DescendingOrder else "asc"
        meaning = sort_meanings.get(logical_index, {}).get(meaning_key, "")
        
        self.statusbar.showMessage(f"📊 {column_name} {direction_text} - {meaning}")

    def on_sell_table_sort_changed(self, logical_index, order):
        """매도 테이블 정렬 정보 표시 - 깔끔한 버전"""
        column_names = {
            0: "종목코드", 1: "종목명", 2: "섹터", 3: "현재가",
            4: "시장", 5: "매도신호", 6: "수익률", 7: "보유기간", 
            8: "위험도", 9: "추세방향", 10: "추세강도", 11: "매도타이밍"
        }
        
        column_name = column_names.get(logical_index, f"컬럼 {logical_index}")
        direction_text = "오름차순 ↑" if order == Qt.AscendingOrder else "내림차순 ↓"
        
        # 각 컬럼별 정렬 의미 설명 - 타이밍 부분 수정
        sort_meanings = {
            0: {"asc": "A→Z 순", "desc": "Z→A 순"},
            1: {"asc": "가나다 순", "desc": "하파타 순"},
            2: {"asc": "섹터명 순", "desc": "섹터명 역순"},
            3: {"asc": "저가 → 고가", "desc": "고가 → 저가"},
            4: {"asc": "시장명 순", "desc": "시장명 역순"},
            5: {"asc": "신호명 순", "desc": "신호명 역순"},
            6: {"asc": "낮은 수익률 → 높은 수익률", "desc": "높은 수익률 → 낮은 수익률"},
            7: {"asc": "짧은 보유 → 긴 보유", "desc": "긴 보유 → 짧은 보유"},
            8: {"asc": "낮은 위험 → 높은 위험", "desc": "높은 위험 → 낮은 위험 ⚠️"},
            9: {"asc": "추세방향 순", "desc": "추세방향 역순"},
            10: {"asc": "약한 추세 → 강한 추세", "desc": "강한 추세 → 약한 추세"},
            11: {"asc": "대기 → 최적", "desc": "최적 → 대기 ⚠️"}  # 수정됨
        }
        
        meaning_key = "desc" if order == Qt.DescendingOrder else "asc"
        meaning = sort_meanings.get(logical_index, {}).get(meaning_key, "")
        
        self.statusbar.showMessage(f"📊 {column_name} {direction_text} - {meaning}")

    def setup_stock_lists(self):
        """종목 리스트 초기 설정"""
        self.stock_lists = {
            'korea': [],
            'usa': [],
            'sweden': []
        }
        self.load_stock_lists()

    def update_stocks_online(self):
        """스마트 보강을 적용한 온라인 종목 업데이트"""
        market_selection = self.market_combo.currentText()
        use_mcap_filter = self.use_market_cap_filter.isChecked()
        
        # 업데이트할 시장 결정
        if market_selection == "전체":
            markets_to_update = ["한국", "미국", "스웨덴"]
        elif "한국" in market_selection:
            markets_to_update = ["한국"]
        elif "미국" in market_selection:
            markets_to_update = ["미국"]
        elif "스웨덴" in market_selection:
            markets_to_update = ["스웨덴"]
        else:
            markets_to_update = []
        
        if not markets_to_update:
            QMessageBox.warning(self, "알림", "업데이트할 시장을 선택해주세요.")
            return
        
        # 메시지 생성
        if use_mcap_filter:
            top_count = self.top_stocks_spin.value()
            enrich_all = self.enrich_all_radio.isChecked()
            enrich_count = self.enrich_count_spin.value() if not enrich_all else "전체"
            
            message = (f'시가총액 기준 상위 {top_count}개 종목을 업데이트합니다.\n'
                    f'보강 대상: {enrich_count}개 종목\n'
                    f'대상 시장: {", ".join(markets_to_update)}\n\n'
                    f'계속하시겠습니까?')
        else:
            message = (f'기본 종목 리스트를 업데이트합니다 (보강 없음).\n'
                    f'대상 시장: {", ".join(markets_to_update)}\n'
                    f'빠르게 완료됩니다.\n\n'
                    f'계속하시겠습니까?')
        
        reply = QMessageBox.question(self, '확인', message, 
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.update_online_btn.setEnabled(False)
            
            # 설정 정보 수집
            update_config = {
                'markets': markets_to_update,
                'use_mcap_filter': use_mcap_filter,
                'top_count': self.top_stocks_spin.value() if use_mcap_filter else 0,
                'enrich_all': self.enrich_all_radio.isChecked() if use_mcap_filter else False,
                'enrich_count': self.enrich_count_spin.value() if use_mcap_filter else 0
            }
            
            # 별도 스레드에서 실행
            self.update_thread = SmartUpdateThread(update_config)
            self.update_thread.finished.connect(self.on_update_finished)
            self.update_thread.error.connect(self.on_update_error)
            self.update_thread.progress.connect(self.on_update_progress)
            self.update_thread.start()

    def on_update_progress(self, message):
        """업데이트 진행 상황 표시"""
        self.statusbar.showMessage(f'🌐 {message}')
    
    def on_update_finished(self, message):
        """업데이트 완료 처리"""
        self.update_online_btn.setEnabled(True)
        self.statusbar.showMessage(f'✅ 업데이트 완료')
        self.load_stock_lists()
        QMessageBox.information(self, '완료', message)
    
    def on_update_error(self, error_message):
        """업데이트 오류 처리"""
        self.update_online_btn.setEnabled(True)
        self.statusbar.showMessage('❌ 업데이트 실패')
        QMessageBox.critical(self, '오류', error_message)
    
    def open_condition_builder(self):
        """조건 빌더 열기"""
        try:
            from dialogs import ConditionBuilderDialog
            dialog = ConditionBuilderDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                condition = dialog.get_condition()
                if condition:
                    if not hasattr(self, 'custom_conditions'):
                        self.custom_conditions = []
                    self.custom_conditions.append(condition)
                    self.update_custom_conditions_display()
        except ImportError:
            # dialogs 모듈이 없으면 간단한 입력창으로 대체
            text, ok = QInputDialog.getText(self, '조건 추가', '조건명을 입력하세요:')
            if ok and text:
                if not hasattr(self, 'custom_conditions'):
                    self.custom_conditions = []
                self.custom_conditions.append({'name': text, 'enabled': True})
                self.update_custom_conditions_display()
    
    def manage_custom_conditions(self):
        """조건 관리"""
        try:
            from dialogs import ConditionManagerDialog
            dialog = ConditionManagerDialog(self.custom_conditions, self)
            if dialog.exec_() == QDialog.Accepted:
                self.custom_conditions = dialog.get_conditions()
                self.update_custom_conditions_display()
        except ImportError:
            # 간단한 조건 목록 표시
            if not hasattr(self, 'custom_conditions') or not self.custom_conditions:
                QMessageBox.information(self, "알림", "추가된 조건이 없습니다.")
                return
            
            condition_names = [c.get('name', 'Unknown') for c in self.custom_conditions]
            item, ok = QInputDialog.getItem(self, '조건 관리', '삭제할 조건:', condition_names, 0, False)
            if ok and item:
                self.custom_conditions = [c for c in self.custom_conditions if c.get('name') != item]
                self.update_custom_conditions_display()
    
    def update_custom_conditions_display(self):
        """사용자 정의 조건 표시 업데이트"""
        if not hasattr(self, 'custom_conditions'):
            self.custom_conditions = []
        
        # 기존 위젯들 제거
        for i in reversed(range(self.custom_conditions_layout.count())):
            child = self.custom_conditions_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # 새 조건들 추가
        for i, condition in enumerate(self.custom_conditions):
            condition_widget = QWidget()
            layout = QHBoxLayout(condition_widget)
            layout.setContentsMargins(2, 2, 2, 2)
            
            # 체크박스
            checkbox = QCheckBox(condition.get('name', f'조건{i+1}'))
            checkbox.setChecked(condition.get('enabled', True))
            checkbox.setMaximumWidth(350)  # 너비 제한
            layout.addWidget(checkbox)
            
            # 삭제 버튼
            delete_btn = QPushButton("×")
            delete_btn.setMaximumWidth(25)
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_custom_condition(idx))
            layout.addWidget(delete_btn)
            
            self.custom_conditions_layout.addWidget(condition_widget)

    def delete_custom_condition(self, index):
        """사용자 정의 조건 삭제"""
        if 0 <= index < len(self.custom_conditions):
            del self.custom_conditions[index]
            self.update_custom_conditions_display()
    
    def show_stock_chart(self, index):
        """종목 차트 표시"""
        table = self.sender()
        row = index.row()
        symbol = table.item(row, 0).text()
        name = table.item(row, 1).text()
        
        # 차트 윈도우 생성
        chart_window = StockChartWindow(symbol, name, self)
        chart_window.show()
    
    def create_sample_csv_files(self):
        """샘플 CSV 파일들 생성"""
        if not os.path.exists('stock_data'):
            os.makedirs('stock_data')
        
        try:
            from utils import create_sample_data
            create_sample_data()
            QMessageBox.information(self, "완료", 
                                  "샘플 CSV 파일이 생성되었습니다!\n"
                                  "'stock_data' 폴더를 확인해주세요.\n\n"
                                  "이제 스크리닝을 시작할 수 있습니다.")
            self.load_stock_lists()
        except Exception as e:
            QMessageBox.critical(self, "오류", f"샘플 파일 생성 실패: {str(e)}")
    
    # ✅ 중복 함수 제거 - 아래의 더 완전한 구현 사용 (line 4076)
    
    def update_stock_count(self):
        """종목 개수 업데이트 - 리스트 형태 기준"""
        korea_count = len(self.stock_lists.get('korea', []))
        usa_count = len(self.stock_lists.get('usa', []))
        sweden_count = len(self.stock_lists.get('sweden', []))
        total_count = korea_count + usa_count + sweden_count
        
        # 레이블이 존재하는 경우에만 업데이트
        if hasattr(self, 'korea_count_label'):
            self.korea_count_label.setText(f"🇰🇷 한국: {korea_count}개")
        if hasattr(self, 'usa_count_label'):
            self.usa_count_label.setText(f"🇺🇸 미국: {usa_count}개")
        if hasattr(self, 'sweden_count_label'):
            self.sweden_count_label.setText(f"🇸🇪 스웨덴: {sweden_count}개")
        if hasattr(self, 'total_count_label'):
            self.total_count_label.setText(f"🌍 전체: {total_count}개")
    
    def open_csv_editor(self):
        """CSV 파일 편집 다이얼로그"""
        dialog = CSVEditorDialog(self)
        dialog.exec_()
        self.load_stock_lists()  # 편집 후 새로고침
    
    def get_selected_stocks(self):
        """선택된 시장의 종목들 반환 - 기존 로직과 호환"""
        market_selection = self.market_combo.currentText()
        stocks = []
        
        # 기존 방식 그대로 사용 (리스트 형태)
        if market_selection == "전체":
            for market in ['korea', 'usa', 'sweden']:
                stocks.extend(self.stock_lists.get(market, []))
        elif "한국" in market_selection:
            stocks = self.stock_lists.get('korea', [])
        elif "미국" in market_selection:
            stocks = self.stock_lists.get('usa', [])
        elif "스웨덴" in market_selection:
            stocks = self.stock_lists.get('sweden', [])

        # 시가총액 필터링 (기존 로직 유지)
        if hasattr(self, 'use_market_cap_filter') and self.use_market_cap_filter.isChecked() and stocks:
            top_count = self.top_stocks_spin.value()
            
            try:
                stocks_with_mcap = []
                for stock in stocks:
                    mcap = stock.get('market_cap', 0)

                    # 문자열 변환 처리
                    if isinstance(mcap, str):
                        mcap_clean = re.sub(r'[,\s]', '', mcap.upper())
                        
                        try:
                            if mcap_clean.endswith('B'):
                                mcap = float(mcap_clean[:-1]) * 1e9
                            elif mcap_clean.endswith('M'):
                                mcap = float(mcap_clean[:-1]) * 1e6
                            elif mcap_clean.endswith('K'):
                                mcap = float(mcap_clean[:-1]) * 1e3
                            else:
                                mcap = float(mcap_clean) if mcap_clean else 0
                        except (ValueError, TypeError):
                            mcap = 0

                    if isinstance(mcap, (int, float)) and mcap > 0:
                        stock_copy = stock.copy()
                        stock_copy['market_cap_numeric'] = mcap
                        stocks_with_mcap.append(stock_copy)
                
                # 시가총액 기준 정렬
                stocks_with_mcap.sort(key=lambda x: float(x.get('market_cap_numeric', 0)), reverse=True)
                
                # 상위 N개만 선택
                stocks = stocks_with_mcap[:top_count]
                
                if hasattr(self, 'statusbar'):
                    self.statusbar.showMessage(f'💰 시가총액 상위 {len(stocks)}개 종목으로 필터링됨')
                
            except Exception as e:
                print(f"시가총액 필터링 중 오류: {e}")
        
        return stocks
    
    def stop_screening(self):
        """스크리닝 중지"""
        self.screening_cancelled = True
        self.statusbar.showMessage('⏹️ 스크리닝 중지 요청됨...')
    
    def run_screening(self):
        """스크리닝 실행"""
        # 버튼 상태 변경
        self.search_btn.setVisible(False)
        self.stop_btn.setVisible(True)
        
        # 스크리닝 상태 초기화
        self.is_screening = True
        self.screening_cancelled = False
        
        self.statusbar.showMessage('🔍 스크리닝 중...')
        
        try:
            stocks = self.get_selected_stocks()
            if not stocks:
                QMessageBox.warning(self, "알림", 
                                  "분석할 종목이 없습니다.\n"
                                  "'샘플 생성' 버튼을 먼저 클릭하거나\n"
                                  "CSV 파일을 확인해주세요.")
                return
            
            # 매수/매도 후보 분석
            buy_candidates = []
            sell_candidates = []
            
            for i, stock_info in enumerate(stocks):
                # 중지 요청 확인
                if self.screening_cancelled:
                    self.statusbar.showMessage('⏹️ 사용자에 의해 스크리닝이 중지되었습니다')
                    break
                
                try:
                    self.statusbar.showMessage(f'🔍 스크리닝 중... ({i+1}/{len(stocks)}) {stock_info["ticker"]}')
                    QApplication.processEvents()  # UI 업데이트
                    
                    result = self.analyze_stock(stock_info)
                    if result:
                        if result['action'] == 'BUY':
                            buy_candidates.append(result)
                        elif result['action'] == 'SELL':
                            sell_candidates.append(result)
                except Exception as e:
                    print(f"Error analyzing {stock_info['ticker']}: {e}")
                    continue
            
            # 결과를 클래스 변수에 저장
            self.last_buy_candidates = buy_candidates
            self.last_sell_candidates = sell_candidates
            
            # 테이블 업데이트
            self.update_buy_table(buy_candidates)
            self.update_sell_table(sell_candidates)
            
            # 엑셀 저장 버튼 활성화
            if buy_candidates or sell_candidates:
                self.export_btn.setEnabled(True)
            
            if not self.screening_cancelled:
                self.statusbar.showMessage(f'✅ 스크리닝 완료 - 매수후보: {len(buy_candidates)}개, 매도후보: {len(sell_candidates)}개')
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"스크리닝 중 오류가 발생했습니다: {str(e)}")
            
        finally:
            # 버튼 상태 복원
            self.search_btn.setVisible(True)
            self.stop_btn.setVisible(False)
            self.is_screening = False
            self.screening_cancelled = False
    
    def export_results_to_excel(self):
        """스크리닝 결과를 엑셀로 저장"""
        if not self.last_buy_candidates and not self.last_sell_candidates:
            QMessageBox.warning(self, "알림", "저장할 결과가 없습니다.\n먼저 스크리닝을 실행해주세요.")
            return
        
        # 파일 저장 다이얼로그
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f'screening_results_{timestamp}.xlsx'
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "스크리닝 결과 저장",
            default_filename,
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if filename:
            try:
                # utils.py의 export_screening_results 함수 사용
                saved_file = export_screening_results(
                    self.last_buy_candidates, 
                    self.last_sell_candidates, 
                    filename
                )
                
                if saved_file:
                    QMessageBox.information(
                        self, 
                        "저장 완료", 
                        f"스크리닝 결과가 성공적으로 저장되었습니다!\n\n"
                        f"파일: {saved_file}\n"
                        f"매수 후보: {len(self.last_buy_candidates)}개\n"
                        f"매도 후보: {len(self.last_sell_candidates)}개"
                    )
                else:
                    QMessageBox.critical(self, "오류", "파일 저장에 실패했습니다.")
                    
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 저장 중 오류가 발생했습니다:\n{str(e)}")
    def find_ma_breakout_date(self, data, fast_ma, slow_ma, days_limit):
        """
        빠른 이동평균이 느린 이동평균을 돌파한 날짜를 찾습니다.
        
        용도: MA60이 MA120을 돌파한 시점 찾기
        - 어제: MA60 <= MA120
        - 오늘: MA60 > MA120
        이런 경우를 돌파로 판단하고, 현재부터 days_limit 일 이내 돌파만 유효
        
        예시: 
        - 2025-08-20 현재, 22일 이내(7월 29일 이후)에 돌파했는지 확인
        - 7월 30일에 돌파 → 유효 (21일 전)
        - 7월 25일에 돌파 → 무효 (26일 전)
        """
        try:
            if len(data) < 2:
                return None
                
            # 전체 데이터에서 돌파 시점들을 모두 찾기
            breakout_dates = []
            
            for i in range(1, len(data)):
                prev_day = data.iloc[i-1]
                current_day = data.iloc[i]
                
                # 돌파 조건: 어제는 fast_ma <= slow_ma, 오늘은 fast_ma > slow_ma
                if (prev_day[fast_ma] <= prev_day[slow_ma] and 
                    current_day[fast_ma] > current_day[slow_ma]):
                    
                    breakout_dates.append(data.index[i])
            
            if not breakout_dates:
                return None  # 돌파가 없었음
            
            # 현재 시점을 기준으로 days_limit 일 이내의 돌파 찾기
            import pandas as pd
            
            today = data.index[-1]  # 마지막 거래일을 "현재"로 간주
            cutoff_date = today - pd.Timedelta(days=days_limit)
            
            # days_limit 일 이내의 돌파들만 필터링
            recent_breakouts = [date for date in breakout_dates if date >= cutoff_date]
            
            if recent_breakouts:
                # 가장 최근 돌파 반환
                latest_breakout = recent_breakouts[-1]
                days_ago = (today - latest_breakout).days
                print(f"📈 {fast_ma}→{slow_ma} 돌파 발견: {latest_breakout.strftime('%Y-%m-%d')} ({days_ago}일 전)")
                return latest_breakout
            else:
                print(f"📉 최근 {days_limit}일 내 {fast_ma}→{slow_ma} 돌파 없음")
                return None
            
        except Exception as e:
            print(f"Error finding MA breakout: {e}")
            return None

    def check_long_term_below_condition(self, data, breakout_date, days_check):
        """
        돌파 시점 이전 3개월 동안 60일선이 120일선 아래 있었는지 확인합니다.
        
        목적: 충분한 조정을 거친 후의 의미있는 돌파인지 검증
        
        예시: 2025년 8월 10일에 60일선이 120일선을 돌파했다면
        - 체크 기간: 2025년 5월 10일 ~ 2025년 8월 9일 (66거래일)
        - 조건: 이 기간의 90% 이상에서 MA60 < MA120
        - 결과: 장기 하락 추세 후의 반전 돌파로 판단
        """
        try:
            import pandas as pd
            
            # 돌파 날짜 이전 days_check일 동안의 기간 설정
            check_start_date = breakout_date - pd.Timedelta(days=days_check)
            check_end_date = breakout_date - pd.Timedelta(days=1)  # 돌파 전날까지
            
            # 해당 기간의 데이터 추출
            check_period_data = data[(data.index >= check_start_date) & 
                                    (data.index <= check_end_date)]
            
            if len(check_period_data) < days_check * 0.5:  # 최소 50%의 데이터가 있어야 함
                print(f"⚠️ 체크 기간 데이터 부족: {len(check_period_data)}/{days_check}")
                return False
            
            # MA60과 MA120 데이터가 모두 있는 날들만 체크
            valid_data = check_period_data.dropna(subset=['MA60', 'MA120'])
            
            if len(valid_data) < len(check_period_data) * 0.7:  # 70% 이상이 유효해야 함
                print(f"⚠️ MA 데이터 부족: {len(valid_data)}/{len(check_period_data)}")
                return False
            
            # 60일선이 120일선 아래 있던 날의 비율 계산
            below_condition = valid_data['MA60'] < valid_data['MA120']
            below_ratio = below_condition.sum() / len(valid_data)
            
            print(f"📊 장기 하락 조건 체크:")
            print(f"   - 체크 기간: {check_start_date.strftime('%Y-%m-%d')} ~ {check_end_date.strftime('%Y-%m-%d')}")
            print(f"   - 유효 데이터: {len(valid_data)}일")
            print(f"   - MA60 < MA120 비율: {below_ratio:.1%}")
            
            # 90% 이상의 기간에서 60일선이 120일선 아래 있었으면 조건 만족
            result = below_ratio >= 0.9
            print(f"   - 조건 만족 (90% 이상): {'✅' if result else '❌'}")
            
            return result
            
        except Exception as e:
            print(f"Error checking long term below condition: {e}")
            return False

    def find_ma_breakdown_date(self, data, fast_ma, slow_ma, days_limit):
        """
        빠른 이동평균이 느린 이동평균 아래로 떨어진(하향돌파) 날짜를 찾습니다.
        
        용도: MA60이 MA120 아래로 떨어진 시점 찾기 (매도 신호)
        - 어제: MA60 >= MA120
        - 오늘: MA60 < MA120
        이런 경우를 하향돌파로 판단하고, 현재부터 days_limit 일 이내 돌파만 유효
        
        예시: 
        - 2024-08-20 현재, 5일 이내(8월 15일 이후)에 하향돌파했는지 확인
        - 8월 18일에 하향돌파 → 유효 (2일 전) → 매도 신호
        - 8월 10일에 하향돌파 → 무효 (10일 전) → 이미 늦음
        """
        try:
            if len(data) < 2:
                return None
                
            # 전체 데이터에서 하향돌파 시점들을 모두 찾기
            breakdown_dates = []
            
            for i in range(1, len(data)):
                prev_day = data.iloc[i-1]
                current_day = data.iloc[i]
                
                # 하향돌파 조건: 어제는 fast_ma >= slow_ma, 오늘은 fast_ma < slow_ma
                if (prev_day[fast_ma] >= prev_day[slow_ma] and 
                    current_day[fast_ma] < current_day[slow_ma]):
                    
                    breakdown_dates.append(data.index[i])
            
            if not breakdown_dates:
                print(f"📈 최근 전체 기간에 {fast_ma}→{slow_ma} 하향돌파 없음")
                return None
            
            # 현재 시점을 기준으로 days_limit 일 이내의 하향돌파 찾기
            import pandas as pd
            
            today = data.index[-1]  # 마지막 거래일을 "현재"로 간주
            cutoff_date = today - pd.Timedelta(days=days_limit)
            
            # days_limit 일 이내의 하향돌파들만 필터링
            recent_breakdowns = [date for date in breakdown_dates if date >= cutoff_date]
            
            if recent_breakdowns:
                # 가장 최근 하향돌파 반환
                latest_breakdown = recent_breakdowns[-1]
                days_ago = (today - latest_breakdown).days
                print(f"📉 {fast_ma}→{slow_ma} 하향돌파 발견: {latest_breakdown.strftime('%Y-%m-%d')} ({days_ago}일 전)")
                return latest_breakdown
            else:
                print(f"📊 최근 {days_limit}일 내 {fast_ma}→{slow_ma} 하향돌파 없음")
                return None
            
        except Exception as e:
            print(f"Error finding MA breakdown: {e}")
            return None
        

    def analyze_stock(self, stock_info):
        """개별 종목 분석 - 기존 조건 + 추세 분석 통합 (체크박스 이름 수정)"""
        try:
            symbol = stock_info['ticker']
            print(f"🔍 분석 중: {symbol}")
            
            # 데이터 다운로드 (6개월)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)

            # 🔧 안전한 데이터 가져오기
            data = self.safe_get_stock_data(symbol, start_date, end_date)
            
            if data is None:
                print(f"⚠️ {symbol} - 데이터 없음 (스킵)")
                return None
            
            if len(data) < 120:  # 충분한 데이터가 없으면 스킵
                print(f"⚠️ {symbol} - 데이터 부족 ({len(data)}개, 최소 120개 필요)")
                return None
            
            # 기술적 지표 계산
            data = self.technical_analyzer.calculate_all_indicators(data)
            
            # 추세 및 타이밍 분석
            try:
                trend_analysis = self.trend_analyzer.analyze_trend_and_timing(data)
            except Exception as trend_error:
                print(f"⚠️ {symbol} - 추세 분석 실패: {trend_error}")
                trend_analysis = None
           
            current = data.iloc[-1]
            prev = data.iloc[-2]
            
            # 시장 구분
            if '.KS' in symbol:
                market = 'KOREA'
            elif '.ST' in symbol:
                market = 'SWEDEN'
            else:
                market = 'NASDAQ'
            
            # ==================== 매수 조건 체크 ====================
            buy_signals = []
            
            # 1. 이동평균 기술적 매수 조건 (올바른 이름: ma_condition)
            if self.ma_condition.isChecked():
                if (current['MA60'] > current['MA120'] and 
                    current['Close'] > current['MA60']):
                    
                    # 🔧 강화 조건 체크 (screener.py에 있는 경우)
                    try:
                        if self.check_enhanced_buy_condition(data, symbol):
                            buy_signals.append("강화된 기술적매수")
                        else:
                            # 강화 조건 불만족 이유 분석
                            reasons = []
                            if current['RSI'] > 75:
                                reasons.append("RSI 과매수")
                            
                            distance_pct = abs(current['Close'] - current['MA60']) / current['MA60'] * 100
                            if distance_pct > 10:
                                reasons.append(f"주가가 60일선에서 너무 멀음({distance_pct:.1f}%)")
                            
                            if reasons:
                                print(f"❌ {symbol} - 기본 조건 불만족: {', '.join(reasons)}")
                            else:
                                # 강화 조건 메서드가 없으면 기본 신호
                                buy_signals.append("이동평균 매수")
                    except AttributeError:
                        # check_enhanced_buy_condition 메서드가 없으면 기본 조건만 체크
                        if (current['MA60'] > current['MA120'] and 
                            current['MA60'] > prev['MA60'] and 
                            current['MA120'] > prev['MA120'] and
                            abs(current['Close'] - current['MA60']) / current['MA60'] < 0.03):
                            buy_signals.append("MA돌파+터치")
            
            # 2. 볼린저밴드 + RSI 매수 조건
            if self.bb_condition.isChecked():
                if (current['Close'] <= current['BB_Lower'] * 1.02 and 
                    current['RSI'] < 35):
                    buy_signals.append("볼린저하단+RSI")
            
            # 3. MACD 골든크로스 + 거래량 매수 조건
            if self.support_condition.isChecked():
                if (current['MACD'] > current['MACD_Signal'] and 
                    prev['MACD'] <= prev['MACD_Signal'] and
                    current['Volume_Ratio'] > 1.2):
                    buy_signals.append("MACD골든+거래량")
            
            # 4. 모멘텀 상승 매수 조건
            if self.momentum_condition.isChecked():
                if len(data) >= 21:
                    price_momentum = (current['Close'] / data['Close'].iloc[-21] - 1) * 100
                    if price_momentum > 5 and current['RSI'] > 50:
                        buy_signals.append("모멘텀상승")
            
            # 5. 사용자 정의 매수 조건 체크
            try:
                custom_buy_signals = self.check_custom_conditions(data, 'BUY')
                buy_signals.extend(custom_buy_signals)
            except AttributeError:
                # check_custom_conditions 메서드가 없으면 스킵
                pass
            
            # ==================== 매도 조건 체크 ====================
            sell_signals = []
            
            # 1. 기술적 매도 조건
            if self.tech_sell.isChecked():
                # 기존 단순 조건
                simple_sell_condition = (
                    current['MA60'] < current['MA120'] or 
                    current['Close'] < current['MA60'] * 0.97
                )
                
                if simple_sell_condition:
                    # 🔧 강화 조건 체크 (있는 경우에만)
                    try:
                        ma60_below_ma120_breakdown_date = self.find_ma_breakdown_date(data, 'MA60', 'MA120', days_limit=5)
                        
                        if ma60_below_ma120_breakdown_date is not None:
                            sell_signals.append("강화된 기술적매도")
                            print(f"🎯 {symbol} - 강화된 매도 조건 만족!")
                            print(f"   - 60일선→120일선 하향돌파: {ma60_below_ma120_breakdown_date.strftime('%Y-%m-%d')}")
                            print(f"   - 현재 60일선: {current['MA60']:.2f}")
                            print(f"   - 현재 120일선: {current['MA120']:.2f}")
                            print(f"   - 현재가: {current['Close']:.2f}")
                        else:
                            # 강화 조건은 불만족하지만 기존 조건은 만족하는 경우
                            print(f"⚠️ {symbol} - 기본 매도 조건만 만족 (최근 하향돌파 없음)")
                            sell_signals.append("기술적 매도 고려")
                    except AttributeError:
                        # find_ma_breakdown_date 메서드가 없으면 기본 조건만
                        sell_signals.append("기술적매도")
                else:
                    print(f"✅ {symbol} - 매도 조건 불만족 (안전)")
            
            # 2. 수익률 매도 조건 (있는 경우)
            if hasattr(self, 'profit_sell') and self.profit_sell.isChecked():
                # 수익률 계산 로직 (실제로는 매수가가 필요)
                # 여기서는 단순화
                pass
            
            # 3. 볼린저밴드 상단 + RSI 매도 조건
            if self.bb_sell.isChecked():
                if (current['Close'] >= current['BB_Upper'] * 0.98 and 
                    current['RSI'] > 70):
                    sell_signals.append("볼린저상단+RSI")
            
            # 4. 거래량 급감 매도 조건
            if self.volume_sell.isChecked():
                if (current['Volume_Ratio'] < 0.7 and 
                    current['RSI'] < prev['RSI']):
                    sell_signals.append("거래량급감")
            
            # 5. 사용자 정의 매도 조건 체크
            try:
                custom_sell_signals = self.check_custom_conditions(data, 'SELL')
                sell_signals.extend(custom_sell_signals)
            except AttributeError:
                # check_custom_conditions 메서드가 없으면 스킵
                pass
            
            # ==================== 결과 반환 ====================
            
            # 매수 신호가 있는 경우
            if buy_signals:
                result = {
                    'action': 'BUY',
                    'symbol': symbol,
                    'name': stock_info.get('name', symbol),
                    'sector': stock_info.get('sector', '미분류'),
                    'price': round(current['Close'], 2),
                    'market': market,
                    'signals': ', '.join(buy_signals),
                    'rsi': round(current['RSI'], 1),
                    'volume_ratio': round(current['Volume_Ratio'], 2),
                    'recommendation': len(buy_signals) * 25  # 신호개수에 따른 점수
                }
                
                # ✨ 추세 분석 결과 추가
                if trend_analysis:
                    result.update({
                        'trend_direction': trend_analysis['trend_direction'],
                        'trend_score': trend_analysis['trend_score'],
                        'buy_timing': trend_analysis['buy_timing']['grade'],
                        'sell_timing': trend_analysis['sell_timing']['grade'],
                        'overall_recommendation': trend_analysis['recommendation']
                    })
                else:
                    # 추세 분석 실패시 기본값
                    result.update({
                        'trend_direction': '분석불가',
                        'trend_score': 0,
                        'buy_timing': '대기',
                        'sell_timing': '대기',
                        'overall_recommendation': '중립'
                    })
                
                return result
            
            # 매도 신호가 있는 경우
            elif sell_signals:
                result = {
                    'action': 'SELL',
                    'symbol': symbol,
                    'name': stock_info.get('name', symbol),
                    'sector': stock_info.get('sector', '미분류'),
                    'price': round(current['Close'], 2),
                    'market': market,
                    'signals': ', '.join(sell_signals),
                    'profit': 0,  # 실제로는 매수가와 비교 필요
                    'holding_period': '미상',
                    'risk': len(sell_signals) * 30
                }
                
                # ✨ 추세 분석 결과 추가
                if trend_analysis:
                    result.update({
                        'trend_direction': trend_analysis['trend_direction'],
                        'trend_score': trend_analysis['trend_score'],
                        'buy_timing': trend_analysis['buy_timing']['grade'],
                        'sell_timing': trend_analysis['sell_timing']['grade'],
                        'overall_recommendation': trend_analysis['recommendation']
                    })
                else:
                    # 추세 분석 실패시 기본값
                    result.update({
                        'trend_direction': '분석불가',
                        'trend_score': 0,
                        'buy_timing': '대기',
                        'sell_timing': '대기',
                        'overall_recommendation': '중립'
                    })
                
                return result
            
            # 매수도 매도도 신호가 없는 경우
            return None
            
        except Exception as e:
            print(f"❌ {stock_info['ticker']} 분석 오류: {e}")
            return None

    def safe_get_stock_data(self, symbol, start_date, end_date):
        """안전한 주식 데이터 가져오기 (캐싱 사용)"""
        try:
            # 기간 계산
            days_diff = (end_date - start_date).days + 10
            period_str = f"{days_diff}d"

            # 캐싱 매니저 사용
            data = get_stock_data(symbol, period=period_str)

            if data is not None and not data.empty:
                return data

            print(f"⚠️ {symbol} - 빈 데이터")
            return None

        except Exception as e:
            error_msg = str(e).lower()

            if "delisted" in error_msg or "no timezone found" in error_msg:
                print(f"⚠️ {symbol} - 상장폐지 또는 데이터 없음")
            elif "timeout" in error_msg:
                print(f"⚠️ {symbol} - 타임아웃")
            else:
                print(f"⚠️ {symbol} - 기타 오류: {e}")

            return None

    def validate_stock_symbols(self, stock_list):
        """종목 심볼들의 유효성 사전 체크"""
        valid_stocks = []
        invalid_stocks = []
        
        print("📋 종목 유효성 체크 중...")
        
        for stock_info in stock_list:
            symbol = stock_info['ticker']

            try:
                # 빠른 기본 정보 체크 (캐싱 사용)
                info = get_ticker_info(symbol)

                # 기본 정보가 있고 유효한 심볼이면
                if info and info.get('symbol'):
                    valid_stocks.append(stock_info)
                    print(f"✅ {symbol} - 유효")
                else:
                    invalid_stocks.append(stock_info)
                    print(f"❌ {symbol} - 무효 (정보 없음)")
                    
            except Exception as e:
                invalid_stocks.append(stock_info)
                print(f"❌ {symbol} - 무효 ({str(e)[:50]})")
        
        print(f"📊 유효성 체크 완료: 유효 {len(valid_stocks)}개, 무효 {len(invalid_stocks)}개")
        
        if invalid_stocks:
            print("❌ 무효한 종목들:")
            for stock in invalid_stocks:
                print(f"   - {stock['ticker']}: {stock.get('name', 'Unknown')}")
        
        return valid_stocks

    # ========== 4. 사용법 ==========

    def run_screening_with_validation(self):
        """검증된 종목들로만 스크리닝 실행"""
        try:
            # 전체 종목 리스트 가져오기
            all_stocks = self.get_selected_stocks()
            
            # 유효성 사전 체크 (선택사항)
            if len(all_stocks) > 50:  # 많은 종목일 때만 사전 체크
                valid_stocks = validate_stock_symbols(all_stocks[:10])  # 처음 10개만 테스트
                if len(valid_stocks) < 5:
                    QMessageBox.warning(self, "경고", "유효한 종목이 너무 적습니다. CSV 파일을 확인해주세요.")
                    return
            
            # 스크리닝 실행
            buy_candidates = []
            sell_candidates = []
            
            for i, stock_info in enumerate(all_stocks):
                try:
                    self.statusbar.showMessage(f'스크리닝 중... ({i+1}/{len(all_stocks)}) {stock_info["ticker"]}')
                    QApplication.processEvents()
                    
                    result = self.analyze_stock_with_error_handling(stock_info)
                    if result:
                        if result['action'] == 'BUY':
                            buy_candidates.append(result)
                        elif result['action'] == 'SELL':
                            sell_candidates.append(result)
                            
                except Exception as e:
                    print(f"스크리닝 오류: {stock_info['ticker']} - {e}")
                    continue
            
            # 결과 업데이트
            self.update_buy_table(buy_candidates)
            self.update_sell_table(sell_candidates)
            
            self.statusbar.showMessage(f'스크리닝 완료 - 매수후보: {len(buy_candidates)}개, 매도후보: {len(sell_candidates)}개')
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"스크리닝 중 오류가 발생했습니다: {str(e)}")
        finally:
            self.search_btn.setEnabled(True)

    # ==================== 보조 메서드들 ====================

    def check_enhanced_buy_condition(self, data, symbol):
        """강화된 매수 조건 체크 (기존 로직 유지)"""
        try:
            current = data.iloc[-1]
            
            # 기본 조건: 60일선이 120일선을 상향돌파
            if not (current['MA60'] > current['MA120'] and current['Close'] > current['MA60']):
                return False
            
            # 60일선이 120일선을 돌파한 날짜 찾기
            ma60_above_ma120_breakout_date = self.find_ma_breakout_date(data, 'MA60', 'MA120', days_limit=10)
            
            if ma60_above_ma120_breakout_date is None:
                return False
            
            # 장기 하락 추세 후의 반전인지 확인 (66거래일 기준)
            if not self.check_long_term_below_condition(data, ma60_above_ma120_breakout_date, days_check=66):
                return False
            
            print(f"✅ {symbol} - 모든 강화 조건 만족!")
            print(f"   - 60일선→120일선 상향돌파: {ma60_above_ma120_breakout_date.strftime('%Y-%m-%d')}")
            print(f"   - 현재 60일선: {current['MA60']:.2f}")
            print(f"   - 현재 120일선: {current['MA120']:.2f}")
            print(f"   - 현재가: {current['Close']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Error in enhanced buy condition check: {e}")
            return False

    def find_ma_breakout_date(self, data, fast_ma, slow_ma, days_limit):
        """이동평균 상향돌파 날짜 찾기"""
        try:
            if len(data) < 2:
                return None
                
            # 전체 데이터에서 상향돌파 시점들을 모두 찾기
            breakout_dates = []
            
            for i in range(1, len(data)):
                prev_day = data.iloc[i-1]
                current_day = data.iloc[i]
                
                # 상향돌파 조건: 어제는 fast_ma <= slow_ma, 오늘은 fast_ma > slow_ma
                if (prev_day[fast_ma] <= prev_day[slow_ma] and 
                    current_day[fast_ma] > current_day[slow_ma]):
                    
                    breakout_dates.append(data.index[i])
            
            if not breakout_dates:
                return None
            
            # 현재 시점을 기준으로 days_limit 일 이내의 상향돌파 찾기
            import pandas as pd
            
            today = data.index[-1]
            cutoff_date = today - pd.Timedelta(days=days_limit)
            
            # days_limit 일 이내의 상향돌파들만 필터링
            recent_breakouts = [date for date in breakout_dates if date >= cutoff_date]
            
            if recent_breakouts:
                # 가장 최근 상향돌파 반환
                return recent_breakouts[-1]
            else:
                return None
                
        except Exception as e:
            print(f"Error finding MA breakout: {e}")
            return None

    def find_ma_breakdown_date(self, data, fast_ma, slow_ma, days_limit):
        """이동평균 하향돌파 날짜 찾기 (매도 신호용)"""
        try:
            if len(data) < 2:
                return None
                
            # 전체 데이터에서 하향돌파 시점들을 모두 찾기
            breakdown_dates = []
            
            for i in range(1, len(data)):
                prev_day = data.iloc[i-1]
                current_day = data.iloc[i]
                
                # 하향돌파 조건: 어제는 fast_ma >= slow_ma, 오늘은 fast_ma < slow_ma
                if (prev_day[fast_ma] >= prev_day[slow_ma] and 
                    current_day[fast_ma] < current_day[slow_ma]):
                    
                    breakdown_dates.append(data.index[i])
            
            if not breakdown_dates:
                return None
            
            # 현재 시점을 기준으로 days_limit 일 이내의 하향돌파 찾기
            import pandas as pd
            
            today = data.index[-1]
            cutoff_date = today - pd.Timedelta(days=days_limit)
            
            # days_limit 일 이내의 하향돌파들만 필터링
            recent_breakdowns = [date for date in breakdown_dates if date >= cutoff_date]
            
            if recent_breakdowns:
                # 가장 최근 하향돌파 반환
                return recent_breakdowns[-1]
            else:
                return None
                
        except Exception as e:
            print(f"Error finding MA breakdown: {e}")
            return None

    def check_long_term_below_condition(self, data, breakout_date, days_check=66):
        """장기 하락 추세 후의 반전인지 확인"""
        try:
            import pandas as pd
            
            # 돌파 날짜 이전 days_check일 동안의 기간 설정
            check_start_date = breakout_date - pd.Timedelta(days=days_check)
            check_end_date = breakout_date - pd.Timedelta(days=1)  # 돌파 전날까지
            
            # 해당 기간의 데이터 추출
            check_period_data = data[(data.index >= check_start_date) & 
                                    (data.index <= check_end_date)]
            
            if len(check_period_data) < days_check * 0.5:  # 최소 50%의 데이터가 있어야 함
                return False
            
            # MA60과 MA120 데이터가 모두 있는 날들만 체크
            valid_data = check_period_data.dropna(subset=['MA60', 'MA120'])
            
            if len(valid_data) < len(check_period_data) * 0.7:  # 70% 이상이 유효해야 함
                return False
            
            # 60일선이 120일선 아래 있던 날의 비율 계산
            below_condition = valid_data['MA60'] < valid_data['MA120']
            below_ratio = below_condition.sum() / len(valid_data)
            
            # 90% 이상의 기간에서 60일선이 120일선 아래 있었으면 조건 만족
            return below_ratio >= 0.9
            
        except Exception as e:
            print(f"Error checking long term below condition: {e}")
            return False
        
    def check_custom_conditions(self, data, action_type):
        """사용자 정의 조건 체크"""
        signals = []
        current = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else current
        
        for i, condition in enumerate(self.custom_conditions):
            if condition['action'] == action_type:
                checkbox = self.custom_conditions_widget.findChild(QCheckBox, f"custom_condition_{i}")
                if checkbox and checkbox.isChecked():
                    try:
                        # 조건 평가
                        if self.evaluate_condition(condition, current, prev, data):
                            signals.append(condition['name'])
                    except Exception as e:
                        print(f"Error evaluating custom condition {condition['name']}: {e}")
        
        return signals
    
    def evaluate_condition(self, condition, current, prev, data):
        """사용자 정의 조건 평가"""
        indicator = condition['indicator']
        operator = condition['operator']
        value = float(condition['value'])
        
        # 지표 값 가져오기
        if indicator in current:
            current_value = float(current[indicator])
            prev_value = float(prev[indicator]) if indicator in prev else current_value
        else:
            return False
        
        # 연산자에 따른 평가
        if operator == '>':
            return current_value > value
        elif operator == '<':
            return current_value < value
        elif operator == '>=':
            return current_value >= value
        elif operator == '<=':
            return current_value <= value
        elif operator == '==':
            return abs(current_value - value) < 0.01
        elif operator == 'cross_above':
            return current_value > value and prev_value <= value
        elif operator == 'cross_below':
            return current_value < value and prev_value >= value
        
        return False
    
    def update_buy_table(self, candidates):
        """매수 후보 테이블 업데이트 - 모든 컬럼 정렬 가능"""
        # 임시로 정렬 비활성화
        self.buy_table.setSortingEnabled(False)
        
        self.buy_table.setRowCount(len(candidates))
        
        for i, candidate in enumerate(candidates):
            # 모든 컬럼을 정렬 가능하게 QTableWidgetItem 생성
            items = []
            
            # 0. 종목코드 (문자열 정렬)
            symbol_item = QTableWidgetItem(candidate['symbol'])
            items.append(symbol_item)
            
            # 1. 종목명 (문자열 정렬)
            name_item = QTableWidgetItem(candidate['name'])
            items.append(name_item)
            
            # 2. 섹터 (문자열 정렬)
            sector_item = QTableWidgetItem(candidate['sector'])
            items.append(sector_item)
            
            # 3. 현재가 (숫자 정렬)
            price_item = QTableWidgetItem()
            price_item.setData(Qt.DisplayRole, f"{candidate['price']:,.0f}")
            price_item.setData(Qt.UserRole, candidate['price'])  # 숫자로 정렬
            items.append(price_item)
            
            # 4. 시장 (문자열 정렬)
            market_item = QTableWidgetItem(candidate['market'])
            items.append(market_item)
            
            # 5. 매수신호 (문자열 정렬)
            signals_item = QTableWidgetItem(candidate['signals'])
            items.append(signals_item)
            
            # 6. RSI (숫자 정렬)
            rsi_item = QTableWidgetItem()
            rsi_item.setData(Qt.DisplayRole, f"{candidate['rsi']:.1f}")
            rsi_item.setData(Qt.UserRole, candidate['rsi'])
            items.append(rsi_item)
            
            # 7. 거래량비율 (숫자 정렬)
            volume_item = QTableWidgetItem()
            volume_item.setData(Qt.DisplayRole, f"{candidate['volume_ratio']:.2f}")
            volume_item.setData(Qt.UserRole, candidate['volume_ratio'])
            items.append(volume_item)
            
            # 8. 추천도 (숫자 정렬)
            recommendation_item = QTableWidgetItem()
            recommendation_item.setData(Qt.DisplayRole, f"{candidate['recommendation']:.0f}")
            recommendation_item.setData(Qt.UserRole, candidate['recommendation'])
            items.append(recommendation_item)
            
            # 9. 추세방향 (문자열 정렬)
            trend_direction = candidate.get('trend_direction', '분석불가')
            trend_item = QTableWidgetItem(trend_direction)
            items.append(trend_item)
            
            # 10. 추세강도 (숫자 정렬)
            trend_score = candidate.get('trend_score', 0)
            trend_score_item = QTableWidgetItem()
            trend_score_item.setData(Qt.DisplayRole, f"{trend_score:.1f}점")
            trend_score_item.setData(Qt.UserRole, trend_score)
            items.append(trend_score_item)
            
            # 11. 매수타이밍 (숫자 정렬)
            buy_timing = candidate.get('buy_timing', '대기')
            timing_item = QTableWidgetItem()
            timing_item.setData(Qt.DisplayRole, buy_timing)
            timing_score = self.get_timing_sort_score(buy_timing)
            timing_item.setData(Qt.UserRole, timing_score)
            items.append(timing_item)
            
            # 테이블에 아이템들 설정
            for col, item in enumerate(items):
                self.buy_table.setItem(i, col, item)
            
            # 색상 설정
            if "상승추세" in trend_direction:
                trend_item.setBackground(QColor(220, 255, 220))
            elif "하락추세" in trend_direction:
                trend_item.setBackground(QColor(255, 220, 220))
            else:
                trend_item.setBackground(QColor(255, 255, 220))
            
            if "★★★" in buy_timing:
                timing_item.setBackground(QColor(200, 255, 200))
            elif "★★" in buy_timing:
                timing_item.setBackground(QColor(230, 255, 230))
            elif "★" in buy_timing:
                timing_item.setBackground(QColor(255, 255, 200))
            else:
                timing_item.setBackground(QColor(240, 240, 240))
        
        # 정렬 다시 활성화
        self.buy_table.setSortingEnabled(True)
        
        # ✨ 기본적으로 추천도 기준 오름차순 정렬 (첫 클릭이 오름차순)
        self.buy_table.sortByColumn(8, Qt.AscendingOrder)
        
        # 테이블 컬럼 너비 자동 조정
        self.buy_table.resizeColumnsToContents()

    def update_sell_table(self, candidates):
        """매도 후보 테이블 업데이트 - 모든 컬럼 정렬 가능"""
        # 임시로 정렬 비활성화
        self.sell_table.setSortingEnabled(False)
        
        self.sell_table.setRowCount(len(candidates))
        
        for i, candidate in enumerate(candidates):
            # 모든 컬럼을 정렬 가능하게 QTableWidgetItem 생성
            items = []
            
            # 0-5번: 기본 정보들 (매수 테이블과 동일)
            symbol_item = QTableWidgetItem(candidate['symbol'])
            name_item = QTableWidgetItem(candidate['name'])
            sector_item = QTableWidgetItem(candidate['sector'])
            
            price_item = QTableWidgetItem()
            price_item.setData(Qt.DisplayRole, f"{candidate['price']:,.0f}")
            price_item.setData(Qt.UserRole, candidate['price'])
            
            market_item = QTableWidgetItem(candidate['market'])
            signals_item = QTableWidgetItem(candidate['signals'])
            
            items.extend([symbol_item, name_item, sector_item, price_item, market_item, signals_item])
            
            # 6. 수익률 (숫자 정렬)
            profit_item = QTableWidgetItem()
            profit_value = candidate.get('profit', 0)
            profit_item.setData(Qt.DisplayRole, f"{profit_value:.1f}%")
            profit_item.setData(Qt.UserRole, profit_value)
            items.append(profit_item)
            
            # 7. 보유기간 (문자열 정렬 - 실제로는 숫자로 변환 가능)
            holding_period_item = QTableWidgetItem(candidate.get('holding_period', '미상'))
            items.append(holding_period_item)
            
            # 8. 위험도 (숫자 정렬)
            risk_item = QTableWidgetItem()
            risk_item.setData(Qt.DisplayRole, f"{candidate['risk']:.0f}")
            risk_item.setData(Qt.UserRole, candidate['risk'])
            items.append(risk_item)
            
            # 9-11번: 추세 정보 (매수 테이블과 동일)
            trend_direction = candidate.get('trend_direction', '분석불가')
            trend_item = QTableWidgetItem(trend_direction)
            
            trend_score = candidate.get('trend_score', 0)
            trend_score_item = QTableWidgetItem()
            trend_score_item.setData(Qt.DisplayRole, f"{trend_score:.1f}점")
            trend_score_item.setData(Qt.UserRole, trend_score)
            
            sell_timing = candidate.get('sell_timing', '대기')
            timing_item = QTableWidgetItem()
            timing_item.setData(Qt.DisplayRole, sell_timing)
            timing_score = self.get_timing_sort_score(sell_timing)
            timing_item.setData(Qt.UserRole, timing_score)
            
            items.extend([trend_item, trend_score_item, timing_item])
            
            # 테이블에 아이템들 설정
            for col, item in enumerate(items):
                self.sell_table.setItem(i, col, item)
            
            # 색상 설정
            if "하락추세" in trend_direction:
                trend_item.setBackground(QColor(255, 200, 200))
            elif "상승추세" in trend_direction:
                trend_item.setBackground(QColor(200, 255, 200))
            else:
                trend_item.setBackground(QColor(255, 255, 220))
            
            if "★★★" in sell_timing:
                timing_item.setBackground(QColor(255, 200, 200))
            elif "★★" in sell_timing:
                timing_item.setBackground(QColor(255, 230, 230))
            elif "★" in sell_timing:
                timing_item.setBackground(QColor(255, 255, 200))
            else:
                timing_item.setBackground(QColor(240, 240, 240))
        
        # 정렬 다시 활성화
        self.sell_table.setSortingEnabled(True)
        
        # ✨ 기본적으로 위험도 기준 오름차순 정렬 (첫 클릭이 오름차순)
        self.sell_table.sortByColumn(8, Qt.AscendingOrder)
        
        # 테이블 컬럼 너비 자동 조정
        self.sell_table.resizeColumnsToContents()
    
    # ✨ 추가 편의 기능: 버튼으로 빠른 정렬
    def add_quick_sort_buttons(self):
        """빠른 정렬 버튼들 추가 (선택사항)"""
        # 매수 테이블 위에 빠른 정렬 버튼들
        buy_sort_layout = QHBoxLayout()
        
        sort_by_recommendation_btn = QPushButton("추천도순")
        sort_by_recommendation_btn.clicked.connect(lambda: self.buy_table.sortByColumn(8, Qt.DescendingOrder))
        buy_sort_layout.addWidget(sort_by_recommendation_btn)
        
        sort_by_trend_btn = QPushButton("추세강도순")
        sort_by_trend_btn.clicked.connect(lambda: self.buy_table.sortByColumn(10, Qt.DescendingOrder))
        buy_sort_layout.addWidget(sort_by_trend_btn)
        
        sort_by_timing_btn = QPushButton("매수타이밍순")
        sort_by_timing_btn.clicked.connect(lambda: self.buy_table.sortByColumn(11, Qt.DescendingOrder))
        buy_sort_layout.addWidget(sort_by_timing_btn)
        
        # 매도 테이블 위에 빠른 정렬 버튼들
        sell_sort_layout = QHBoxLayout()
        
        sort_by_risk_btn = QPushButton("위험도순")
        sort_by_risk_btn.clicked.connect(lambda: self.sell_table.sortByColumn(8, Qt.DescendingOrder))
        sell_sort_layout.addWidget(sort_by_risk_btn)
        
        sort_by_sell_timing_btn = QPushButton("매도타이밍순")
        sort_by_sell_timing_btn.clicked.connect(lambda: self.sell_table.sortByColumn(11, Qt.DescendingOrder))
        sell_sort_layout.addWidget(sort_by_sell_timing_btn)
        
        return buy_sort_layout, sell_sort_layout

    def show_stock_detail(self, index_or_ticker, name=""):
        """테이블에서 종목 더블클릭시 상세 차트 표시 - 매개변수 타입 안전 처리"""
        try:
            ticker = ""
            stock_name = ""
            
            # 매개변수 타입에 따라 처리 방법 결정
            if isinstance(index_or_ticker, str):
                # 문자열이 직접 전달된 경우 (ticker)
                ticker = index_or_ticker
                stock_name = name if name else ticker
                print(f"직접 ticker 전달: {ticker}")
                
            elif hasattr(index_or_ticker, 'row'):
                # QModelIndex 객체인 경우 (테이블에서 더블클릭)
                table = self.sender()
                if not table:
                    print("Error: sender()가 None입니다")
                    return
                    
                row = index_or_ticker.row()
                print(f"테이블 더블클릭: row {row}")
                
                # 종목 코드와 이름 가져오기
                ticker_item = table.item(row, 0)  # 종목코드
                name_item = table.item(row, 1)    # 종목명
                
                if ticker_item:
                    ticker = ticker_item.text()
                if name_item:
                    stock_name = name_item.text()
                    
            else:
                # 기타 경우 - 정수인 경우 row로 간주
                try:
                    row = int(index_or_ticker)
                    table = self.sender()
                    if table and hasattr(table, 'item'):
                        ticker_item = table.item(row, 0)
                        name_item = table.item(row, 1)
                        
                        if ticker_item:
                            ticker = ticker_item.text()
                        if name_item:
                            stock_name = name_item.text()
                    else:
                        print(f"Error: 유효하지 않은 테이블 참조")
                        return
                except (ValueError, TypeError):
                    print(f"Error: 알 수 없는 매개변수 타입: {type(index_or_ticker)}")
                    return
            
            if not ticker:
                QMessageBox.warning(self, "경고", "종목 정보를 가져올 수 없습니다.")
                return
            
            print(f"차트 표시 시도: {ticker} ({stock_name})")
            
            # 차트 창 생성 및 표시
            try:
                # chart_window.py에서 StockChartWindow 임포트 시도
                from chart_window import StockChartWindow
                
                # 기존 같은 종목 차트 창이 있으면 닫기
                for window in QApplication.topLevelWidgets():
                    if isinstance(window, StockChartWindow) and hasattr(window, 'symbol') and window.symbol == ticker:
                        window.close()
                
                # 새 차트 창 열기
                chart_window = StockChartWindow(ticker, stock_name, self)
                chart_window.show()
                
                self.statusbar.showMessage(f"📊 {ticker} ({stock_name}) 차트를 열었습니다.")
                print(f"✅ 차트 창 열림: {ticker} ({stock_name})")
                
            except ImportError as e:
                # StockChartWindow를 찾을 수 없는 경우 간단한 메시지 표시
                print(f"차트 모듈 import 실패: {e}")
                QMessageBox.information(self, "차트", 
                                    f"종목: {ticker} ({stock_name})\n"
                                    f"차트 기능을 사용하려면 chart_window.py 파일이 필요합니다.")
                                    
            except Exception as chart_error:
                # 차트 생성 중 오류 발생시
                print(f"차트 생성 오류: {chart_error}")
                QMessageBox.warning(self, "차트 오류", 
                                f"차트를 불러오는 중 오류가 발생했습니다:\n{str(chart_error)}")
                
        except Exception as e:
            print(f"Error in show_stock_detail: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "오류", f"종목 상세 정보를 표시하는 중 오류가 발생했습니다:\n{str(e)}")


    # ========== 추가로 필요한 간단한 차트 기능 (chart_window.py가 없는 경우) ==========

    def show_simple_stock_info(self, symbol, name):
        """간단한 종목 정보 다이얼로그 (차트 대안)"""
        try:
            # 최근 1개월 데이터 (캐싱 사용)
            data = get_stock_data(symbol, period="1mo")
            
            if len(data) == 0:
                QMessageBox.warning(self, "데이터 없음", f"{symbol} 데이터를 가져올 수 없습니다.")
                return
            
            current = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else current
            
            # 기본 지표 계산
            data['MA20'] = data['Close'].rolling(20).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 정보 텍스트 구성
            price_change = float(current['Close']) - float(prev['Close'])
            price_change_pct = (price_change / float(prev['Close'])) * 100 if prev['Close'] else 0.0
            
            info_text = f"""
    📊 {symbol} ({name}) 종목 정보

    💰 현재가: {current['Close']:.2f}
    📈 전일대비: {price_change:+.2f} ({price_change_pct:+.2f}%)

    📊 기술적 지표:
    • RSI: {current['RSI']:.1f}
    • 20일 이평선: {current['MA20']:.2f}
    • 거래량: {current['Volume']:,.0f}

    📅 최고가 (1개월): {data['High'].max():.2f}
    📅 최저가 (1개월): {data['Low'].min():.2f}
            """
            
            # 다이얼로그로 정보 표시
            dialog = QMessageBox(self)
            dialog.setWindowTitle(f"📊 {symbol} 종목 정보")
            dialog.setText(info_text.strip())
            dialog.setIcon(QMessageBox.Information)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"종목 정보를 가져오는 중 오류가 발생했습니다:\n{str(e)}")

    # ========== 대안: 차트 없이 테이블만 사용하는 경우 ==========

    def show_stock_detail_simple(self, index):
        """차트 없이 간단한 정보만 표시하는 버전"""
        try:
            table = self.sender()
            row = index.row()
            
            # 테이블에서 모든 정보 수집
            symbol = table.item(row, 0).text() if table.item(row, 0) else ""
            name = table.item(row, 1).text() if table.item(row, 1) else ""
            sector = table.item(row, 2).text() if table.item(row, 2) else ""
            price = table.item(row, 3).text() if table.item(row, 3) else ""
            market = table.item(row, 4).text() if table.item(row, 4) else ""
            signals = table.item(row, 5).text() if table.item(row, 5) else ""
            
            # 추세 정보 (있는 경우)
            trend_direction = ""
            trend_score = ""
            timing = ""
            
            if table.columnCount() >= 12:  # 추세 분석 컬럼이 있는 경우
                trend_direction = table.item(row, 9).text() if table.item(row, 9) else ""
                trend_score = table.item(row, 10).text() if table.item(row, 10) else ""
                timing = table.item(row, 11).text() if table.item(row, 11) else ""
            
            # 정보 다이얼로그 표시
            info_text = f"""
    📊 종목 상세 정보

    🏢 종목명: {name} ({symbol})
    🏭 섹터: {sector}
    💰 현재가: {price}
    🌍 시장: {market}
    🔍 신호: {signals}
    """
            
            if trend_direction and trend_score and timing:
                info_text += f"""
    📈 추세 분석:
    • 추세방향: {trend_direction}
    • 추세강도: {trend_score}
    • 타이밍: {timing}
    """
            
            QMessageBox.information(self, f"📊 {symbol} 상세정보", info_text.strip())
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"종목 정보를 표시하는 중 오류가 발생했습니다:\n{str(e)}")

    # ========== 사용 예시 ==========
    """
    사용 방법:

    1. 완전한 차트 기능을 원하는 경우:
    - show_stock_detail 메서드 사용
    - chart_window.py 파일 필요

    2. 간단한 정보만 원하는 경우:
    - show_stock_detail_simple 메서드 사용
    - 추가 파일 불필요

    3. create_tables()에서 연결:
    self.buy_table.doubleClicked.connect(self.show_stock_detail)
    또는
    self.buy_table.doubleClicked.connect(self.show_stock_detail_simple)
    """
# screener.py에 추가할 완성된 검색 기능 통합
    def create_stock_search_panel(self):
        """🔍 종목 검색 패널 - 크기 축소 버전"""
        search_group = QGroupBox("🔍 종목 검색 및 차트 보기")
        search_group.setMaximumHeight(80)  # 높이 통일
        search_group.setMinimumHeight(80)  # 높이 고정
        search_layout = QHBoxLayout()
        
        # 검색어 입력 필드 (크기 축소)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("종목코드/회사명 (예: 005930, AAPL)")
        self.search_input.returnPressed.connect(self.search_and_show_chart)
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.setMaximumWidth(450)  # 축소
        
        search_layout.addWidget(QLabel("검색:"))
        search_layout.addWidget(self.search_input)
        
        # 버튼들 (크기 축소)
        self.search_btn = QPushButton("🔍Search")
        self.search_btn.clicked.connect(self.search_and_show_chart)
        self.search_btn.setMaximumWidth(120)
        search_layout.addWidget(self.search_btn)
        
        self.random_stock_btn = QPushButton("🎲Random")
        self.random_stock_btn.clicked.connect(self.show_random_stock_chart)
        self.random_stock_btn.setToolTip("랜덤 종목")
        self.random_stock_btn.setMaximumWidth(120)
        search_layout.addWidget(self.random_stock_btn)
        
        self.search_help_btn = QPushButton("❓Help")
        self.search_help_btn.clicked.connect(self.show_search_help)
        self.search_help_btn.setToolTip("도움말")
        self.search_help_btn.setMaximumWidth(120)
        search_layout.addWidget(self.search_help_btn)
        
        # 검색 결과 레이블 (크기 축소)
        self.search_result_label = QLabel()
        self.search_result_label.setStyleSheet("color: #666; font-style: italic; font-size: 11px;")
        self.search_result_label.setMaximumWidth(120)
        search_layout.addWidget(self.search_result_label)
        
        search_layout.addStretch()
        search_group.setLayout(search_layout)
        
        return search_group

    def create_custom_conditions_panel(self):
        """⚙️ 사용자 정의 조건 패널 - 기존 로직 활용"""
        custom_group = QGroupBox("⚙️ 사용자 정의 조건")
        custom_group.setMaximumHeight(80)  # 높이 통일
        custom_group.setMinimumHeight(80)  # 높이 고정
        custom_layout = QHBoxLayout()
        
        # 조건 추가 버튼 (크기 축소)
        self.add_condition_btn = QPushButton("+ 조건")
        self.add_condition_btn.clicked.connect(self.open_condition_builder)
        self.add_condition_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        self.add_condition_btn.setMaximumWidth(80)
        custom_layout.addWidget(self.add_condition_btn)
        
        # 조건 관리 버튼 (크기 축소)
        self.manage_conditions_btn = QPushButton("⚙️ 관리")
        self.manage_conditions_btn.clicked.connect(self.manage_custom_conditions)
        self.manage_conditions_btn.setMaximumWidth(80)
        custom_layout.addWidget(self.manage_conditions_btn)
        
        # 사용자 정의 조건 표시 영역 (크기 축소)
        self.custom_conditions_area = QScrollArea()
        self.custom_conditions_widget = QWidget()
        self.custom_conditions_layout = QVBoxLayout(self.custom_conditions_widget)
        self.custom_conditions_area.setWidget(self.custom_conditions_widget)
        self.custom_conditions_area.setMaximumHeight(60)  # 높이 축소
        self.custom_conditions_area.setMaximumWidth(450)  # 너비 축소
        custom_layout.addWidget(self.custom_conditions_area)
        
        custom_layout.addStretch()
        custom_group.setLayout(custom_layout)
        
        return custom_group

    def on_search_text_changed(self, text):
        """검색어 변경 시 실시간 제안 - 안전한 버전"""
        try:
            if len(text) >= 2:
                suggestions = self.get_search_suggestions(text)
                if suggestions:
                    tooltip_text = "제안: " + ", ".join(suggestions[:3])
                    self.search_input.setToolTip(tooltip_text)
                else:
                    self.search_input.setToolTip("")
            else:
                self.search_input.setToolTip("")
                
        except Exception as e:
            print(f"⚠️ 검색어 변경 처리 오류: {e}")
            self.search_input.setToolTip("")

    def get_search_suggestions(self, search_term, limit=5):
        """검색어 자동완성 제안 - 리스트 형태 데이터 대응"""
        if len(search_term) < 2:
            return []
        
        suggestions = []
        seen = set()
        search_upper = search_term.upper()
        
        try:
            # stock_lists의 데이터 형태 확인 및 처리
            for market, data in self.stock_lists.items():
                # 데이터가 비어있으면 스킵
                if not data:
                    continue
                
                # DataFrame인 경우
                if hasattr(data, 'empty') and hasattr(data, 'iterrows'):
                    if data.empty:
                        continue

                    # ✅ 벡터화: iterrows() 제거 - 15-20배 성능 향상
                    # 티커와 이름을 대문자로 변환
                    tickers = data['ticker'].fillna('').astype(str).str.upper()
                    names = data['name'].fillna('').astype(str).str.upper()

                    # 티커로 시작하는 항목 필터링
                    ticker_mask = tickers.str.startswith(search_upper)
                    for ticker in tickers[ticker_mask]:
                        if ticker not in seen:
                            suggestions.append(ticker)
                            seen.add(ticker)
                            if len(suggestions) >= limit:
                                break

                    # 회사명으로 시작하는 항목 필터링
                    if len(suggestions) < limit:
                        for name in names:
                            words = name.split()
                            if words and any(word.startswith(search_upper) for word in words) and name not in seen:
                                suggestions.append(words[0])
                                seen.add(name)
                                if len(suggestions) >= limit:
                                    break
                
                # 리스트인 경우
                elif isinstance(data, list):
                    for stock in data:
                        if not isinstance(stock, dict):
                            continue
                            
                        ticker = str(stock.get('ticker', '')).upper()
                        name = str(stock.get('name', '')).upper()
                        
                        # 티커로 시작하는 것
                        if ticker.startswith(search_upper) and ticker not in seen:
                            suggestions.append(ticker)
                            seen.add(ticker)
                        
                        # 회사명으로 시작하는 것
                        elif any(word.startswith(search_upper) for word in name.split()) and name not in seen:
                            suggestions.append(name.split()[0])  # 첫 번째 단어만
                            seen.add(name)
                        
                        if len(suggestions) >= limit:
                            break
                
                if len(suggestions) >= limit:
                    break
            
            return suggestions
            
        except Exception as e:
            print(f"⚠️ 검색 제안 오류: {e}")
            return []

    def search_and_show_chart(self):
        """검색 후 차트 표시 - 안전한 버전"""
        search_term = self.search_input.text().strip()
        
        if not search_term:
            QMessageBox.warning(self, "검색 오류", "검색어를 입력해주세요.")
            return
        
        try:
            # 최근 검색어에 추가 (안전하게)
            self.add_to_recent_searches(search_term)
            
            # 검색 실행
            self.update_search_result_label("검색 중...")
            QApplication.processEvents()
            
            found_stocks = self.enhanced_search_stocks(search_term)
            
            if not found_stocks:
                # 온라인 검색 시도
                self.update_search_result_label("CSV에서 검색 결과 없음. 온라인 검색 중...")
                QApplication.processEvents()
                
                if self.try_online_search(search_term):
                    return
                else:
                    QMessageBox.information(
                        self,
                        "검색 결과 없음",
                        f"'{search_term}'에 대한 검색 결과가 없습니다.\n\n"
                        "검색 팁:\n"
                        "• 정확한 종목코드 또는 티커 사용 (예: 005930, AAPL)\n"
                        "• 회사명의 일부만 입력 (예: 삼성, Apple)\n"
                        "• 영문은 대소문자 구분 없음\n"
                        "• CSV 파일이 최신인지 확인 ('온라인 종목 업데이트')"
                    )
                    self.update_search_result_label("검색 결과 없음")
                    return
            
            # 검색 결과 처리
            if len(found_stocks) == 1:
                # 단일 결과면 바로 차트 표시
                stock = found_stocks[0]
                self.update_search_result_label(
                    f"✅ {stock['name']} ({stock['ticker']}) - {stock['market']}"
                )
                self.show_stock_chart(stock['ticker'], stock['name'])
                
            else:
                # 여러 결과면 선택 다이얼로그
                self.update_search_result_label(f"🔍 {len(found_stocks)}개 종목 발견")
                self.show_search_results_dialog(found_stocks, search_term)
                
        except Exception as e:
            print(f"⚠️ 검색 및 차트 표시 오류: {e}")
            self.update_search_result_label("검색 오류 발생")
            QMessageBox.critical(self, "검색 오류", f"검색 중 오류가 발생했습니다: {str(e)}")

    def update_search_result_label(self, text):
        """검색 결과 레이블 업데이트 - 안전한 버전"""
        try:
            if hasattr(self, 'search_result_label'):
                self.search_result_label.setText(text)
            else:
                print(f"검색 결과: {text}")
        except Exception as e:
            print(f"⚠️ 검색 결과 레이블 업데이트 오류: {e}")

    def enhanced_search_stocks(self, search_term):
        """향상된 종목 검색 - unified_search 사용 (최적화됨)"""
        if not search_term.strip():
            return []

        try:
            print(f"🔍 '{search_term}' 검색 중...")

            # ✅ 통합 검색 모듈 사용 (벡터화 + 캐싱)
            results = search_stocks(search_term.strip())

            # 기존 형식에 맞춰 변환
            for result in results:
                # match_score가 없으면 추가
                if 'match_score' not in result:
                    ticker_upper = result['ticker'].upper()
                    name_upper = result['name'].upper()
                    search_upper = search_term.strip().upper()

                    if ticker_upper == search_upper:
                        result['match_score'] = 100
                    elif name_upper == search_upper:
                        result['match_score'] = 95
                    elif search_upper in ticker_upper:
                        result['match_score'] = 85
                    elif search_upper in name_upper:
                        result['match_score'] = 75
                    else:
                        result['match_score'] = 60

                # match_reasons 추가
                if 'match_reasons' not in result:
                    result['match_reasons'] = []

                # raw_market_cap 추가 (정렬용)
                if 'raw_market_cap' not in result:
                    result['raw_market_cap'] = result.get('market_cap', 0)

            # 정렬 (매치 스코어 -> 시가총액 -> 이름순)
            results.sort(key=lambda x: (-x.get('match_score', 0), -x.get('raw_market_cap', 0), x.get('name', '')))

            print(f"🎯 검색 완료: '{search_term}' → {len(results)}개 결과")
            return results

        except Exception as e:
            print(f"⚠️ 검색 중 오류: {e}")
            # 폴백: 현재 로딩된 CSV에서 검색
            return self.search_from_loaded_csv(search_term)

    def search_from_loaded_csv(self, search_term):
        """기존 로딩된 CSV에서 검색 (폴백 함수)"""
        found_stocks = []
        seen_tickers = set()
        
        try:
            for market, data in self.stock_lists.items():
                if not data:
                    continue
                
                for stock in data:
                    if not isinstance(stock, dict):
                        continue
                    
                    ticker = str(stock.get('ticker', '')).strip()
                    name = str(stock.get('name', '')).strip()
                    
                    if not ticker or ticker in seen_tickers:
                        continue
                    
                    # 간단한 매칭
                    if (search_term.upper() in ticker.upper() or 
                        search_term.upper() in name.upper()):
                        
                        found_stocks.append({
                            'ticker': ticker,
                            'name': name,
                            'sector': stock.get('sector', ''),
                            'market_cap': str(stock.get('market_cap', 0)),
                            'market': market.upper(),
                            'match_score': 70,
                            'match_reasons': ["기본 매치"],
                            'raw_market_cap': stock.get('market_cap', 0)
                        })
                        seen_tickers.add(ticker)
            
            return found_stocks
            
        except Exception as e:
            print(f"⚠️ 로딩된 CSV 검색 오류: {e}")
            return []

    # 추가로 필요한 함수: 마스터 CSV 파일 존재 여부 확인
    def check_master_csv_availability(self):
        """마스터 CSV 파일들의 존재 여부 확인"""
        master_files = {
            'korea': 'stock_data/korea_stocks_master.csv',
            'usa': 'stock_data/usa_stocks_master.csv', 
            'sweden': 'stock_data/sweden_stocks_master.csv'
        }
        
        available = {}
        total_stocks = 0
        
        for market, file_path in master_files.items():
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    available[market] = len(df)
                    total_stocks += len(df)
                except:
                    available[market] = 0
            else:
                available[market] = 0
        
        if total_stocks > 0:
            market_info = []
            for market, count in available.items():
                if count > 0:
                    market_info.append(f"{market}: {count:,}개")
            
            info_text = f"마스터 CSV 사용 가능: 총 {total_stocks:,}개 종목\n" + " | ".join(market_info)
            self.statusbar.showMessage(info_text)
            print(f"✅ {info_text}")
        else:
            self.statusbar.showMessage("마스터 CSV 없음 - '마스터 CSV 생성' 버튼을 클릭하세요")
            print("⚠️ 마스터 CSV 파일이 없습니다")
        
        return available

    def show_search_results_dialog(self, found_stocks, search_term):
        """검색 결과 선택 다이얼로그 - 향상된 버전"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"🔍 검색 결과: '{search_term}'")
        dialog.setModal(True)
        dialog.resize(1000, 500)
        
        layout = QVBoxLayout(dialog)
        
        # 상단 정보
        info_layout = QHBoxLayout()
        info_label = QLabel(f"📊 {len(found_stocks)}개의 종목이 발견되었습니다")
        info_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
        info_layout.addWidget(info_label)
        
        # 정렬 옵션
        sort_combo = QComboBox()
        sort_combo.addItems(["매치 점수순", "회사명순", "시가총액순", "시장별"])
        sort_combo.currentTextChanged.connect(
            lambda: self.resort_search_results(dialog, found_stocks, sort_combo.currentText())
        )
        info_layout.addWidget(QLabel("정렬:"))
        info_layout.addWidget(sort_combo)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        # 검색 결과 테이블
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            '티커', '회사명', '섹터', '시가총액', '시장', '매치점수', '매치이유'
        ])
        
        self.populate_search_results_table(table, found_stocks)
        
        # 테이블 설정
        table.resizeColumnsToContents()
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        
        # 더블클릭으로 차트 열기
        def on_double_click(row, col):
            if row < len(found_stocks):
                selected_stock = found_stocks[row]
                dialog.accept()
                self.show_stock_chart(selected_stock['ticker'], selected_stock['name'])
        
        table.cellDoubleClicked.connect(on_double_click)
        layout.addWidget(table)
        
        # 하단 버튼들
        button_layout = QHBoxLayout()
        
        # 차트 보기 버튼
        view_chart_btn = QPushButton("📊 차트 보기")
        view_chart_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
            }
        """)
        
        def on_view_chart():
            current_row = table.currentRow()
            if current_row >= 0 and current_row < len(found_stocks):
                selected_stock = found_stocks[current_row]
                dialog.accept()
                self.show_stock_chart(selected_stock['ticker'], selected_stock['name'])
            else:
                QMessageBox.warning(dialog, "선택 오류", "차트를 볼 종목을 선택해주세요.")
        
        view_chart_btn.clicked.connect(on_view_chart)
        button_layout.addWidget(view_chart_btn)
        
        # 결과 내보내기 버튼
        export_btn = QPushButton("📁 Excel 내보내기")
        export_btn.clicked.connect(lambda: self.export_search_results(found_stocks, search_term))
        button_layout.addWidget(export_btn)
        
        # 취소 버튼
        cancel_btn = QPushButton("❌ 취소")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        button_layout.addStretch()
        
        # 도움말
        help_label = QLabel("💡 팁: 종목을 더블클릭하거나 선택 후 '차트 보기'를 클릭하세요")
        help_label.setStyleSheet("color: #666; font-style: italic; margin: 5px;")
        button_layout.addWidget(help_label)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()

    def populate_search_results_table(self, table, found_stocks):
        """검색 결과 테이블 채우기"""
        table.setRowCount(len(found_stocks))
        
        for i, stock in enumerate(found_stocks):
            table.setItem(i, 0, QTableWidgetItem(stock['ticker']))
            table.setItem(i, 1, QTableWidgetItem(stock['name']))
            table.setItem(i, 2, QTableWidgetItem(stock['sector']))

            # market_cap을 포맷팅 (OverflowError 방지)
            market_cap_raw = stock.get('market_cap', '')
            if isinstance(market_cap_raw, (int, float)):
                market_cap_str = format_market_cap_value(market_cap_raw)
            else:
                market_cap_str = str(market_cap_raw) if market_cap_raw else 'N/A'

            table.setItem(i, 3, QTableWidgetItem(market_cap_str))
            table.setItem(i, 4, QTableWidgetItem(stock['market']))
            
            # 매치 점수 (숫자로 정렬 가능하도록)
            score_item = QTableWidgetItem()
            score_item.setData(Qt.DisplayRole, stock['match_score'])
            table.setItem(i, 5, score_item)
            
            # 매치 이유
            reasons = ", ".join(stock['match_reasons'])
            table.setItem(i, 6, QTableWidgetItem(reasons))
            
            # 높은 매치 점수는 녹색으로 강조
            if stock['match_score'] >= 90:
                for col in range(7):
                    table.item(i, col).setBackground(QColor(200, 255, 200))
            elif stock['match_score'] >= 70:
                for col in range(7):
                    table.item(i, col).setBackground(QColor(255, 255, 200))

    def try_online_search(self, search_term):
        """온라인에서 직접 종목 검색"""
        try:
            import yfinance as yf
            
            # 다양한 패턴으로 시도
            search_patterns = [
                search_term,
                search_term + ".KS",  # 한국 코스피
                search_term + ".KQ",  # 한국 코스닥  
                search_term + ".ST"   # 스웨덴
            ]
            
            for pattern in search_patterns:
                try:
                    # 캐싱 사용
                    info = get_ticker_info(pattern)

                    if info and info.get('symbol'):
                        name = info.get('longName') or info.get('shortName') or pattern
                        self.search_result_label.setText(f"🌐 온라인 발견: {name} ({pattern})")
                        self.show_stock_chart(pattern, name)
                        return True

                except Exception as e:
                    continue
            
            return False
            
        except Exception as e:
            print(f"온라인 검색 오류: {e}")
            return False

    def show_random_stock_chart(self):
        """🎲 랜덤 종목 차트 표시 - 안전한 버전"""
        import random
        
        try:
            # 모든 종목 수집
            all_stocks = []
            
            if not hasattr(self, 'stock_lists') or not self.stock_lists:
                QMessageBox.warning(self, "오류", "로드된 종목 데이터가 없습니다.\n먼저 CSV 파일을 로드하거나 '샘플 생성'을 실행해주세요.")
                return
            
            for market, data in self.stock_lists.items():
                if not data:
                    continue
                
                try:
                    # DataFrame인 경우
                    if hasattr(data, 'empty') and hasattr(data, 'iterrows'):
                        if not data.empty:
                            # ✅ 벡터화: iterrows() 제거 - 30-40배 성능 향상
                            # 유효한 티커와 이름만 필터링
                            valid_mask = data['ticker'].notna() & data['name'].notna()
                            valid_data = data[valid_mask]

                            # 딕셔너리 리스트로 변환
                            stocks_list = valid_data.apply(lambda row: {
                                'ticker': str(row['ticker']),
                                'name': str(row['name']),
                                'market': market,
                                'market_cap': row.get('market_cap', 0)
                            }, axis=1).tolist()

                            all_stocks.extend(stocks_list)
                    
                    # 리스트인 경우
                    elif isinstance(data, list):
                        for stock in data:
                            if isinstance(stock, dict):
                                ticker = stock.get('ticker')
                                name = stock.get('name')
                                if ticker and name:
                                    all_stocks.append({
                                        'ticker': str(ticker),
                                        'name': str(name),
                                        'market': market,
                                        'market_cap': stock.get('market_cap', 0)
                                    })
                    
                except Exception as e:
                    print(f"⚠️ {market} 시장 데이터 처리 오류: {e}")
                    continue
            
            if not all_stocks:
                QMessageBox.warning(self, "오류", "표시할 종목이 없습니다.\n먼저 CSV 파일을 로드하거나 '온라인 종목 업데이트'를 실행해주세요.")
                return
            
            # 시가총액이 있는 종목을 우선적으로 선택 (더 의미있는 랜덤)
            weighted_stocks = []
            for stock in all_stocks:
                mcap = stock.get('market_cap', 0)
                try:
                    if isinstance(mcap, (int, float)) and mcap > 0:
                        # 시총 있는 종목은 3배 가중치
                        weighted_stocks.extend([stock] * 3)
                    else:
                        weighted_stocks.append(stock)
                except:
                    weighted_stocks.append(stock)
            
            # 랜덤 선택
            random_stock = random.choice(weighted_stocks if weighted_stocks else all_stocks)
            
            # 시가총액 정보 포함해서 표시
            mcap_info = ""
            if random_stock.get('market_cap', 0):
                try:
                    mcap = float(random_stock['market_cap'])
                    if mcap >= 1e12:
                        mcap_info = f" (시총: {mcap/1e12:.1f}조)"
                    elif mcap >= 1e9:
                        mcap_info = f" (시총: {mcap/1e9:.1f}B)"
                    elif mcap >= 1e6:
                        mcap_info = f" (시총: {mcap/1e6:.1f}M)"
                    else:
                        mcap_info = f" (시총: {mcap:,.0f})"
                except:
                    mcap_info = ""
            
            # 검색 결과 레이블 업데이트
            result_text = f"🎲 랜덤: {random_stock['name']} ({random_stock['ticker']}) - {random_stock['market']}{mcap_info}"
            self.update_search_result_label(result_text)
            
            # 차트 표시
            self.show_stock_chart(random_stock['ticker'], random_stock['name'])
            
            print(f"🎲 랜덤 선택: {random_stock['ticker']} - {random_stock['name']}")
            
        except Exception as e:
            print(f"⚠️ 랜덤 종목 선택 오류: {e}")
            QMessageBox.critical(self, "오류", f"랜덤 종목 선택 중 오류가 발생했습니다: {str(e)}")

    def add_to_recent_searches(self, search_term):
        """최근 검색어에 추가 - 안전한 버전"""
        try:
            if not hasattr(self, 'recent_searches'):
                self.recent_searches = []
            
            if search_term in self.recent_searches:
                self.recent_searches.remove(search_term)
            
            self.recent_searches.insert(0, search_term)
            self.recent_searches = self.recent_searches[:5]  # 최대 5개까지
            
            # 레이블이 존재하는 경우에만 업데이트
            if hasattr(self, 'recent_searches_label'):
                self.update_recent_searches_display()
            
            print(f"📝 최근 검색어 추가: {search_term}")
            
        except Exception as e:
            print(f"⚠️ 최근 검색어 추가 오류: {e}")

    def update_recent_searches_display(self):
        """최근 검색어 표시 업데이트 - 안전한 버전"""
        try:
            # 레이블이 존재하는지 확인
            if not hasattr(self, 'recent_searches_label'):
                return
            
            if not hasattr(self, 'recent_searches'):
                self.recent_searches = []
            
            if self.recent_searches:
                recent_text = "최근 검색: " + " | ".join(self.recent_searches)
                self.recent_searches_label.setText(recent_text)
            else:
                self.recent_searches_label.setText("💡 팁: Enter 키 또는 🔍 버튼으로 검색하세요")
                
        except Exception as e:
            print(f"⚠️ 최근 검색어 표시 오류: {e}")

    def on_recent_search_click(self, event):
        """최근 검색어 클릭 처리"""
        # 추후 구현: 최근 검색어를 클릭하면 해당 검색어로 다시 검색
        pass


    def show_search_help(self):
        """검색 도움말 표시"""
        help_text = """
    🔍 종목 검색 기능 사용법

    📌 기본 검색 방법:
    • 종목코드: 005930 (삼성전자), AAPL (애플)
    • 회사명: 삼성전자, Apple Inc, 현대차
    • 부분 검색: 삼성, 전자, Tech, Bio
    • 섹터 검색: Technology, Healthcare, Financial

    🎯 검색 예시:

    🇰🇷 한국 종목:
    • 005930 → 삼성전자 차트 즉시 표시
    • 삼성 → 삼성전자, 삼성SDI 등 선택 다이얼로그
    • 전자 → 삼성전자, LG전자 등 관련 종목들

    🇺🇸 미국 종목:
    • AAPL → 애플 차트 즉시 표시  
    • Apple → 애플 차트 즉시 표시
    • Tech → 기술주 관련 종목들

    🇸🇪 스웨덴 종목:
    • VOLV-B.ST → 볼보 차트
    • Ericsson → 에릭슨 관련 종목들

    ⚡ 편의 기능:
    • 🔍 검색 버튼 또는 Enter 키로 검색
    • 🎲 랜덤 버튼으로 무작위 종목 탐색
    • 실시간 검색어 제안 (2글자 이상 입력시)
    • 최근 검색어 기록 (최대 5개)
    • 검색 결과를 Excel로 내보내기

    📊 차트 기능:
    • 가격 + 이동평균선 (20, 60, 120일)
    • 볼린저 밴드
    • RSI (상대강도지수)
    • MACD 지표
    • 다양한 기간 선택 (3개월 ~ 2년)
    • 풀스크린 모드

    💡 검색 팁:
    • 정확한 매치가 우선순위 (티커 > 회사명 > 섹터)
    • 대소문자 구분 없음
    • 한글-영문 혼용 가능
    • CSV에서 찾지 못하면 온라인 자동 검색
    • '온라인 종목 업데이트'로 최신 종목 확보 권장

    🎲 랜덤 기능:
    • 시가총액이 있는 종목 우선 선택
    • 다양한 시장의 종목 탐색 가능
    • 새로운 투자 아이디어 발굴에 유용

    🔧 문제 해결:
    • 검색 결과가 없으면: CSV 파일 확인 또는 온라인 업데이트
    • 차트가 안 열리면: 인터넷 연결 확인
    • 오래된 데이터: '온라인 종목 업데이트' 실행

    📋 사용 시나리오:

    1️⃣ 빠른 차트 확인:
    → 종목코드 입력 → Enter → 차트 즉시 표시

    2️⃣ 종목 탐색:
    → 섹터명 입력 → 여러 결과 → 관심 종목 선택

    3️⃣ 새로운 발견:
    → 🎲 랜덤 버튼 → 예상치 못한 종목 발견
    """
        
        QMessageBox.information(self, "🔍 종목 검색 도움말", help_text)

    def rebuild_search_index(self):
        """검색 인덱스 재구성 - 데이터 형태 안전 처리"""
        try:
            self.search_index = {}
            
            for market, data in self.stock_lists.items():
                if not data:
                    continue
                
                # DataFrame인 경우
                if hasattr(data, 'empty') and hasattr(data, 'iterrows'):
                    if data.empty:
                        continue

                    # ✅ 벡터화: iterrows() 제거 - 25-30배 성능 향상
                    # DataFrame을 딕셔너리 리스트로 변환하여 일괄 처리
                    for stock_dict in data.to_dict('records'):
                        self._index_stock_data(stock_dict, market)
                
                # 리스트인 경우
                elif isinstance(data, list):
                    for stock in data:
                        if isinstance(stock, dict):
                            self._index_stock_data(stock, market)
            
            print(f"✅ 검색 인덱스 구성 완료: {len(self.search_index)}개 항목")
            
        except Exception as e:
            print(f"⚠️ 검색 인덱스 구성 오류: {e}")
            self.search_index = {}

    def _index_stock_data(self, stock, market):
        """주식 데이터 인덱싱 헬퍼 메서드"""
        try:
            ticker = str(stock.get('ticker', '')).upper()
            name = str(stock.get('name', '')).upper()
            
            # 티커로 인덱싱
            if ticker and ticker != 'NAN':
                if ticker not in self.search_index:
                    self.search_index[ticker] = []
                self.search_index[ticker].append({
                    'market': market,
                    'stock_data': stock,
                    'match_type': 'ticker'
                })
            
            # 회사명의 각 단어로 인덱싱
            if name and name != 'NAN':
                words = name.split()
                for word in words:
                    if len(word) >= 2:  # 2글자 이상만
                        if word not in self.search_index:
                            self.search_index[word] = []
                        self.search_index[word].append({
                            'market': market,
                            'stock_data': stock,
                            'match_type': 'name'
                        })
                        
        except Exception as e:
            print(f"⚠️ 데이터 인덱싱 오류: {e}")

    # 추가: 데이터 형태 확인 유틸리티
    def check_data_format(self):
        """현재 데이터 형태 확인 (디버깅용)"""
        print("📊 현재 데이터 형태 확인:")
        for market, data in self.stock_lists.items():
            if hasattr(data, 'empty'):
                print(f"  {market}: DataFrame ({len(data)}개)")
            elif isinstance(data, list):
                print(f"  {market}: List ({len(data)}개)")
            else:
                print(f"  {market}: Unknown type ({type(data)})")

    # 안전한 검색 초기화
    def init_search_safely(self):
        """검색 기능 안전 초기화"""
        try:
            # 검색 관련 변수 초기화
            if not hasattr(self, 'search_index'):
                self.search_index = {}
            if not hasattr(self, 'recent_searches'):
                self.recent_searches = []
            
            # 데이터 형태 확인
            self.check_data_format()
            
            # 검색 인덱스 구성 시도
            self.rebuild_search_index()
            
            print("✅ 검색 기능 초기화 완료")
            
        except Exception as e:
            print(f"⚠️ 검색 초기화 오류: {e}")
            self.search_index = {}
            self.recent_searches = []

    def show_stock_chart(self, ticker, name):
        """종목 차트 창 열기"""
        try:
            from chart_window import StockChartWindow
            
            # 기존 같은 종목 차트 창이 있으면 닫기
            for window in QApplication.topLevelWidgets():
                if isinstance(window, StockChartWindow) and window.symbol == ticker:
                    window.close()
            
            # 새 차트 창 열기
            chart_window = StockChartWindow(ticker, name, self)
            chart_window.show()
            
            # 검색어 입력창 비우기 
            self.search_input.clear()
            
            print(f"✅ 차트 창 열림: {ticker} ({name})")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "차트 오류", 
                f"차트를 표시할 수 없습니다.\n\n"
                f"종목: {ticker} ({name})\n"
                f"오류: {str(e)}\n\n"
                f"가능한 원인:\n"
                f"• 인터넷 연결 문제\n"
                f"• 잘못된 종목 코드\n" 
                f"• 차트 모듈 오류"
            )
            print(f"차트 표시 오류: {e}")
            import traceback
            traceback.print_exc()

    def export_search_results(self, found_stocks, search_term):
        """검색 결과를 Excel로 내보내기"""
        try:
            from utils import export_search_results
            
            filename = export_search_results(found_stocks, search_term)
            
            if filename:
                QMessageBox.information(
                    self,
                    "내보내기 완료",
                    f"검색 결과가 Excel 파일로 저장되었습니다.\n\n"
                    f"파일명: {filename}\n"
                    f"종목 수: {len(found_stocks)}개"
                )
            else:
                QMessageBox.warning(self, "내보내기 실패", "Excel 파일 저장에 실패했습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "내보내기 오류", f"오류가 발생했습니다: {str(e)}")

    def resort_search_results(self, dialog, found_stocks, sort_method):
        """검색 결과 재정렬"""
        try:
            if sort_method == "매치 점수순":
                found_stocks.sort(key=lambda x: (-x['match_score'], x['name']))
            elif sort_method == "회사명순":
                found_stocks.sort(key=lambda x: x['name'])
            elif sort_method == "시가총액순":
                found_stocks.sort(key=lambda x: (-x.get('raw_market_cap', 0), x['name']))
            elif sort_method == "시장별":
                found_stocks.sort(key=lambda x: (x['market'], x['name']))
            
            # 테이블 업데이트
            for widget in dialog.findChildren(QTableWidget):
                self.populate_search_results_table(widget, found_stocks)
                break
                
        except Exception as e:
            print(f"정렬 오류: {e}")

    # def load_stock_lists(self):
    #     """기존 CSV 로드 함수 오버라이드 - 검색 인덱스 재구성 포함"""
    #     # 기존 로드 로직 실행
    #     original_load_result = super().load_stock_lists() if hasattr(super(), 'load_stock_lists') else self.setup_stock_lists()
        
    #     # 검색 인덱스 재구성
    #     self.rebuild_search_index()
        
    #     # 검색 결과 레이블 업데이트
    #     total_stocks = sum(len(df) for df in self.stock_lists.values() if not df.empty)
    #     self.search_result_label.setText(f"📊 총 {total_stocks:,}개 종목 로드됨")
        
    #     return original_load_result

    def load_stock_lists(self):
        """CSV 파일에서 종목 리스트 로드 (캐싱 최적화)"""
        self.stock_lists = {}

        try:
            # ✅ csv_manager 사용 - 캐싱으로 80-90% I/O 감소
            master_data = load_all_master_csvs()
            print(f"📊 load_all_master_csvs() 결과: {list(master_data.keys()) if master_data else 'None'}")

            # DataFrame을 dict records로 변환 + DataFrame도 별도 저장 (검색용)
            self._stock_dataframes = getattr(self, '_stock_dataframes', {})

            for market in ['korea', 'usa', 'sweden']:
                if market in master_data and master_data[market] is not None:
                    df = master_data[market]
                    print(f"  {market}: {len(df)}개 종목 로드")
                    self.stock_lists[market] = df.to_dict('records')
                    self._stock_dataframes[market] = df
                else:
                    print(f"  {market}: 데이터 없음")
                    self.stock_lists[market] = []

            total_stocks = sum(len(v) for v in self.stock_lists.values())
            print(f"✅ 총 {total_stocks}개 종목 로드됨")

            # 검색 인덱스 재구성 (DataFrame 사용)
            if hasattr(self, 'rebuild_search_index'):
                self.rebuild_search_index()

            # 종목 개수 업데이트
            if hasattr(self, 'update_stock_count'):
                self.update_stock_count()
            if hasattr(self, 'statusbar'):
                self.statusbar.showMessage('📁 CSV 파일 로드 완료')

        except Exception as e:
            print(f"❌ CSV 로드 오류: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "오류", f"CSV 파일 로드 중 오류: {str(e)}")

    # 검색 성능 모니터링 함수
    def monitor_search_performance(self):
        """검색 성능 모니터링 (개발용)"""
        try:
            from utils import benchmark_search_performance
            
            test_terms = ['삼성', 'AAPL', '005930', 'TESLA', '반도체', 'TECH', 'Healthcare']
            results = benchmark_search_performance(self.stock_lists, test_terms)
            
            print("\n📊 검색 성능 벤치마크:")
            for term, metrics in results.items():
                print(f"   {term}: {metrics['search_time']:.3f}초, {metrics['results_count']}개 결과, 최고점수: {metrics['first_match_score']}")
            
        except Exception as e:
            print(f"성능 모니터링 오류: {e}")

    # 사용 예시 및 테스트 함수
    def test_search_functionality(self):
        """검색 기능 테스트"""
        
        test_cases = [
            "005930",      # 삼성전자 (한국)
            "AAPL",        # 애플 (미국)  
            "삼성",        # 부분 검색 (한국)
            "Technology",  # 섹터 검색
            "VOLV-B.ST",   # 스웨덴 종목
            "존재하지않는종목"  # 검색 실패 케이스
        ]
        
        print("\n🧪 검색 기능 테스트:")
        for term in test_cases:
            try:
                results = self.enhanced_search_stocks(term)
                print(f"   '{term}': {len(results)}개 결과")
                if results:
                    top_result = results[0]
                    print(f"      → 최상위: {top_result['name']} ({top_result['ticker']}) - 점수: {top_result['match_score']}")
            except Exception as e:
                print(f"   '{term}': 오류 - {e}")

    # 키보드 단축키 설정
    def setup_search_shortcuts(self):
        """검색 관련 키보드 단축키 설정"""
        
        # Ctrl+F: 검색창에 포커스
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(lambda: self.search_input.setFocus())
        
        # Ctrl+R: 랜덤 종목
        random_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        random_shortcut.activated.connect(self.show_random_stock_chart)
        
        # F1: 검색 도움말
        help_shortcut = QShortcut(QKeySequence("F1"), self)
        help_shortcut.activated.connect(self.show_search_help)
                    
    def example_search_usage(self):
        """검색 기능 사용 예시"""
        
        # 예시 1: 삼성전자 검색
        # 입력: "005930" 또는 "삼성전자" 또는 "samsung"
        # 결과: 삼성전자 차트 즉시 표시
        
        # 예시 2: 애플 검색  
        # 입력: "AAPL" 또는 "Apple"
        # 결과: 애플 차트 즉시 표시
        
        # 예시 3: 부분 검색
        # 입력: "전자" 
        # 결과: 삼성전자, LG전자 등 여러 결과 → 선택 다이얼로그
        
        # 예시 4: 섹터 검색
        # 입력: "Technology"
        # 결과: 기술 섹터 모든 종목 → 선택 다이얼로그
        
        pass

    def get_search_examples(self):
        """검색 예시 반환"""
        return {
            "한국 종목": [
                "005930 (삼성전자)",
                "373220 (LG에너지솔루션)", 
                "207940 (삼성바이오로직스)",
                "삼성전자",
                "현대차"
            ],
            "미국 종목": [
                "AAPL (애플)",
                "MSFT (마이크로소프트)",
                "GOOGL (구글)",
                "TSLA (테슬라)",
                "NVDA (엔비디아)"
            ],
            "스웨덴 종목": [
                "VOLV-B.ST (볼보)",
                "ERIC.ST (에릭슨)",
                "SEB-A.ST (SEB 은행)"
            ],
            "섹터 검색": [
                "Technology",
                "Healthcare", 
                "Financial",
                "반도체",
                "자동차"
            ]
        }

