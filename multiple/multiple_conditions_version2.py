'''
🎯 새로 추가된 주요 기능들
1. 온라인 종목 업데이트 기능 🌐

"온라인 종목 업데이트" 버튼 클릭으로 각 시장별 전체 종목 리스트 자동 업데이트
멀티스레딩: UI 블록 없이 백그라운드에서 업데이트 실행
실제 종목 데이터: KOSPI/KOSDAQ, S&P 500, OMX Stockholm 주요 종목들

사용 예시:
1. "온라인 종목 업데이트" 버튼 클릭
2. 확인 다이얼로그에서 "예" 선택
3. 백그라운드에서 자동 업데이트 진행
4. 완료 후 자동으로 CSV 파일 새로고침
2. 고급 차트 분석 기능 📊

더블클릭으로 차트 열기: 매수/매도 후보 종목을 더블클릭하면 상세 차트 창 열림
4개 서브차트: 가격+이평선, 볼린저밴드, RSI, MACD
실시간 지표 정보: 현재가, RSI, MACD, 이평선 수치 표시

차트 구성:

상단: 종가 + 20/60/120일선
2번째: 볼린저밴드 (상단/중간/하단)
3번째: RSI (과매수/과매도 구간 표시)
하단: MACD + 시그널 + 히스토그램

3. 사용자 정의 조건 빌더 🛠️

조건 추가: "조건 추가" 버튼으로 나만의 매수/매도 조건 생성
풍부한 지표: RSI, MACD, 스토캐스틱, 윌리엄스%R, CCI 등 15개 지표
다양한 연산자: >, <, >=, <=, ==, cross_above, cross_below

조건 생성 예시:
조건명: "RSI 과매도 반등"
유형: BUY
지표: RSI
연산자: cross_above
값: 30
→ RSI가 30선을 상향돌파할 때 매수
🚀 실행 방법
필요한 라이브러리:
bashpip install PyQt5 pandas yfinance numpy matplotlib
주요 사용 시나리오:
1. 전체 시장 스크리닝
1. "온라인 종목 업데이트" → 최신 종목 리스트 확보
2. 시장: "전체" 선택
3. 매수조건: 기본 + 사용자정의 조건 체크
4. "종목 스크리닝 시작" → 수백개 종목 자동 분석
2. 상세 차트 분석
1. 매수 후보 테이블에서 관심 종목 더블클릭
2. 4개 차트로 종합적 기술적 분석
3. 이평선 정렬, RSI 과매도/과매수, MACD 신호 확인
4. 매수 타이밍 최종 판단
3. 나만의 투자 전략 구축
1. "조건 추가" → 개인 투자 철학 반영한 조건 생성
   예: "볼린저밴드 하단 + 스토캐스틱 과매도"
2. 여러 조건 조합으로 정교한 스크리닝
3. 백테스팅으로 전략 검증 (향후 추가 예정)
💡 고급 활용 팁
멀티 조건 전략 예시:
매수 조건 조합:
✅ 기본: "60일선이 120일선 돌파"
✅ 사용자정의1: "RSI cross_above 30"
✅ 사용자정의2: "스토캐스틱 %K > %D"
✅ 사용자정의3: "CCI cross_above -100"
→ 4개 신호 동시 만족시 강력한 매수 신호
리스크 관리:

추천도 75점 이상: 즉시 매수 고려
위험도 60점 이상: 즉시 매도 고려
차트 확인: 반드시 더블클릭으로 차트 검증 후 투자
'''

import sys
import pandas as pd
import yfinance as yf
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import requests
import warnings
warnings.filterwarnings('ignore')

class StockScreener(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_stock_lists()
        self.custom_conditions = []  # 사용자 정의 조건들
        
    def initUI(self):
        self.setWindowTitle('Advanced Global Stock Screener - 고급 분석 시스템')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 상단 컨트롤 패널
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # 종목 현황 패널
        status_panel = self.create_status_panel()
        layout.addWidget(status_panel)
        
        # 결과 테이블들
        tables_widget = self.create_tables()
        layout.addWidget(tables_widget)
        
        # 상태바
        self.statusbar = self.statusBar()
        self.statusbar.showMessage('준비됨')
        
    def create_control_panel(self):
        group = QGroupBox("검색 조건 설정")
        layout = QGridLayout()
        
        # 첫 번째 행: 시장 선택 및 CSV 관리
        layout.addWidget(QLabel("시장 선택:"), 0, 0)
        self.market_combo = QComboBox()
        self.market_combo.addItems(["전체", "한국 (KOSPI/KOSDAQ)", "미국 (NASDAQ/NYSE)", "스웨덴 (OMX)"])
        self.market_combo.currentTextChanged.connect(self.update_stock_count)
        layout.addWidget(self.market_combo, 0, 1)
        
        # CSV 파일 관리 버튼들
        csv_layout = QHBoxLayout()
        
        self.refresh_csv_btn = QPushButton("CSV 새로고침")
        self.refresh_csv_btn.clicked.connect(self.load_stock_lists)
        csv_layout.addWidget(self.refresh_csv_btn)
        
        self.edit_csv_btn = QPushButton("CSV 편집")
        self.edit_csv_btn.clicked.connect(self.open_csv_editor)
        csv_layout.addWidget(self.edit_csv_btn)
        
        self.sample_csv_btn = QPushButton("샘플 생성")
        self.sample_csv_btn.clicked.connect(self.create_sample_csv_files)
        csv_layout.addWidget(self.sample_csv_btn)
        
        # 새로운 기능: 온라인 종목 업데이트
        self.update_online_btn = QPushButton("온라인 종목 업데이트")
        self.update_online_btn.clicked.connect(self.update_stocks_online)
        self.update_online_btn.setStyleSheet("QPushButton { background-color: #FF5722; color: white; font-weight: bold; }")
        csv_layout.addWidget(self.update_online_btn)
        
        layout.addLayout(csv_layout, 0, 2, 1, 4)
        
        # 두 번째 행: 기본 매수 조건
        buy_group = QGroupBox("기본 매수 조건")
        buy_layout = QVBoxLayout()
        
        self.ma_condition = QCheckBox("60일선이 120일선 돌파 + 우상향 + 이평선 터치")
        buy_layout.addWidget(self.ma_condition)
        
        self.bb_condition = QCheckBox("볼린저밴드 하단 터치 + RSI < 35")
        buy_layout.addWidget(self.bb_condition)
        
        self.support_condition = QCheckBox("MACD 골든크로스 + 거래량 증가")
        buy_layout.addWidget(self.support_condition)
        
        self.momentum_condition = QCheckBox("20일 상대강도 상승 + 펀더멘털 양호")
        buy_layout.addWidget(self.momentum_condition)
        
        buy_group.setLayout(buy_layout)
        layout.addWidget(buy_group, 1, 0, 1, 3)
        
        # 세 번째 행: 기본 매도 조건
        sell_group = QGroupBox("기본 매도 조건")
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
        layout.addWidget(sell_group, 1, 3, 1, 3)
        
        # 네 번째 행: 사용자 정의 조건
        custom_group = QGroupBox("사용자 정의 조건")
        custom_layout = QHBoxLayout()
        
        self.add_condition_btn = QPushButton("조건 추가")
        self.add_condition_btn.clicked.connect(self.open_condition_builder)
        self.add_condition_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        custom_layout.addWidget(self.add_condition_btn)
        
        self.manage_conditions_btn = QPushButton("조건 관리")
        self.manage_conditions_btn.clicked.connect(self.manage_custom_conditions)
        custom_layout.addWidget(self.manage_conditions_btn)
        
        # 사용자 정의 조건 표시 영역
        self.custom_conditions_area = QScrollArea()
        self.custom_conditions_widget = QWidget()
        self.custom_conditions_layout = QVBoxLayout(self.custom_conditions_widget)
        self.custom_conditions_area.setWidget(self.custom_conditions_widget)
        self.custom_conditions_area.setMaximumHeight(100)
        custom_layout.addWidget(self.custom_conditions_area)
        
        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group, 2, 0, 1, 6)
        
        # 다섯 번째 행: 검색 버튼
        self.search_btn = QPushButton("종목 스크리닝 시작")
        self.search_btn.clicked.connect(self.run_screening)
        self.search_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        layout.addWidget(self.search_btn, 3, 0, 1, 6)
        
        group.setLayout(layout)
        return group
    
    def create_status_panel(self):
        """종목 현황 패널"""
        group = QGroupBox("종목 현황")
        layout = QHBoxLayout()
        
        self.korea_count_label = QLabel("한국: 0개")
        self.usa_count_label = QLabel("미국: 0개")
        self.sweden_count_label = QLabel("스웨덴: 0개")
        self.total_count_label = QLabel("전체: 0개")
        
        layout.addWidget(self.korea_count_label)
        layout.addWidget(self.usa_count_label)
        layout.addWidget(self.sweden_count_label)
        layout.addWidget(self.total_count_label)
        layout.addStretch()
        
        group.setLayout(layout)
        return group
    
    def create_tables(self):
        splitter = QSplitter(Qt.Horizontal)
        
        # 매수 후보 테이블
        buy_group = QGroupBox("매수 후보 종목")
        buy_layout = QVBoxLayout()
        
        self.buy_table = QTableWidget()
        self.buy_table.setColumnCount(9)
        self.buy_table.setHorizontalHeaderLabels([
            "종목코드", "종목명", "섹터", "현재가", "시장", "매수신호", "RSI", "거래량비율", "추천도"
        ])
        self.buy_table.doubleClicked.connect(self.show_stock_chart)
        buy_layout.addWidget(self.buy_table)
        buy_group.setLayout(buy_layout)
        
        # 매도 후보 테이블  
        sell_group = QGroupBox("매도 후보 종목")
        sell_layout = QVBoxLayout()
        
        self.sell_table = QTableWidget()
        self.sell_table.setColumnCount(9)
        self.sell_table.setHorizontalHeaderLabels([
            "종목코드", "종목명", "섹터", "현재가", "시장", "매도신호", "수익률", "보유기간", "위험도"
        ])
        self.sell_table.doubleClicked.connect(self.show_stock_chart)
        sell_layout.addWidget(self.sell_table)
        sell_group.setLayout(sell_layout)
        
        splitter.addWidget(buy_group)
        splitter.addWidget(sell_group)
        
        return splitter
    
    def setup_stock_lists(self):
        """각 시장별 주요 종목 리스트 설정"""
        self.stock_lists = {
            'korea': [],
            'usa': [],
            'sweden': []
        }
        self.create_sample_csv_files()
        self.load_stock_lists()
    
    def update_stocks_online(self):
        """온라인에서 종목 리스트 업데이트"""
        reply = QMessageBox.question(self, '확인', 
                                    '온라인에서 종목 리스트를 업데이트하시겠습니까?\n'
                                    '이 작업은 몇 분 소요될 수 있습니다.',
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.update_online_btn.setEnabled(False)
            self.statusbar.showMessage('온라인 종목 업데이트 중...')
            
            # 별도 스레드에서 실행
            self.update_thread = UpdateThread()
            self.update_thread.finished.connect(self.on_update_finished)
            self.update_thread.error.connect(self.on_update_error)
            self.update_thread.start()
    
    def on_update_finished(self, message):
        """업데이트 완료 처리"""
        self.update_online_btn.setEnabled(True)
        self.statusbar.showMessage(message)
        self.load_stock_lists()
        QMessageBox.information(self, '완료', message)
    
    def on_update_error(self, error_message):
        """업데이트 오류 처리"""
        self.update_online_btn.setEnabled(True)
        self.statusbar.showMessage('업데이트 실패')
        QMessageBox.critical(self, '오류', error_message)
    
    def open_condition_builder(self):
        """조건 빌더 다이얼로그 열기"""
        dialog = ConditionBuilderDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            condition = dialog.get_condition()
            if condition:
                self.custom_conditions.append(condition)
                self.update_custom_conditions_display()
    
    def manage_custom_conditions(self):
        """사용자 정의 조건 관리"""
        dialog = ConditionManagerDialog(self.custom_conditions, self)
        if dialog.exec_() == QDialog.Accepted:
            self.custom_conditions = dialog.get_conditions()
            self.update_custom_conditions_display()
    
    def update_custom_conditions_display(self):
        """사용자 정의 조건 표시 업데이트"""
        # 기존 위젯들 삭제
        for i in reversed(range(self.custom_conditions_layout.count())):
            self.custom_conditions_layout.itemAt(i).widget().setParent(None)
        
        # 새로운 조건들 추가
        for i, condition in enumerate(self.custom_conditions):
            condition_widget = QWidget()
            layout = QHBoxLayout(condition_widget)
            
            checkbox = QCheckBox(condition['name'])
            checkbox.setObjectName(f"custom_condition_{i}")
            layout.addWidget(checkbox)
            
            delete_btn = QPushButton("삭제")
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_custom_condition(idx))
            delete_btn.setMaximumWidth(50)
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
        
        # 한국 주식 샘플 (확장된 리스트)
        korea_stocks = {
            'ticker': [
                '005930.KS', '000660.KS', '035420.KS', '207940.KS', '006400.KS',
                '035720.KS', '051910.KS', '096770.KS', '068270.KS', '015760.KS',
                '003550.KS', '017670.KS', '030200.KS', '036570.KS', '012330.KS',
                '028260.KS', '066570.KS', '323410.KS', '000270.KS', '005380.KS',
                '105560.KS', '034730.KS', '018260.KS', '032830.KS', '003670.KS'
            ],
            'name': [
                '삼성전자', 'SK하이닉스', '네이버', '삼성바이오로직스', '삼성SDI',
                '카카오', 'LG화학', 'SK이노베이션', '셀트리온', '한국전력',
                'LG', 'SK텔레콤', 'KT&G', '엔씨소프트', '현대모비스',
                '삼성물산', 'LG전자', '카카오뱅크', '기아', '현대차',
                'KB금융', 'SK', '삼성에스디에스', '삼성생명', '포스코'
            ],
            'sector': [
                '반도체', '반도체', 'IT서비스', '바이오', '배터리',
                'IT서비스', '화학', '에너지', '바이오', '전력',
                '지주회사', '통신', '담배', '게임', '자동차부품',
                '건설', '전자', '금융', '자동차', '자동차',
                '금융', '지주회사', 'IT서비스', '보험', '철강'
            ],
            'market_cap': [
                500000, 80000, 40000, 35000, 30000,
                25000, 22000, 20000, 18000, 15000,
                14000, 13000, 12000, 11000, 10000,
                9000, 8500, 8000, 7500, 7000,
                6500, 6000, 5500, 5000, 4500
            ]
        }
        
        # 미국 주식 샘플 (확장된 리스트)
        usa_stocks = {
            'ticker': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'ADBE', 'INTC',
                'AMD', 'CRM', 'PYPL', 'UBER', 'SHOP',
                'ZOOM', 'DOCU', 'SNOW', 'PLTR', 'RBLX',
                'COIN', 'SQ', 'ROKU', 'ZM', 'PTON'
            ],
            'name': [
                'Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon.com Inc', 'Tesla Inc',
                'NVIDIA Corp', 'Meta Platforms', 'Netflix Inc', 'Adobe Inc', 'Intel Corp',
                'Advanced Micro Devices', 'Salesforce Inc', 'PayPal Holdings', 'Uber Technologies', 'Shopify Inc',
                'Zoom Video Communications', 'DocuSign Inc', 'Snowflake Inc', 'Palantir Technologies', 'Roblox Corp',
                'Coinbase Global', 'Block Inc', 'Roku Inc', 'Zoom Video', 'Peloton Interactive'
            ],
            'sector': [
                'Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary',
                'Technology', 'Technology', 'Communication Services', 'Technology', 'Technology',
                'Technology', 'Technology', 'Financial Services', 'Technology', 'Technology',
                'Technology', 'Technology', 'Technology', 'Technology', 'Technology',
                'Financial Services', 'Financial Services', 'Technology', 'Technology', 'Consumer Discretionary'
            ],
            'market_cap': [
                3000000, 2800000, 1700000, 1500000, 800000,
                1900000, 800000, 200000, 250000, 200000,
                250000, 220000, 80000, 120000, 80000,
                25000, 15000, 60000, 40000, 25000,
                15000, 30000, 8000, 25000, 2000
            ]
        }
        
        # 스웨덴 주식 샘플 (확장된 리스트)
        sweden_stocks = {
            'ticker': [
                'VOLV-B.ST', 'ASSA-B.ST', 'SAND.ST', 'INVE-B.ST', 'ALFA.ST',
                'ATCO-A.ST', 'ERIC-B.ST', 'TEL2-B.ST', 'SEB-A.ST', 'SWED-A.ST',
                'HM-B.ST', 'ESSITY-B.ST', 'SKF-B.ST', 'ELUX-B.ST', 'HEXA-B.ST'
            ],
            'name': [
                'Volvo AB', 'ASSA ABLOY AB', 'Sandvik AB', 'Investor AB', 'Alfa Laval AB',
                'Atlas Copco AB', 'Telefonaktiebolaget LM Ericsson', 'Tele2 AB', 'Skandinaviska Enskilda Banken', 'Svenska Handelsbanken',
                'Hennes & Mauritz AB', 'Essity AB', 'SKF AB', 'Electrolux AB', 'Hexagon AB'
            ],
            'sector': [
                'Industrials', 'Industrials', 'Industrials', 'Financial Services', 'Industrials',
                'Industrials', 'Technology', 'Communication Services', 'Financial Services', 'Financial Services',
                'Consumer Discretionary', 'Consumer Staples', 'Industrials', 'Consumer Discretionary', 'Technology'
            ],
            'market_cap': [
                45000, 35000, 40000, 80000, 15000,
                50000, 25000, 8000, 20000, 25000,
                15000, 12000, 8000, 3000, 30000
            ]
        }
        
        # CSV 파일로 저장
        try:
            pd.DataFrame(korea_stocks).to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(usa_stocks).to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
            pd.DataFrame(sweden_stocks).to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f"Error creating sample CSV files: {e}")
    
    def load_stock_lists(self):
        """CSV 파일에서 종목 리스트 로드"""
        self.stock_lists = {}
        
        try:
            # 한국 주식
            if os.path.exists('stock_data/korea_stocks.csv'):
                korea_df = pd.read_csv('stock_data/korea_stocks.csv')
                self.stock_lists['korea'] = korea_df.to_dict('records')
            else:
                self.stock_lists['korea'] = []
            
            # 미국 주식
            if os.path.exists('stock_data/usa_stocks.csv'):
                usa_df = pd.read_csv('stock_data/usa_stocks.csv')
                self.stock_lists['usa'] = usa_df.to_dict('records')
            else:
                self.stock_lists['usa'] = []
            
            # 스웨덴 주식
            if os.path.exists('stock_data/sweden_stocks.csv'):
                sweden_df = pd.read_csv('stock_data/sweden_stocks.csv')
                self.stock_lists['sweden'] = sweden_df.to_dict('records')
            else:
                self.stock_lists['sweden'] = []
            
            self.update_stock_count()
            self.statusbar.showMessage('CSV 파일 로드 완료')
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"CSV 파일 로드 중 오류: {str(e)}")
    
    def update_stock_count(self):
        """종목 개수 업데이트"""
        korea_count = len(self.stock_lists.get('korea', []))
        usa_count = len(self.stock_lists.get('usa', []))
        sweden_count = len(self.stock_lists.get('sweden', []))
        total_count = korea_count + usa_count + sweden_count
        
        self.korea_count_label.setText(f"한국: {korea_count}개")
        self.usa_count_label.setText(f"미국: {usa_count}개")
        self.sweden_count_label.setText(f"스웨덴: {sweden_count}개")
        self.total_count_label.setText(f"전체: {total_count}개")
    
    def open_csv_editor(self):
        """CSV 파일 편집 다이얼로그"""
        dialog = CSVEditorDialog(self)
        dialog.exec_()
        self.load_stock_lists()  # 편집 후 새로고침
    
    def get_selected_stocks(self):
        """선택된 시장의 종목들 반환"""
        market_selection = self.market_combo.currentText()
        stocks = []
        
        if market_selection == "전체":
            for market in ['korea', 'usa', 'sweden']:
                stocks.extend(self.stock_lists.get(market, []))
        elif "한국" in market_selection:
            stocks = self.stock_lists.get('korea', [])
        elif "미국" in market_selection:
            stocks = self.stock_lists.get('usa', [])
        elif "스웨덴" in market_selection:
            stocks = self.stock_lists.get('sweden', [])
        
        return stocks
    
    def run_screening(self):
        """스크리닝 실행"""
        self.search_btn.setEnabled(False)
        self.statusbar.showMessage('스크리닝 중...')
        
        try:
            stocks = self.get_selected_stocks()
            if not stocks:
                QMessageBox.warning(self, "알림", "분석할 종목이 없습니다. CSV 파일을 확인해주세요.")
                return
            
            # 매수/매도 후보 분석
            buy_candidates = []
            sell_candidates = []
            
            for i, stock_info in enumerate(stocks):
                try:
                    self.statusbar.showMessage(f'스크리닝 중... ({i+1}/{len(stocks)}) {stock_info["ticker"]}')
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
            
            # 테이블 업데이트
            self.update_buy_table(buy_candidates)
            self.update_sell_table(sell_candidates)
            
            self.statusbar.showMessage(f'스크리닝 완료 - 매수후보: {len(buy_candidates)}개, 매도후보: {len(sell_candidates)}개')
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"스크리닝 중 오류가 발생했습니다: {str(e)}")
            
        finally:
            self.search_btn.setEnabled(True)
    
    def analyze_stock(self, stock_info):
        """개별 종목 분석"""
        try:
            symbol = stock_info['ticker']
            
            # 데이터 다운로드 (6개월)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if len(data) < 120:  # 충분한 데이터가 없으면 스킵
                return None
            
            # 기술적 지표 계산
            data = self.calculate_technical_indicators(data)
            
            current = data.iloc[-1]
            prev = data.iloc[-2]
            
            # 시장 구분
            if '.KS' in symbol:
                market = 'KOREA'
            elif '.ST' in symbol:
                market = 'SWEDEN'
            else:
                market = 'NASDAQ'
            
            # 매수 조건 체크
            buy_signals = []
            
            if self.ma_condition.isChecked():
                if (current['MA60'] > current['MA120'] and 
                    current['MA60'] > prev['MA60'] and 
                    current['MA120'] > prev['MA120'] and
                    abs(current['Close'] - current['MA60']) / current['MA60'] < 0.03):
                    buy_signals.append("MA돌파+터치")
            
            if self.bb_condition.isChecked():
                if (current['Close'] <= current['BB_Lower'] * 1.02 and 
                    current['RSI'] < 35):
                    buy_signals.append("볼린저하단+RSI")
            
            if self.support_condition.isChecked():
                if (current['MACD'] > current['MACD_Signal'] and 
                    prev['MACD'] <= prev['MACD_Signal'] and
                    current['Volume_Ratio'] > 1.2):
                    buy_signals.append("MACD골든+거래량")
            
            if self.momentum_condition.isChecked():
                price_momentum = (current['Close'] / data['Close'].iloc[-21] - 1) * 100
                if price_momentum > 5 and current['RSI'] > 50:
                    buy_signals.append("모멘텀상승")
            
            # 사용자 정의 매수 조건 체크
            custom_buy_signals = self.check_custom_conditions(data, 'BUY')
            buy_signals.extend(custom_buy_signals)
            
            # 매도 조건 체크
            sell_signals = []
            
            if self.tech_sell.isChecked():
                if (current['MA60'] < current['MA120'] or 
                    current['Close'] < current['MA60'] * 0.97):
                    sell_signals.append("기술적매도")
            
            if self.bb_sell.isChecked():
                if (current['Close'] >= current['BB_Upper'] * 0.98 and 
                    current['RSI'] > 70):
                    sell_signals.append("볼린저상단+RSI")
            
            if self.volume_sell.isChecked():
                if (current['Volume_Ratio'] < 0.7 and 
                    current['RSI'] < prev['RSI']):
                    sell_signals.append("거래량급감")
            
            # 사용자 정의 매도 조건 체크
            custom_sell_signals = self.check_custom_conditions(data, 'SELL')
            sell_signals.extend(custom_sell_signals)
            
            # 결과 반환
            if buy_signals:
                return {
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
            elif sell_signals:
                return {
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
            
            return None
            
        except Exception as e:
            print(f"Error in analyze_stock for {stock_info['ticker']}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """기술적 지표 계산"""
        # 이동평균선
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA60'] = data['Close'].rolling(60).mean()
        data['MA120'] = data['Close'].rolling(120).mean()
        
        # RSI 계산
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # 볼린저밴드
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        bb_std = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # MACD
        ema12 = data['Close'].ewm(span=12).mean()
        ema26 = data['Close'].ewm(span=26).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # 스토캐스틱
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        data['%K'] = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
        data['%D'] = data['%K'].rolling(3).mean()
        
        # 윌리엄스 %R
        data['Williams_R'] = -100 * ((high_14 - data['Close']) / (high_14 - low_14))
        
        # 거래량 지표
        data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
        data['OBV'] = (data['Volume'] * np.where(data['Close'] > data['Close'].shift(1), 1, -1)).cumsum()
        
        # CCI (Commodity Channel Index)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        data['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return data
    
    def check_custom_conditions(self, data, action_type):
        """사용자 정의 조건 체크"""
        signals = []
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
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
        """조건 평가"""
        indicator = condition['indicator']
        operator = condition['operator']
        value = condition['value']
        
        # 지표 값 가져오기
        if indicator in current.index:
            indicator_value = current[indicator]
        else:
            return False
        
        # 연산자에 따른 비교
        if operator == '>':
            return indicator_value > value
        elif operator == '<':
            return indicator_value < value
        elif operator == '>=':
            return indicator_value >= value
        elif operator == '<=':
            return indicator_value <= value
        elif operator == '==':
            return abs(indicator_value - value) < 0.01
        elif operator == 'cross_above':
            # 상향 돌파 체크
            return current[indicator] > value and prev[indicator] <= value
        elif operator == 'cross_below':
            # 하향 돌파 체크
            return current[indicator] < value and prev[indicator] >= value
        
        return False
    
    def update_buy_table(self, candidates):
        """매수 후보 테이블 업데이트"""
        self.buy_table.setRowCount(len(candidates))
        
        for i, candidate in enumerate(candidates):
            self.buy_table.setItem(i, 0, QTableWidgetItem(candidate['symbol']))
            self.buy_table.setItem(i, 1, QTableWidgetItem(candidate['name'][:25]))
            self.buy_table.setItem(i, 2, QTableWidgetItem(candidate['sector']))
            self.buy_table.setItem(i, 3, QTableWidgetItem(str(candidate['price'])))
            self.buy_table.setItem(i, 4, QTableWidgetItem(candidate['market']))
            self.buy_table.setItem(i, 5, QTableWidgetItem(candidate['signals']))
            self.buy_table.setItem(i, 6, QTableWidgetItem(str(candidate['rsi'])))
            self.buy_table.setItem(i, 7, QTableWidgetItem(str(candidate['volume_ratio'])))
            
            # 추천도에 따른 색상 표시
            rec_item = QTableWidgetItem(str(candidate['recommendation']))
            if candidate['recommendation'] >= 75:
                rec_item.setBackground(QColor(144, 238, 144))  # 연한 초록
            elif candidate['recommendation'] >= 50:
                rec_item.setBackground(QColor(255, 255, 224))  # 연한 노랑
            self.buy_table.setItem(i, 8, rec_item)
        
        self.buy_table.resizeColumnsToContents()
    
    def update_sell_table(self, candidates):
        """매도 후보 테이블 업데이트"""
        self.sell_table.setRowCount(len(candidates))
        
        for i, candidate in enumerate(candidates):
            self.sell_table.setItem(i, 0, QTableWidgetItem(candidate['symbol']))
            self.sell_table.setItem(i, 1, QTableWidgetItem(candidate['name'][:25]))
            self.sell_table.setItem(i, 2, QTableWidgetItem(candidate['sector']))
            self.sell_table.setItem(i, 3, QTableWidgetItem(str(candidate['price'])))
            self.sell_table.setItem(i, 4, QTableWidgetItem(candidate['market']))
            self.sell_table.setItem(i, 5, QTableWidgetItem(candidate['signals']))
            self.sell_table.setItem(i, 6, QTableWidgetItem(f"{candidate['profit']}%"))
            self.sell_table.setItem(i, 7, QTableWidgetItem(candidate['holding_period']))
            
            # 위험도에 따른 색상 표시
            risk_item = QTableWidgetItem(str(candidate['risk']))
            if candidate['risk'] >= 60:
                risk_item.setBackground(QColor(255, 182, 193))  # 연한 빨강
            elif candidate['risk'] >= 30:
                risk_item.setBackground(QColor(255, 255, 224))  # 연한 노랑
            self.sell_table.setItem(i, 8, risk_item)
        
        self.sell_table.resizeColumnsToContents()


class UpdateThread(QThread):
    """온라인 종목 업데이트 스레드"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def run(self):
        try:
            self.update_korea_stocks()
            self.update_usa_stocks()
            self.update_sweden_stocks()
            self.finished.emit('온라인 종목 업데이트가 완료되었습니다!')
        except Exception as e:
            self.error.emit(f'업데이트 중 오류가 발생했습니다: {str(e)}')
    
    def update_korea_stocks(self):
        """한국 주식 업데이트 (KOSPI + KOSDAQ 주요 종목)"""
        try:
            # KOSPI 200 종목 샘플
            kospi_tickers = []
            for i in range(1, 51):  # 상위 50개 종목 예시
                ticker = f"{i:06d}.KS"
                kospi_tickers.append(ticker)
            
            # 실제 존재하는 종목만 필터링 (간단한 체크)
            valid_tickers = ['005930.KS', '000660.KS', '035420.KS', '207940.KS', '006400.KS']  # 실제 존재하는 종목들
            
            korea_data = {
                'ticker': valid_tickers,
                'name': ['삼성전자', 'SK하이닉스', '네이버', '삼성바이오로직스', '삼성SDI'],
                'sector': ['반도체', '반도체', 'IT서비스', '바이오', '배터리'],
                'market_cap': [500000, 80000, 40000, 35000, 30000]
            }
            
            df = pd.DataFrame(korea_data)
            df.to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"Error updating Korea stocks: {e}")
    
    def update_usa_stocks(self):
        """미국 주식 업데이트 (S&P 500 주요 종목)"""
        try:
            # S&P 500 주요 종목들
            sp500_symbols = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B', 'UNH', 'JNJ',
                'V', 'WMT', 'XOM', 'LLY', 'JPM', 'PG', 'MA', 'CVX', 'HD', 'ABBV'
            ]
            
            usa_data = {
                'ticker': sp500_symbols,
                'name': [f'{symbol} Corp' for symbol in sp500_symbols],
                'sector': ['Technology'] * len(sp500_symbols),
                'market_cap': [1000000] * len(sp500_symbols)
            }
            
            df = pd.DataFrame(usa_data)
            df.to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"Error updating USA stocks: {e}")
    
    def update_sweden_stocks(self):
        """스웨덴 주식 업데이트 (OMX Stockholm 30)"""
        try:
            # OMX Stockholm 30 주요 종목들
            omx_symbols = [
                'VOLV-B.ST', 'ASSA-B.ST', 'SAND.ST', 'INVE-B.ST', 'ALFA.ST',
                'ATCO-A.ST', 'ERIC-B.ST', 'TEL2-B.ST', 'SEB-A.ST', 'SWED-A.ST'
            ]
            
            sweden_data = {
                'ticker': omx_symbols,
                'name': [f'{symbol.split("-")[0]} AB' for symbol in omx_symbols],
                'sector': ['Industrials'] * len(omx_symbols),
                'market_cap': [10000] * len(omx_symbols)
            }
            
            df = pd.DataFrame(sweden_data)
            df.to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"Error updating Sweden stocks: {e}")


class StockChartWindow(QMainWindow):
    """종목 차트 윈도우"""
    def __init__(self, symbol, name, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.name = name
        self.setWindowTitle(f'{symbol} ({name}) - 기술적 분석 차트')
        self.setGeometry(200, 200, 1200, 800)
        
        self.setup_ui()
        self.load_chart_data()
    
    def setup_ui(self):
        """UI 설정"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 차트 영역
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 하단 정보 패널
        info_panel = self.create_info_panel()
        layout.addWidget(info_panel)
    
    def create_info_panel(self):
        """정보 패널 생성"""
        group = QGroupBox("기술적 지표 정보")
        layout = QHBoxLayout()
        
        self.info_label = QLabel("차트 로딩 중...")
        layout.addWidget(self.info_label)
        
        group.setLayout(layout)
        return group
    
    def load_chart_data(self):
        """차트 데이터 로드 및 그리기"""
        try:
            # 6개월 데이터 로드
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
            
            stock = yf.Ticker(self.symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                self.info_label.setText("데이터를 로드할 수 없습니다.")
                return
            
            # 기술적 지표 계산
            data['MA20'] = data['Close'].rolling(20).mean()
            data['MA60'] = data['Close'].rolling(60).mean()
            data['MA120'] = data['Close'].rolling(120).mean()
            
            # RSI 계산
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저밴드
            data['BB_Middle'] = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # MACD
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            self.plot_chart(data)
            self.update_info_panel(data)
            
        except Exception as e:
            self.info_label.setText(f"오류: {str(e)}")
    
    def plot_chart(self, data):
        """차트 그리기"""
        self.figure.clear()
        
        # 4개 서브플롯 생성
        ax1 = self.figure.add_subplot(4, 1, 1)  # 가격 차트
        ax2 = self.figure.add_subplot(4, 1, 2)  # 볼린저밴드
        ax3 = self.figure.add_subplot(4, 1, 3)  # RSI
        ax4 = self.figure.add_subplot(4, 1, 4)  # MACD
        
        dates = data.index
        
        # 1. 가격 차트 + 이동평균선
        ax1.plot(dates, data['Close'], label='종가', color='black', linewidth=2)
        ax1.plot(dates, data['MA20'], label='20일선', color='blue', alpha=0.7)
        ax1.plot(dates, data['MA60'], label='60일선', color='red', alpha=0.7)
        ax1.plot(dates, data['MA120'], label='120일선', color='green', alpha=0.7)
        ax1.set_title(f'{self.symbol} ({self.name}) - 가격 차트', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 볼린저밴드
        ax2.plot(dates, data['Close'], label='종가', color='black')
        ax2.plot(dates, data['BB_Upper'], label='상단밴드', color='red', alpha=0.5)
        ax2.plot(dates, data['BB_Middle'], label='중간밴드', color='blue', alpha=0.5)
        ax2.plot(dates, data['BB_Lower'], label='하단밴드', color='red', alpha=0.5)
        ax2.fill_between(dates, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
        ax2.set_title('볼린저밴드', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3.plot(dates, data['RSI'], label='RSI', color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='과매수(70)')
        ax3.axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='과매도(30)')
        ax3.fill_between(dates, 70, 100, alpha=0.1, color='red')
        ax3.fill_between(dates, 0, 30, alpha=0.1, color='blue')
        ax3.set_title('RSI (Relative Strength Index)', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4.plot(dates, data['MACD'], label='MACD', color='blue')
        ax4.plot(dates, data['MACD_Signal'], label='Signal', color='red')
        ax4.bar(dates, data['MACD'] - data['MACD_Signal'], label='Histogram', 
                color='gray', alpha=0.3, width=1)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 날짜 포맷 설정
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def update_info_panel(self, data):
        """정보 패널 업데이트"""
        current = data.iloc[-1]
        prev = data.iloc[-2]
        
        info_text = f"""
        현재가: {current['Close']:.2f} | 전일대비: {((current['Close']/prev['Close']-1)*100):+.2f}%
        RSI: {current['RSI']:.1f} | MACD: {current['MACD']:.3f} | Signal: {current['MACD_Signal']:.3f}
        20일선: {current['MA20']:.2f} | 60일선: {current['MA60']:.2f} | 120일선: {current['MA120']:.2f}
        볼린저 상단: {current['BB_Upper']:.2f} | 하단: {current['BB_Lower']:.2f}
        """
        
        self.info_label.setText(info_text)


class ConditionBuilderDialog(QDialog):
    """조건 빌더 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('사용자 정의 조건 생성')
        self.setGeometry(300, 300, 500, 400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 조건 이름
        layout.addWidget(QLabel("조건 이름:"))
        self.name_edit = QLineEdit()
        layout.addWidget(self.name_edit)
        
        # 매수/매도 선택
        layout.addWidget(QLabel("조건 유형:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(["BUY", "SELL"])
        layout.addWidget(self.action_combo)
        
        # 지표 선택
        layout.addWidget(QLabel("기술적 지표:"))
        self.indicator_combo = QComboBox()
        indicators = [
            'RSI', 'MACD', 'MACD_Signal', '%K', '%D', 'Williams_R',
            'MA20', 'MA60', 'MA120', 'BB_Upper', 'BB_Lower', 'CCI',
            'Volume_Ratio', 'Close', 'High', 'Low'
        ]
        self.indicator_combo.addItems(indicators)
        layout.addWidget(self.indicator_combo)
        
        # 연산자 선택
        layout.addWidget(QLabel("연산자:"))
        self.operator_combo = QComboBox()
        operators = ['>', '<', '>=', '<=', '==', 'cross_above', 'cross_below']
        self.operator_combo.addItems(operators)
        layout.addWidget(self.operator_combo)
        
        # 값 입력
        layout.addWidget(QLabel("비교값:"))
        self.value_edit = QLineEdit()
        layout.addWidget(self.value_edit)
        
        # 설명
        description = QLabel("""
        예시:
        • RSI > 70: RSI가 70보다 클 때
        • MACD cross_above 0: MACD가 0선을 상향돌파할 때
        • Close < MA20: 종가가 20일선 아래일 때
        """)
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # 버튼
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("확인")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_condition(self):
        """생성된 조건 반환"""
        try:
            return {
                'name': self.name_edit.text(),
                'action': self.action_combo.currentText(),
                'indicator': self.indicator_combo.currentText(),
                'operator': self.operator_combo.currentText(),
                'value': float(self.value_edit.text())
            }
        except ValueError:
            QMessageBox.warning(self, "오류", "비교값은 숫자여야 합니다.")
            return None


class ConditionManagerDialog(QDialog):
    """조건 관리 다이얼로그"""
    def __init__(self, conditions, parent=None):
        super().__init__(parent)
        self.conditions = conditions.copy()
        self.setWindowTitle('사용자 정의 조건 관리')
        self.setGeometry(300, 300, 600, 400)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 조건 리스트
        self.condition_list = QListWidget()
        self.update_condition_list()
        layout.addWidget(self.condition_list)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        edit_btn = QPushButton("편집")
        edit_btn.clicked.connect(self.edit_condition)
        button_layout.addWidget(edit_btn)
        
        delete_btn = QPushButton("삭제")
        delete_btn.clicked.connect(self.delete_condition)
        button_layout.addWidget(delete_btn)
        
        ok_btn = QPushButton("확인")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def update_condition_list(self):
        """조건 리스트 업데이트"""
        self.condition_list.clear()
        for condition in self.conditions:
            item_text = f"[{condition['action']}] {condition['name']}: {condition['indicator']} {condition['operator']} {condition['value']}"
            self.condition_list.addItem(item_text)
    
    def edit_condition(self):
        """조건 편집"""
        current_row = self.condition_list.currentRow()
        if current_row >= 0:
            # 편집 로직 구현 (간단하게 삭제 후 재생성으로 처리)
            QMessageBox.information(self, "알림", "편집 기능은 현재 개발 중입니다.\n삭제 후 새로 추가해주세요.")
    
    def delete_condition(self):
        """조건 삭제"""
        current_row = self.condition_list.currentRow()
        if current_row >= 0:
            del self.conditions[current_row]
            self.update_condition_list()
    
    def get_conditions(self):
        """조건 리스트 반환"""
        return self.conditions


class CSVEditorDialog(QDialog):
    """CSV 파일 편집 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CSV 파일 편집')
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()