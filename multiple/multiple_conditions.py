import sys
import pandas as pd
import yfinance as yf
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

class StockScreener(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.create_sample_csv_files()
        self.load_stock_lists()
        
    def initUI(self):
        self.setWindowTitle('Global Stock Screener - CSV 기반 종목 관리')
        self.setGeometry(100, 100, 1500, 900)
        
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
        
        # 시장 선택
        layout.addWidget(QLabel("시장 선택:"), 0, 0)
        self.market_combo = QComboBox()
        self.market_combo.addItems(["전체", "한국 (KOSPI/KOSDAQ)", "미국 (NASDAQ/NYSE)", "스웨덴 (OMX)"])
        self.market_combo.currentTextChanged.connect(self.update_stock_count)
        layout.addWidget(self.market_combo, 0, 1)
        
        # CSV 파일 관리 버튼들
        csv_layout = QHBoxLayout()
        self.refresh_csv_btn = QPushButton("CSV 파일 새로고침")
        self.refresh_csv_btn.clicked.connect(self.load_stock_lists)
        csv_layout.addWidget(self.refresh_csv_btn)
        
        self.edit_csv_btn = QPushButton("CSV 파일 편집")
        self.edit_csv_btn.clicked.connect(self.open_csv_editor)
        csv_layout.addWidget(self.edit_csv_btn)
        
        self.sample_csv_btn = QPushButton("샘플 CSV 생성")
        self.sample_csv_btn.clicked.connect(self.create_sample_csv_files)
        csv_layout.addWidget(self.sample_csv_btn)
        
        layout.addLayout(csv_layout, 0, 2, 1, 2)
        
        # 매수 조건 설정
        buy_group = QGroupBox("매수 조건")
        buy_layout = QVBoxLayout()
        
        # 이평선 조건
        self.ma_condition = QCheckBox("60일선이 120일선 돌파 + 우상향 + 이평선 터치")
        buy_layout.addWidget(self.ma_condition)
        
        # 볼린저밴드 조건
        self.bb_condition = QCheckBox("볼린저밴드 하단 터치 + RSI < 35")
        buy_layout.addWidget(self.bb_condition)
        
        # 지지선 조건
        self.support_condition = QCheckBox("MACD 골든크로스 + 거래량 증가")
        buy_layout.addWidget(self.support_condition)
        
        # 상대강도 조건
        self.momentum_condition = QCheckBox("20일 상대강도 상승 + 펀더멘털 양호")
        buy_layout.addWidget(self.momentum_condition)
        
        buy_group.setLayout(buy_layout)
        layout.addWidget(buy_group, 1, 0, 1, 2)
        
        # 매도 조건 설정
        sell_group = QGroupBox("매도 조건")
        sell_layout = QVBoxLayout()
        
        # 기술적 매도
        self.tech_sell = QCheckBox("데드크로스 + 60일선 3% 하향이탈")
        sell_layout.addWidget(self.tech_sell)
        
        # 수익률 매도
        self.profit_sell = QCheckBox("20% 수익달성 또는 -7% 손절")
        sell_layout.addWidget(self.profit_sell)
        
        # 볼린저 상단 매도
        self.bb_sell = QCheckBox("볼린저 상단 + RSI > 70")
        sell_layout.addWidget(self.bb_sell)
        
        # 거래량 급감 매도
        self.volume_sell = QCheckBox("거래량 급감 + 모멘텀 약화")
        sell_layout.addWidget(self.volume_sell)
        
        sell_group.setLayout(sell_layout)
        layout.addWidget(sell_group, 1, 2, 1, 2)
        
        # 검색 버튼
        self.search_btn = QPushButton("종목 스크리닝 시작")
        self.search_btn.clicked.connect(self.run_screening)
        self.search_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 12px; font-size: 14px; }")
        layout.addWidget(self.search_btn, 2, 0, 1, 4)
        
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
        self.buy_table.doubleClicked.connect(self.show_stock_detail)
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
        self.sell_table.doubleClicked.connect(self.show_stock_detail)
        sell_layout.addWidget(self.sell_table)
        sell_group.setLayout(sell_layout)
        
        splitter.addWidget(buy_group)
        splitter.addWidget(sell_group)
        
        return splitter
    
    def create_sample_csv_files(self):
        """샘플 CSV 파일들 생성"""
        if not os.path.exists('stock_data'):
            os.makedirs('stock_data')
        
        # 한국 주식 샘플
        korea_stocks = {
            'ticker': [
                '005930.KS', '000660.KS', '035420.KS', '207940.KS', '006400.KS',
                '035720.KS', '051910.KS', '096770.KS', '068270.KS', '015760.KS',
                '003550.KS', '017670.KS', '030200.KS', '036570.KS', '012330.KS',
                '028260.KS', '066570.KS', '323410.KS', '000270.KS', '005380.KS'
            ],
            'name': [
                '삼성전자', 'SK하이닉스', '네이버', '삼성바이오로직스', '삼성SDI',
                '카카오', 'LG화학', 'SK이노베이션', '셀트리온', '한국전력',
                'LG', 'SK텔레콤', 'KT&G', '엔씨소프트', '현대모비스',
                '삼성물산', 'LG전자', '카카오뱅크', '기아', '현대차'
            ],
            'sector': [
                '반도체', '반도체', 'IT서비스', '바이오', '배터리',
                'IT서비스', '화학', '에너지', '바이오', '전력',
                '지주회사', '통신', '담배', '게임', '자동차부품',
                '건설', '전자', '금융', '자동차', '자동차'
            ],
            'market_cap': [
                500000, 80000, 40000, 35000, 30000,
                25000, 22000, 20000, 18000, 15000,
                14000, 13000, 12000, 11000, 10000,
                9000, 8500, 8000, 7500, 7000
            ]
        }
        
        # 미국 주식 샘플
        usa_stocks = {
            'ticker': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'NFLX', 'ADBE', 'INTC',
                'AMD', 'CRM', 'PYPL', 'UBER', 'SHOP',
                'ZOOM', 'DOCU', 'SNOW', 'PLTR', 'RBLX'
            ],
            'name': [
                'Apple Inc', 'Microsoft Corp', 'Alphabet Inc', 'Amazon.com Inc', 'Tesla Inc',
                'NVIDIA Corp', 'Meta Platforms', 'Netflix Inc', 'Adobe Inc', 'Intel Corp',
                'Advanced Micro Devices', 'Salesforce Inc', 'PayPal Holdings', 'Uber Technologies', 'Shopify Inc',
                'Zoom Video Communications', 'DocuSign Inc', 'Snowflake Inc', 'Palantir Technologies', 'Roblox Corp'
            ],
            'sector': [
                'Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary',
                'Technology', 'Technology', 'Communication Services', 'Technology', 'Technology',
                'Technology', 'Technology', 'Financial Services', 'Technology', 'Technology',
                'Technology', 'Technology', 'Technology', 'Technology', 'Technology'
            ],
            'market_cap': [
                3000000, 2800000, 1700000, 1500000, 800000,
                1900000, 800000, 200000, 250000, 200000,
                250000, 220000, 80000, 120000, 80000,
                25000, 15000, 60000, 40000, 25000
            ]
        }
        
        # 스웨덴 주식 샘플
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
        pd.DataFrame(korea_stocks).to_csv('stock_data/korea_stocks.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(usa_stocks).to_csv('stock_data/usa_stocks.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(sweden_stocks).to_csv('stock_data/sweden_stocks.csv', index=False, encoding='utf-8-sig')
        
        QMessageBox.information(self, "완료", "샘플 CSV 파일이 생성되었습니다!\n'stock_data' 폴더를 확인해주세요.")
    
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
            data['MA20'] = data['Close'].rolling(20).mean()
            data['MA60'] = data['Close'].rolling(60).mean()
            data['MA120'] = data['Close'].rolling(120).mean()
            
            # RSI 계산
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # 볼린저밴드 계산
            data['BB_Middle'] = data['Close'].rolling(20).mean()
            bb_std = data['Close'].rolling(20).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # MACD 계산
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            data['MACD'] = ema12 - ema26
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            
            # 거래량 비율
            data['Volume_Ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            
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
    
    def show_stock_detail(self, index):
        """종목 상세 정보 표시"""
        table = self.sender()
        row = index.row()
        symbol = table.item(row, 0).text()
        name = table.item(row, 1).text()
        
        QMessageBox.information(self, "종목 정보", 
                               f"종목: {symbol} ({name})\n"
                               f"Yahoo Finance에서 더 자세한 정보를 확인하세요.\n"
                               f"URL: https://finance.yahoo.com/quote/{symbol}")


class CSVEditorDialog(QDialog):
    """CSV 파일 편집 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('CSV 파일 편집')
        self.setGeometry(200, 200, 800, 600)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 파일 선택
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("편집할 파일:"))
        
        self.file_combo = QComboBox()
        self.file_combo.addItems(["korea_stocks.csv", "usa_stocks.csv", "sweden_stocks.csv"])
        self.file_combo.currentTextChanged.connect(self.load_csv_file)
        file_layout.addWidget(self.file_combo)
        
        self.load_btn = QPushButton("파일 로드")
        self.load_btn.clicked.connect(self.load_csv_file)
        file_layout.addWidget(self.load_btn)
        
        layout.addLayout(file_layout)
        
        # 테이블
        self.table = QTableWidget()
        layout.addWidget(self.table)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.add_row_btn = QPushButton("행 추가")
        self.add_row_btn.clicked.connect(self.add_row)
        button_layout.addWidget(self.add_row_btn)
        
        self.delete_row_btn = QPushButton("행 삭제")
        self.delete_row_btn.clicked.connect(self.delete_row)
        button_layout.addWidget(self.delete_row_btn)
        
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self.save_csv_file)
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("취소")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # 초기 파일 로드
        self.load_csv_file()
    
    def load_csv_file(self):
        """선택된 CSV 파일 로드"""
        filename = self.file_combo.currentText()
        filepath = f'stock_data/{filename}'
        
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                
                self.table.setRowCount(len(df))
                self.table.setColumnCount(len(df.columns))
                self.table.setHorizontalHeaderLabels(df.columns.tolist())
                
                for i in range(len(df)):
                    for j in range(len(df.columns)):
                        item = QTableWidgetItem(str(df.iloc[i, j]))
                        self.table.setItem(i, j, item)
                
                self.table.resizeColumnsToContents()
                
            except Exception as e:
                QMessageBox.warning(self, "오류", f"파일 로드 실패: {str(e)}")
        else:
            QMessageBox.warning(self, "알림", f"파일이 존재하지 않습니다: {filepath}")
    
    def add_row(self):
        """새로운 행 추가"""
        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        
        # 기본값 설정
        if self.file_combo.currentText() == "korea_stocks.csv":
            self.table.setItem(row_count, 0, QTableWidgetItem("000000.KS"))
            self.table.setItem(row_count, 1, QTableWidgetItem("새종목"))
            self.table.setItem(row_count, 2, QTableWidgetItem("기타"))
            self.table.setItem(row_count, 3, QTableWidgetItem("1000"))
        elif self.file_combo.currentText() == "usa_stocks.csv":
            self.table.setItem(row_count, 0, QTableWidgetItem("NEWSTK"))
            self.table.setItem(row_count, 1, QTableWidgetItem("New Stock"))
            self.table.setItem(row_count, 2, QTableWidgetItem("Technology"))
            self.table.setItem(row_count, 3, QTableWidgetItem("1000"))
        else:  # sweden
            self.table.setItem(row_count, 0, QTableWidgetItem("NEW.ST"))
            self.table.setItem(row_count, 1, QTableWidgetItem("New Stock AB"))
            self.table.setItem(row_count, 2, QTableWidgetItem("Industrials"))
            self.table.setItem(row_count, 3, QTableWidgetItem("1000"))
    
    def delete_row(self):
        """선택된 행 삭제"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.table.removeRow(current_row)
    
    def save_csv_file(self):
        """CSV 파일 저장"""
        filename = self.file_combo.currentText()
        filepath = f'stock_data/{filename}'
        
        try:
            # 테이블 데이터를 DataFrame으로 변환
            data = []
            headers = []
            
            for j in range(self.table.columnCount()):
                headers.append(self.table.horizontalHeaderItem(j).text())
            
            for i in range(self.table.rowCount()):
                row_data = []
                for j in range(self.table.columnCount()):
                    item = self.table.item(i, j)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                data.append(row_data)
            
            df = pd.DataFrame(data, columns=headers)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            QMessageBox.information(self, "완료", f"{filename} 파일이 저장되었습니다.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 저장 실패: {str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin: 5px;
            padding-top: 15px;
            background-color: white;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #333333;
        }
        QTableWidget {
            gridline-color: #e0e0e0;
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 5px;
        }
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #f0f0f0;
        }
        QTableWidget::item:selected {
            background-color: #3daee9;
            color: white;
        }
        QTableWidget::horizontalHeader {
            background-color: #f8f9fa;
            border: none;
            font-weight: bold;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 8px 15px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
            border-color: #999999;
        }
        QPushButton:pressed {
            background-color: #e0e0e0;
        }
        QCheckBox {
            font-size: 12px;
            spacing: 8px;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #cccccc;
            border-radius: 4px;
            background-color: white;
        }
    """)
    
    screener = StockScreener()
    screener.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()