"""
chart_window.py
종목 차트 표시 윈도우 - 완전한 버전
"""

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
import platform

from utils import TechnicalAnalysis
import unicodedata

# 최적화 모듈 import
from cache_manager import get_stock_data
from matplotlib_optimizer import ChartManager

def has_hangul(s):
    for ch in s:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        if "HANGUL" in name:
            return True
    return False

# 한글 폰트 설정
def setup_korean_font():
    """한글 폰트 설정"""
    try:
        import matplotlib.font_manager as fm
        
        # 운영체제별 한글 폰트 설정
        system = platform.system()
        if system == "Windows":
            # Windows 한글 폰트
            fonts = ['Malgun Gothic', 'Arial Unicode MS', 'MS Gothic']
        elif system == "Darwin":  # macOS
            # macOS 한글 폰트
            fonts = ['AppleGothic', 'Arial Unicode MS']
        else:  # Linux
            # Linux 한글 폰트
            fonts = ['DejaVu Sans', 'Liberation Sans']
        
        # 사용 가능한 폰트 찾기
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        korean_font = None
        
        for font in fonts:
            if font in available_fonts:
                korean_font = font
                break
        
        if korean_font:
            plt.rcParams['font.family'] = korean_font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 한글 폰트 설정: {korean_font}")
        else:
            # 한글 폰트가 없으면 기본 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            print("⚠️ 한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
            
    except Exception as e:
        print(f"⚠️ 폰트 설정 중 오류: {e}")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

# 초기 폰트 설정
setup_korean_font()

class StockChartWindow(QMainWindow):
    """종목 차트 윈도우 - 개선된 버전 (메모리 최적화)"""
    def __init__(self, symbol, name, parent=None):
        super().__init__(parent)
        self.symbol = symbol
        self.name = name
        self.technical_analyzer = TechnicalAnalysis()

        # 차트 메모리 관리자
        self.chart_manager = ChartManager()

        # 한글 이름을 영문으로 변경 (폰트 문제 해결)
        display_name = name if not has_hangul(name) else symbol

        self.setWindowTitle(f'📊 {symbol} ({display_name}) - Technical Analysis Chart')

        # 윈도우 크기를 더 크게 설정
        self.setGeometry(100, 100, 1600, 1000)  # 1200x800 → 1600x1000

        self.setup_ui()
        self.load_chart_data()
    
    def setup_ui(self):
        """UI 설정 - 정보 패널 높이 증가"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 상단 컨트롤 패널 (높이 고정)
        control_panel = self.create_control_panel()
        control_panel.setMaximumHeight(80)
        layout.addWidget(control_panel)
        
        # 차트 영역 (확장 가능)
        self.figure = Figure(figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=3)  # 차트가 더 많은 공간 차지
        
        # 하단 정보 패널 (높이 증가 + 스크롤)
        info_panel = self.create_info_panel()
        info_panel.setMaximumHeight(200)  # 150 → 200으로 증가
        layout.addWidget(info_panel, stretch=1)   # 정보 패널도 약간의 확장성

    def create_control_panel(self):
        """컨트롤 패널 생성 - 차트 레이아웃 옵션 추가"""
        group = QGroupBox("Chart Settings")
        layout = QHBoxLayout()
        
        # 기간 선택
        layout.addWidget(QLabel("Period:"))
        self.period_combo = QComboBox()
        self.period_combo.addItems(["3 Months", "6 Months", "1 Year", "2 Years"])
        self.period_combo.setCurrentText("6 Months")
        self.period_combo.currentTextChanged.connect(self.load_chart_data)
        layout.addWidget(self.period_combo)
        
        # 차트 레이아웃 선택 추가
        layout.addWidget(QLabel("Layout:"))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(["Standard (5 Charts)", "Compact (3 Charts)", "Price Focus (2 Charts)"])
        self.layout_combo.setCurrentText("Standard (5 Charts)")
        self.layout_combo.currentTextChanged.connect(self.load_chart_data)
        layout.addWidget(self.layout_combo)
        
        # 새로고침 버튼
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_chart_data)
        layout.addWidget(refresh_btn)
        
        # 전체화면 버튼 추가
        fullscreen_btn = QPushButton("🖥️ Fullscreen")
        fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(fullscreen_btn)
        
        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_info_panel(self):
        """정보 패널 생성 - 스크롤 가능한 버전"""
        group = QGroupBox("📊 Technical Indicators Info")
        layout = QVBoxLayout()
        
        # 스크롤 영역 생성
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 내용에 맞춰 크기 조정
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)    # 필요시 세로 스크롤바
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 필요시 가로 스크롤바
        
        # 스크롤 가능한 위젯 생성
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # 정보 표시용 라벨
        self.info_label = QLabel("Loading chart data...")
        self.info_label.setWordWrap(True)           # 자동 줄바꿈
        self.info_label.setAlignment(Qt.AlignTop)   # 상단 정렬
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)  # 마우스로 텍스트 선택 가능
        
        # 폰트 설정 (더 읽기 쉽게)
        font = self.info_label.font()
        font.setFamily("Consolas")  # 고정폭 폰트 (숫자 정렬이 깔끔)
        font.setPointSize(10)       # 적당한 크기
        self.info_label.setFont(font)
        
        # 배경색과 패딩 설정
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 10px;
                color: #212529;
            }
        """)
        
        # 스크롤 위젯에 라벨 추가
        scroll_layout.addWidget(self.info_label)
        scroll_layout.addStretch()  # 남은 공간 채우기
        
        # 스크롤 영역에 위젯 설정
        scroll_area.setWidget(scroll_widget)
        
        # 그룹박스에 스크롤 영역 추가
        layout.addWidget(scroll_area)
        group.setLayout(layout)
        
        return group

    def toggle_fullscreen(self):
        """전체화면 토글"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def get_chart_layout(self):
        """선택된 차트 레이아웃 반환"""
        layout_text = self.layout_combo.currentText()
        if "Compact" in layout_text:
            return "compact"  # 가격+RSI+거래량
        elif "Price Focus" in layout_text:
            return "price_focus"  # 가격+거래량만
        else:
            return "standard"  # 전체 5개

    def get_period_days(self):
        """선택된 기간에 따른 일수 반환 (120일선 계산을 위해 충분한 데이터 확보)"""
        period_map = {
            "3 Months": 90 + 120,   # 표시기간 + 120일선 계산용
            "6 Months": 180 + 120,  # 표시기간 + 120일선 계산용  
            "1 Year": 365 + 120,    # 표시기간 + 120일선 계산용
            "2 Years": 730 + 120    # 표시기간 + 120일선 계산용
        }
        return period_map.get(self.period_combo.currentText(), 300)

    def get_display_days(self):
        """실제 차트에 표시할 기간"""
        period_map = {
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730
        }
        return period_map.get(self.period_combo.currentText(), 180)

    def load_chart_data(self):
        """차트 데이터 로드 - 오류 처리 강화"""
        try:
            self.info_label.setText("Loading data...")
            QApplication.processEvents()
            
            # 충분한 데이터 로드
            total_days = self.get_period_days()
            display_days = self.get_display_days()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=total_days)
            
            # 🔧 여러 방법으로 데이터 시도
            data = self.fetch_stock_data_with_retry(self.symbol, start_date, end_date)
            
            if data is None or data.empty:
                error_msg = f"❌ '{self.symbol}' 데이터를 불러올 수 없습니다.\n"
                error_msg += "가능한 원인:\n"
                error_msg += "• 상장폐지되었거나 거래가 중단된 종목\n"
                error_msg += "• 잘못된 종목 코드\n"
                error_msg += "• 일시적인 서버 문제"
                self.info_label.setText(error_msg)
                return
            
            # 시간대 정보 처리
            if data.index.tz is not None:
                data.index = data.index.tz_convert('UTC').tz_localize(None)
            
            # 기술적 지표 계산
            data = self.technical_analyzer.calculate_all_indicators(data)
            
            # 표시할 기간 필터링
            display_start_date = end_date - timedelta(days=display_days)
            import pandas as pd
            display_start_timestamp = pd.Timestamp(display_start_date)
            display_data = data[data.index >= display_start_timestamp]
            
            if display_data.empty:
                display_rows = min(display_days, len(data))
                display_data = data.tail(display_rows)
                print(f"⚠️ 날짜 필터링 실패, 최근 {display_rows}개 데이터 사용")
            
            self.plot_chart(display_data)
            self.update_info_panel(display_data)

        except Exception as e:
            error_msg = f"❌ 차트 로딩 오류: {str(e)}\n"
            error_msg += f"종목: {self.symbol}\n"
            error_msg += "다른 종목을 시도해보세요."
            self.info_label.setText(error_msg)
            print(f"Chart loading error for {self.symbol}: {e}")

    def fetch_stock_data_with_retry(self, symbol, start_date, end_date):
        """여러 방법으로 주식 데이터 시도 (캐싱 사용)"""

        # 1차 시도: 원래 심볼 그대로 (캐싱 사용)
        try:
            print(f"📊 데이터 로딩 시도 1: {symbol}")
            # 기간 계산
            days_diff = (end_date - start_date).days + 10
            period_str = f"{days_diff}d"

            data = get_stock_data(symbol, period=period_str)

            if data is not None and not data.empty:
                print(f"✅ 성공: {symbol} - {len(data)}개 데이터")
                return data
        except Exception as e:
            print(f"❌ 1차 시도 실패: {e}")

        # 2차 시도: 심볼 변형 (한국 주식의 경우)
        if '.KQ' in symbol:
            try:
                alt_symbol = symbol.replace('.KQ', '.KS')
                print(f"📊 데이터 로딩 시도 2: {alt_symbol} (.KQ → .KS)")
                days_diff = (end_date - start_date).days + 10
                period_str = f"{days_diff}d"

                data = get_stock_data(alt_symbol, period=period_str)

                if data is not None and not data.empty:
                    print(f"✅ 성공: {alt_symbol} - {len(data)}개 데이터")
                    return data
            except Exception as e:
                print(f"❌ 2차 시도 실패: {e}")
        
        elif '.KS' in symbol:
            try:
                alt_symbol = symbol.replace('.KS', '.KQ')
                print(f"📊 데이터 로딩 시도 2: {alt_symbol} (.KS → .KQ)")
                days_diff = (end_date - start_date).days + 10
                period_str = f"{days_diff}d"

                data = get_stock_data(alt_symbol, period=period_str)

                if data is not None and not data.empty:
                    print(f"✅ 성공: {alt_symbol} - {len(data)}개 데이터")
                    return data
            except Exception as e:
                print(f"❌ 2차 시도 실패: {e}")

        # 3차 시도: 더 긴 기간으로 시도
        try:
            print(f"📊 데이터 로딩 시도 3: {symbol} (기간 확장)")
            data = get_stock_data(symbol, period="1y")

            if data is not None and not data.empty:
                print(f"✅ 성공 (확장): {symbol} - {len(data)}개 데이터")
                return data
        except Exception as e:
            print(f"❌ 3차 시도 실패: {e}")

        # 4차 시도: 단기 데이터
        try:
            print(f"📊 데이터 로딩 시도 4: {symbol} (단기)")
            data = get_stock_data(symbol, period="1mo")

            if data is not None and not data.empty:
                print(f"✅ 성공 (단기): {symbol} - {len(data)}개 데이터")
                return data
        except Exception as e:
            print(f"❌ 4차 시도 실패: {e}")
        
        print(f"❌ 모든 시도 실패: {symbol}")
        return None

    def plot_chart(self, data):
        """차트 그리기 - 레이아웃별 최적화"""
        self.figure.clear()
        
        layout_type = self.get_chart_layout()
        
        if layout_type == "price_focus":
            self.plot_price_focus_layout(data)
        elif layout_type == "compact":
            self.plot_compact_layout(data)
        else:
            self.plot_standard_layout(data)
        
        # 레이아웃 조정
        self.figure.tight_layout(pad=2.0)  # 여백 증가
        self.canvas.draw()

    def plot_price_focus_layout(self, data):
        """가격 중심 레이아웃 (2개 차트)"""
        # 큰 가격 차트 + 작은 거래량 차트
        ax1 = self.figure.add_subplot(4, 1, (1, 3))  # 위 3/4 차지
        ax2 = self.figure.add_subplot(4, 1, 4)       # 아래 1/4 차지
        
        dates = data.index
        
        # 1. 메인 가격 차트 (캔들 + 이동평균 + 볼린저밴드)
        self._plot_candles(ax1, data, bar_width_factor=0.8)  # 캔들 두껍게
        
        # 이동평균선
        ax1.plot(dates, data['MA20'], label='MA20', color='green', alpha=0.8, linewidth=2)
        ax1.plot(dates, data['MA60'], label='MA60', color='blue', alpha=0.8, linewidth=2)
        
        # 120일선 (유효한 데이터만)
        ma120_valid = data['MA120'].notna()
        if ma120_valid.sum() > len(data) * 0.5:
            valid_dates = dates[ma120_valid]
            valid_ma120 = data.loc[ma120_valid, 'MA120']
            ax1.plot(valid_dates, valid_ma120, label='MA120', color='red', alpha=0.8, linewidth=2.5)
        
        # 볼린저밴드 (반투명)
        ax1.plot(dates, data['BB_Upper'], color='purple', alpha=0.4, linewidth=1)
        ax1.plot(dates, data['BB_Lower'], color='purple', alpha=0.4, linewidth=1)
        ax1.fill_between(dates, data['BB_Upper'], data['BB_Lower'], alpha=0.05, color='purple')
        
        ax1.set_title(f'{self.symbol} ({self.name}) - Price Chart with Technical Indicators', 
                     fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 현재가 표시
        current_price = data['Close'].iloc[-1]
        ax1.axhline(y=current_price, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax1.text(dates[-1], current_price, f'  {current_price:.2f}', 
                verticalalignment='center', color='red', fontweight='bold', fontsize=12)
        
        # 2. 거래량 차트
        colors = ['red' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'blue' 
                 for i in range(1, len(data))]
        colors.insert(0, 'gray')
        
        ax2.bar(dates, data['Volume'], color=colors, alpha=0.7, width=1)
        ax2.plot(dates, data['Volume'].rolling(20).mean(), 
                label='20-day Avg Volume', color='orange', linewidth=2)
        ax2.set_title('Volume', fontsize=14)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        self._format_dates([ax1, ax2], data)

    def plot_compact_layout(self, data):
        """간소화 레이아웃 (3개 차트)"""
        ax1 = self.figure.add_subplot(3, 1, 1)  # 가격
        ax2 = self.figure.add_subplot(3, 1, 2)  # RSI
        ax3 = self.figure.add_subplot(3, 1, 3)  # 거래량
        
        dates = data.index
        
        # 1. 가격 차트
        self._plot_candles(ax1, data)
        ax1.plot(dates, data['MA20'], label='MA20', color='green', alpha=0.8, linewidth=1.5)
        ax1.plot(dates, data['MA60'], label='MA60', color='blue', alpha=0.8, linewidth=1.5)
        
        ma120_valid = data['MA120'].notna()
        if ma120_valid.sum() > len(data) * 0.5:
            valid_dates = dates[ma120_valid]
            valid_ma120 = data.loc[ma120_valid, 'MA120']
            ax1.plot(valid_dates, valid_ma120, label='MA120', color='red', alpha=0.8, linewidth=2)
        
        ax1.set_title(f'{self.symbol} - Price & Moving Averages', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 현재가 표시
        current_price = data['Close'].iloc[-1]
        ax1.axhline(y=current_price, color='red', linestyle='--', alpha=0.7)
        ax1.text(dates[-1], current_price, f'  {current_price:.2f}', 
                verticalalignment='center', color='red', fontweight='bold')
        
        # 2. RSI
        ax2.plot(dates, data['RSI'], label='RSI', color='purple', linewidth=2)
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='blue', linestyle='--', alpha=0.7)
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax2.fill_between(dates, 70, 100, alpha=0.1, color='red')
        ax2.fill_between(dates, 0, 30, alpha=0.1, color='blue')
        ax2.set_title('RSI', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 현재 RSI 값 표시
        current_rsi = data['RSI'].iloc[-1]
        ax2.text(dates[-1], current_rsi, f'  {current_rsi:.1f}', 
                verticalalignment='center', color='purple', fontweight='bold')
        
        # 3. 거래량
        colors = ['red' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'blue' 
                 for i in range(1, len(data))]
        colors.insert(0, 'gray')
        
        ax3.bar(dates, data['Volume'], color=colors, alpha=0.6, width=1)
        ax3.plot(dates, data['Volume'].rolling(20).mean(), 
                label='20-day Avg', color='orange', linewidth=2)
        ax3.set_title('Volume', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        self._format_dates([ax1, ax2, ax3], data)

    def plot_standard_layout(self, data):
        """표준 레이아웃 (5개 차트)"""
        ax1 = self.figure.add_subplot(5, 1, 1)  # 가격 차트
        ax2 = self.figure.add_subplot(5, 1, 2)  # 볼린저밴드
        ax3 = self.figure.add_subplot(5, 1, 3)  # RSI
        ax4 = self.figure.add_subplot(5, 1, 4)  # MACD
        ax5 = self.figure.add_subplot(5, 1, 5)  # 거래량
        
        dates = data.index
        
        # 1. 가격 차트 + 이동평균선
        self._plot_candles(ax1, data)
        ax1.plot(dates, data['MA20'], label='MA20', color='green', alpha=0.7, linewidth=1.5)
        ax1.plot(dates, data['MA60'], label='MA60', color='blue', alpha=0.7, linewidth=1.5)
        
        # 120일선 (유효한 데이터만)
        ma120_valid = data['MA120'].notna()
        if ma120_valid.sum() > len(data) * 0.5:
            valid_dates = dates[ma120_valid]
            valid_ma120 = data.loc[ma120_valid, 'MA120']
            ax1.plot(valid_dates, valid_ma120, label='MA120', color='red', alpha=0.7, linewidth=2)
            
            # 120일선 불완전 구간 표시
            if ma120_valid.sum() < len(data):
                ax1.plot(dates[~ma120_valid], data.loc[~ma120_valid, 'MA120'], 
                        label='MA120 (불완전)', color='red', alpha=0.3, linestyle='--', linewidth=1)
        else:
            # 120일선 데이터가 너무 적으면 경고 표시
            ax1.text(0.02, 0.98, '⚠️ 120일선 데이터 부족', transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax1.set_title(f'{self.symbol} ({self.name}) - Price Chart', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 현재가 표시
        current_price = data['Close'].iloc[-1]
        ax1.axhline(y=current_price, color='red', linestyle='--', alpha=0.7)
        ax1.text(dates[-1], current_price, f'  {current_price:.2f}', 
                verticalalignment='center', color='red', fontweight='bold')
        
        # 2. 볼린저밴드
        self._plot_candles(ax2, data)
        ax2.plot(dates, data['BB_Upper'], label='BB Upper', color='red', alpha=0.7)
        ax2.plot(dates, data['BB_Middle'], label='BB Middle(MA20)', color='green', alpha=0.7)
        ax2.plot(dates, data['BB_Lower'], label='BB Lower', color='red', alpha=0.7)
        ax2.fill_between(dates, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
        ax2.set_title('Bollinger Bands', fontsize=10)
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3.plot(dates, data['RSI'], label='RSI', color='purple', linewidth=2)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought(70)')
        ax3.axhline(y=30, color='blue', linestyle='--', alpha=0.7, label='Oversold(30)')
        ax3.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        ax3.fill_between(dates, 70, 100, alpha=0.1, color='red')
        ax3.fill_between(dates, 0, 30, alpha=0.1, color='blue')
        ax3.set_title('RSI (Relative Strength Index)', fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 현재 RSI 값 표시
        current_rsi = data['RSI'].iloc[-1]
        ax3.text(dates[-1], current_rsi, f'  {current_rsi:.1f}', 
                verticalalignment='center', color='purple', fontweight='bold')
        
        # 4. MACD
        ax4.plot(dates, data['MACD'], label='MACD', color='blue', linewidth=2)
        ax4.plot(dates, data['MACD_Signal'], label='Signal', color='red', linewidth=2)
        ax4.bar(dates, data['MACD_Histogram'], label='Histogram', 
                color='gray', alpha=0.3, width=1)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD', fontsize=10)
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. 거래량
        colors = ['red' if data['Close'].iloc[i] >= data['Close'].iloc[i-1] else 'blue' 
                 for i in range(1, len(data))]
        colors.insert(0, 'gray')
        
        ax5.bar(dates, data['Volume'], color=colors, alpha=0.6, width=1)
        ax5.plot(dates, data['Volume'].rolling(20).mean(), 
                label='20-day Avg Volume', color='orange', linewidth=2)
        ax5.set_title('Volume', fontsize=10)
        ax5.legend(loc='upper left', fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        self._format_dates([ax1, ax2, ax3, ax4, ax5], data)

    def _plot_candles(self, ax, data, bar_width_factor=0.7):
        """캔들 그리기 - 막대 너비 조정 가능"""
        dates = data.index.to_pydatetime()
        o = data['Open'].to_numpy(dtype=float)
        h = data['High'].to_numpy(dtype=float)
        l = data['Low'].to_numpy(dtype=float)
        c = data['Close'].to_numpy(dtype=float)
        
        date_nums = mdates.date2num(dates)
        bar_width = (np.diff(date_nums).min() * bar_width_factor) if len(date_nums) > 1 else 0.6
        
        up = c >= o
        down = ~up
        
        dates_np = np.array(dates)
        
        # 꼬리
        ax.vlines(dates_np[up], l[up], h[up], color='red', linewidth=1, alpha=0.9)
        ax.vlines(dates_np[down], l[down], h[down], color='blue', linewidth=1, alpha=0.9)
        
        # 바디
        ax.bar(dates_np[up], (c - o)[up], bottom=o[up], width=bar_width,
               color='red', edgecolor='red', linewidth=0.5, align='center')
        ax.bar(dates_np[down], (c - o)[down], bottom=o[down], width=bar_width,
               color='blue', edgecolor='blue', linewidth=0.5, align='center')

    def _format_dates(self, axes, data):
        """날짜 포맷 설정 - 기간별 최적화"""
        data_length = len(data)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            # 데이터 길이에 따라 날짜 표시 간격 조정
            if data_length > 300:  # 1년 이상
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))
            elif data_length > 120:  # 6개월 이상
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            else:  # 3개월 이하
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        
        # 마지막 축만 x축 레이블 표시
        if axes:
            plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    def update_info_panel(self, data):
        """정보 패널 업데이트 - 더 상세한 정보"""
        if len(data) < 2:
            self.info_label.setText("데이터가 부족합니다(2개 이상의 봉 필요).")
            return

        current = data.iloc[-1]
        prev = data.iloc[-2]

        # 변화율 계산
        try:
            price_change = float(current['Close']) - float(prev['Close'])
            price_change_pct = (price_change / float(prev['Close'])) * 100 if prev['Close'] else 0.0
        except Exception:
            price_change, price_change_pct = 0.0, 0.0

        # 볼린저밴드 위치
        try:
            band_range = float(current['BB_Upper']) - float(current['BB_Lower'])
            bb_position = (float(current['Close']) - float(current['BB_Lower'])) / band_range if band_range != 0 else 0.5
        except Exception:
            bb_position = 0.5

        if bb_position > 0.8:
            bb_signal = "🔴 상단 근접 (매도 관심)"
        elif bb_position < 0.2:
            bb_signal = "🟢 하단 근접 (매수 관심)"
        else:
            bb_signal = "⚪ 중앙 영역 (관망)"

        # MACD 신호 분석
        macd_now = float(current.get('MACD', 0.0))
        macd_sig_now = float(current.get('MACD_Signal', 0.0))
        macd_prev = float(prev.get('MACD', 0.0))
        macd_sig_prev = float(prev.get('MACD_Signal', 0.0))

        macd_cross_up = (macd_now > macd_sig_now) and (macd_prev <= macd_sig_prev)
        macd_cross_down = (macd_now < macd_sig_now) and (macd_prev >= macd_sig_prev)
        
        if macd_cross_up:
            macd_desc = "🟢 골든크로스 발생 (강력한 매수 신호)"
        elif macd_cross_down:
            macd_desc = "🔴 데드크로스 발생 (강력한 매도 신호)"
        elif macd_now > macd_sig_now:
            macd_desc = "🟢 MACD > Signal (상승 모멘텀)"
        else:
            macd_desc = "🔴 MACD < Signal (하락 모멘텀)"

        # RSI 상세 분석
        rsi_now = float(current.get('RSI', 50.0))
        if rsi_now >= 80:
            rsi_desc = "🔴 극도 과매수 (즉시 매도 고려)"
        elif rsi_now >= 70:
            rsi_desc = "🟠 과매수 (매도 준비)"
        elif rsi_now >= 60:
            rsi_desc = "🟡 강세 구간 (상승 지속 가능)"
        elif rsi_now >= 40:
            rsi_desc = "⚪ 중립 구간 (방향성 애매)"
        elif rsi_now >= 30:
            rsi_desc = "🟡 약세 구간 (하락 지속 가능)"
        elif rsi_now >= 20:
            rsi_desc = "🟢 과매도 (매수 준비)"
        else:
            rsi_desc = "🔵 극도 과매도 (적극 매수 고려)"

        # 이동평균선 배열 상세 분석
        ma20 = float(current.get('MA20', float('nan')))
        ma60 = float(current.get('MA60', float('nan')))
        ma120 = float(current.get('MA120', float('nan')))

        if ma20 > ma60 > ma120:
            ma_desc = "🟢 완전 정배열 (강한 상승 추세)"
            trend_strength = "매우 강함"
        elif ma20 > ma60:
            ma_desc = "🟢 부분 정배열 (단기 상승 추세)"
            trend_strength = "보통"
        elif ma20 < ma60 < ma120:
            ma_desc = "🔴 완전 역배열 (강한 하락 추세)"
            trend_strength = "매우 약함"
        elif ma20 < ma60:
            ma_desc = "🔴 부분 역배열 (단기 하락 추세)"
            trend_strength = "약함"
        else:
            ma_desc = "🟡 혼재 (방향성 불분명)"
            trend_strength = "중립"

        # 거래량 분석
        vol_now = float(current.get('Volume', 0.0))
        if 'Volume_Ratio' in data.columns:
            vol_ratio = float(current.get('Volume_Ratio', 1.0))
        else:
            vol_ma20 = float(data['Volume'].rolling(20, min_periods=1).mean().iloc[-1])
            vol_ratio = (vol_now / vol_ma20) if vol_ma20 else 1.0

        if vol_ratio > 3.0:
            vol_desc = "🔥 대량 거래 (주목 필요)"
        elif vol_ratio > 2.0:
            vol_desc = "📈 높은 거래량 (관심 증가)"
        elif vol_ratio > 1.5:
            vol_desc = "📊 보통 이상 거래량"
        elif vol_ratio > 0.8:
            vol_desc = "⚪ 보통 거래량"
        else:
            vol_desc = "📉 낮은 거래량 (관심 부족)"

        # 종합 투자 의견
        bullish_points = 0
        bearish_points = 0
        
        # 점수 계산
        if macd_cross_up or (macd_now > macd_sig_now): bullish_points += 1
        if rsi_now < 30: bullish_points += 1
        if bb_position < 0.2: bullish_points += 1
        if ma20 > ma60 > ma120: bullish_points += 2
        elif ma20 > ma60: bullish_points += 1
        if vol_ratio > 1.5: bullish_points += 1

        if macd_cross_down or (macd_now < macd_sig_now): bearish_points += 1
        if rsi_now > 70: bearish_points += 1
        if bb_position > 0.8: bearish_points += 1
        if ma20 < ma60 < ma120: bearish_points += 2
        elif ma20 < ma60: bearish_points += 1

        # 종합 의견
        if bullish_points >= 4:
            overall = "🟢 강력 매수 추천"
        elif bullish_points >= 2 and bullish_points > bearish_points:
            overall = "🟢 매수 관심 구간"
        elif bearish_points >= 4:
            overall = "🔴 강력 매도 추천"
        elif bearish_points >= 2 and bearish_points > bullish_points:
            overall = "🔴 매도 관심 구간"
        else:
            overall = "⚪ 중립/관망 구간"

        # 최종 정보 텍스트 구성 (더 상세하고 구조화)
        info_text = f"""
    📊 {self.symbol} ({self.name}) - 현재 상황

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💰 가격 정보
    현재가: {current['Close']:.2f}
    전일대비: {price_change:+.2f} ({price_change_pct:+.2f}%)
    고가: {current['High']:.2f} | 저가: {current['Low']:.2f}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📈 기술적 지표 분석
    RSI: {rsi_now:.1f} → {rsi_desc}
    
    MACD: {macd_now:.4f} | Signal: {macd_sig_now:.4f}
    → {macd_desc}
    
    볼린저밴드: {bb_signal}
    → 현재 위치: {bb_position:.1%} (하단 0% ← → 100% 상단)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📏 이동평균선 분석
    20일선: {ma20:.2f}
    60일선: {ma60:.2f}
    120일선: {ma120:.2f}
    → {ma_desc}
    → 추세 강도: {trend_strength}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 거래량 분석
    현재 거래량: {vol_now:,.0f}
    20일 평균 대비: {vol_ratio:.2f}배
    → {vol_desc}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💡 종합 투자 의견
    매수 신호: {bullish_points}개
    매도 신호: {bearish_points}개
    
    → {overall}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ⚠️  투자 주의사항
    • 이 분석은 기술적 분석에 기반한 참고 자료입니다
    • 실제 투자 시에는 다양한 요소를 종합적으로 고려하세요
    • 리스크 관리를 위해 분산투자를 권장합니다
        """
        
        self.info_label.setText(info_text.strip())

    def create_info_panel_with_font_control(self):
        """정보 패널 + 폰트 크기 조정 기능"""
        group = QGroupBox("📊 Technical Indicators Info")
        main_layout = QVBoxLayout()
        
        # 폰트 크기 조정 버튼들
        font_control_layout = QHBoxLayout()
        font_control_layout.addWidget(QLabel("폰트:"))
        
        smaller_btn = QPushButton("A-")
        smaller_btn.setMaximumWidth(30)
        smaller_btn.clicked.connect(self.decrease_font_size)
        font_control_layout.addWidget(smaller_btn)
        
        larger_btn = QPushButton("A+")
        larger_btn.setMaximumWidth(30)
        larger_btn.clicked.connect(self.increase_font_size)
        font_control_layout.addWidget(larger_btn)
        
        font_control_layout.addStretch()
        main_layout.addLayout(font_control_layout)
        
        # 스크롤 영역 (위와 동일)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.info_label = QLabel("Loading chart data...")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop)
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        # 초기 폰트 설정
        self.current_font_size = 10
        self.update_info_font()
        
        scroll_layout.addWidget(self.info_label)
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)
        
        group.setLayout(main_layout)
        return group

    def increase_font_size(self):
        """폰트 크기 증가"""
        self.current_font_size = min(16, self.current_font_size + 1)
        self.update_info_font()

    def decrease_font_size(self):
        """폰트 크기 감소"""
        self.current_font_size = max(8, self.current_font_size - 1)
        self.update_info_font()

    def update_info_font(self):
        """정보 라벨 폰트 업데이트"""
        font = self.info_label.font()
        font.setPointSize(self.current_font_size)
        self.info_label.setFont(font)

    def closeEvent(self, event):
        """윈도우 닫을 때 메모리 정리"""
        try:
            # 차트 메모리 정리
            self.chart_manager.close_all()
            print("✅ 차트 메모리 정리 완료")
        except Exception as e:
            print(f"⚠️ 메모리 정리 오류: {e}")
        finally:
            event.accept()