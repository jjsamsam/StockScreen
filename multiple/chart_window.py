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

# 스마트 신호 생성기
from smart_signal_generator import SmartSignalGenerator

# 로거 설정
from logger_config import get_logger
logger = get_logger(__name__)

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
            logger.info(f"한글 폰트 설정: {korean_font}")
        else:
            # 한글 폰트가 없으면 기본 설정
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            logger.warning("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    except Exception as e:
        logger.warning(f"폰트 설정 중 오류: {e}")
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
        self.smart_signal_generator = SmartSignalGenerator()

        # 차트 메모리 관리자
        self.chart_manager = ChartManager()

        # 십자선 관련 변수 (여러 subplot 지원)
        self.crosshair_hline = None  # 가로선 (클릭한 subplot에만)
        self.crosshair_vlines = []   # 세로선 (모든 subplot에)
        self.crosshair_text = None
        self.crosshair_visible = False

        # 매매 신호 관련 변수
        self.show_signals = False
        self.buy_signals = []  # (날짜, 강도) 튜플 리스트
        self.sell_signals = []  # (날짜, 강도) 튜플 리스트
        self.signal_annotations = []

        # 한글 이름을 영문으로 변경 (폰트 문제 해결)
        display_name = name if not has_hangul(name) else symbol

        self.setWindowTitle(f'📊 {symbol} ({display_name}) - Technical Analysis Chart')

        # 윈도우 크기를 더 크게 설정
        self.setGeometry(100, 100, 1600, 1000)  # 1200x800 → 1600x1000

        self.setup_ui()
        self.load_chart_data()
    
    def setup_ui(self):
        """UI 설정 - 정보 패널을 스플리터로 개선"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 상단 컨트롤 패널 (높이 고정)
        control_panel = self.create_control_panel()
        control_panel.setMaximumHeight(80)
        main_layout.addWidget(control_panel)

        # 스플리터로 차트와 정보 패널 구분 (사용자가 크기 조절 가능)
        splitter = QSplitter(Qt.Vertical)

        # 차트 영역
        self.figure = Figure(figsize=(16, 12))
        self.canvas = FigureCanvas(self.figure)

        # 마우스 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        splitter.addWidget(self.canvas)

        # 하단 정보 패널 (스크롤 가능, 최소 높이만 설정)
        info_panel = self.create_info_panel()
        info_panel.setMinimumHeight(150)  # 최소 높이만 설정
        # setMaximumHeight 제거 - 사용자가 자유롭게 조절 가능
        splitter.addWidget(info_panel)

        # 초기 비율: 차트 70%, 정보 30%
        splitter.setSizes([700, 300])
        splitter.setStretchFactor(0, 7)  # 차트가 더 많은 공간
        splitter.setStretchFactor(1, 3)  # 정보 패널

        main_layout.addWidget(splitter)

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
        self.layout_combo.setCurrentText("Price Focus (2 Charts)")
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

        # 스크리닝 신호 표시 버튼 추가
        self.show_signals_btn = QPushButton("🎯 Show Buy/Sell Signals")
        self.show_signals_btn.clicked.connect(self.toggle_trading_signals)
        self.show_signals_btn.setCheckable(True)
        layout.addWidget(self.show_signals_btn)

        # 종목 검색 버튼 추가
        search_btn = QPushButton("🔍 Search Stock")
        search_btn.clicked.connect(self.show_stock_search_dialog)
        layout.addWidget(search_btn)

        layout.addStretch()
        group.setLayout(layout)
        return group

    def create_info_panel(self):
        """정보 패널 생성 - 스크롤 가능 + 폰트 크기 조절"""
        group = QGroupBox("📊 Technical Indicators Info")
        main_layout = QVBoxLayout()

        # 상단: 폰트 크기 조절 버튼
        font_control_layout = QHBoxLayout()
        font_control_layout.addWidget(QLabel("폰트 크기:"))

        # 초기 폰트 크기 저장
        self.current_font_size = 11  # 10 → 11로 증가

        decrease_font_btn = QPushButton("🔻 작게")
        decrease_font_btn.setMaximumWidth(80)
        decrease_font_btn.clicked.connect(lambda: self.adjust_font_size(-1))
        font_control_layout.addWidget(decrease_font_btn)

        increase_font_btn = QPushButton("🔺 크게")
        increase_font_btn.setMaximumWidth(80)
        increase_font_btn.clicked.connect(lambda: self.adjust_font_size(+1))
        font_control_layout.addWidget(increase_font_btn)

        reset_font_btn = QPushButton("↩️ 초기화")
        reset_font_btn.setMaximumWidth(80)
        reset_font_btn.clicked.connect(lambda: self.adjust_font_size(0, reset=True))
        font_control_layout.addWidget(reset_font_btn)

        font_control_layout.addStretch()
        main_layout.addLayout(font_control_layout)

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
        font.setPointSize(self.current_font_size)  # 크기 증가
        self.info_label.setFont(font)

        # 배경색과 패딩 설정
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 4px;
                padding: 15px;
                color: #212529;
            }
        """)

        # 스크롤 위젯에 라벨 추가
        scroll_layout.addWidget(self.info_label)
        scroll_layout.addStretch()  # 남은 공간 채우기

        # 스크롤 영역에 위젯 설정
        scroll_area.setWidget(scroll_widget)

        # 그룹박스에 스크롤 영역 추가
        main_layout.addWidget(scroll_area)
        group.setLayout(main_layout)

        return group

    def adjust_font_size(self, delta, reset=False):
        """폰트 크기 조절"""
        if reset:
            self.current_font_size = 11  # 초기값
        else:
            self.current_font_size = max(8, min(20, self.current_font_size + delta))

        font = self.info_label.font()
        font.setPointSize(self.current_font_size)
        self.info_label.setFont(font)

        logger.info(f"폰트 크기 변경: {self.current_font_size}pt")

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

            # 0원인 가격 데이터 필터링 (장 마감 후 무효 데이터 제거)
            # Close, Open, High, Low가 모두 0이거나 NaN인 행 제거
            invalid_mask = (
                (data['Close'] == 0) | (data['Close'].isna()) |
                (data['Open'] == 0) | (data['Open'].isna()) |
                (data['High'] == 0) | (data['High'].isna()) |
                (data['Low'] == 0) | (data['Low'].isna())
            )

            if invalid_mask.any():
                invalid_count = invalid_mask.sum()
                logger.warning(f"⚠️ 무효 데이터 {invalid_count}개 제거 (가격이 0원 또는 NaN)")
                data = data[~invalid_mask].copy()

                if data.empty:
                    error_msg = "❌ 유효한 가격 데이터가 없습니다."
                    self.info_label.setText(error_msg)
                    return

            # 기술적 지표 계산
            data = self.technical_analyzer.calculate_all_indicators(data)
            
            # 표시할 기간 필터링
            display_start_date = end_date - timedelta(days=display_days)
            import pandas as pd
            display_start_timestamp = pd.Timestamp(display_start_date)

            logger.info(f"📅 기간 필터링: {display_days}일 표시 (전체 데이터: {len(data)}개)")
            logger.info(f"   시작일: {display_start_timestamp}, 종료일: {end_date}")
            logger.info(f"   데이터 첫날: {data.index[0]}, 마지막날: {data.index[-1]}")

            display_data = data[data.index >= display_start_timestamp]

            logger.info(f"   필터링 후: {len(display_data)}개 데이터")

            if display_data.empty:
                display_rows = min(display_days, len(data))
                display_data = data.tail(display_rows)
                logger.warning(f"날짜 필터링 실패, 최근 {display_rows}개 데이터 사용")
            elif len(display_data) < display_days * 0.5:  # 예상보다 너무 적으면
                logger.warning(f"⚠️ 필터링된 데이터가 예상보다 적음: {len(display_data)}개 (예상: ~{display_days}개)")
                # 개수 기준으로 다시 필터링
                display_rows = min(display_days, len(data))
                display_data = data.tail(display_rows)
                logger.info(f"   → 최근 {display_rows}개 데이터로 전환")

            # 데이터 저장 (신호 감지용)
            self.data = data  # 전체 데이터 (기술적 지표 포함)

            self.plot_chart(display_data)
            self.update_info_panel(display_data)

        except Exception as e:
            error_msg = f"❌ 차트 로딩 오류: {str(e)}\n"
            error_msg += f"종목: {self.symbol}\n"
            error_msg += "다른 종목을 시도해보세요."
            self.info_label.setText(error_msg)
            logger.error(f"Chart loading error for {self.symbol}: {e}")

    def fetch_stock_data_with_retry(self, symbol, start_date, end_date):
        """여러 방법으로 주식 데이터 시도 (캐싱 사용)"""

        # 1차 시도: 원래 심볼 그대로 (표준 period 사용)
        try:
            logger.info(f"데이터 로딩 시도 1: {symbol}")
            # 기간 계산 - yfinance 표준 period 사용
            days_diff = (end_date - start_date).days

            # yfinance period 매핑
            if days_diff <= 7:
                period_str = "5d"
            elif days_diff <= 30:
                period_str = "1mo"
            elif days_diff <= 90:
                period_str = "3mo"
            elif days_diff <= 180:
                period_str = "6mo"
            elif days_diff <= 365:
                period_str = "1y"
            elif days_diff <= 730:
                period_str = "2y"
            else:
                period_str = "5y"

            logger.info(f"   기간: {days_diff}일 → {period_str}")
            # 차트용 데이터는 검증 비활성화
            data = get_stock_data(symbol, period=period_str, validate_cache=False)

            if data is not None and not data.empty and len(data) > 15:
                logger.info(f"✅ 성공: {symbol} - {len(data)}개 데이터")
                return data
            else:
                logger.warning(f"1차 시도 데이터 부족: {len(data) if data is not None else 0}개")
        except Exception as e:
            logger.error(f"1차 시도 실패: {e}")

        # 2차 시도: 심볼 변형 (한국 주식의 경우)
        if '.KQ' in symbol:
            try:
                alt_symbol = symbol.replace('.KQ', '.KS')
                logger.info(f"데이터 로딩 시도 2: {alt_symbol} (.KQ → .KS)")

                # 동일한 period 매핑 로직
                days_diff = (end_date - start_date).days
                if days_diff <= 180:
                    period_str = "6mo"
                elif days_diff <= 365:
                    period_str = "1y"
                elif days_diff <= 730:
                    period_str = "2y"
                else:
                    period_str = "5y"

                data = get_stock_data(alt_symbol, period=period_str, validate_cache=False)

                if data is not None and not data.empty and len(data) > 15:
                    logger.info(f"✅ 성공: {alt_symbol} - {len(data)}개 데이터")
                    return data
            except Exception as e:
                logger.error(f"2차 시도 실패: {e}")

        elif '.KS' in symbol:
            try:
                alt_symbol = symbol.replace('.KS', '.KQ')
                logger.info(f"데이터 로딩 시도 2: {alt_symbol} (.KS → .KQ)")

                # 동일한 period 매핑 로직
                days_diff = (end_date - start_date).days
                if days_diff <= 180:
                    period_str = "6mo"
                elif days_diff <= 365:
                    period_str = "1y"
                elif days_diff <= 730:
                    period_str = "2y"
                else:
                    period_str = "5y"

                data = get_stock_data(alt_symbol, period=period_str, validate_cache=False)

                if data is not None and not data.empty and len(data) > 15:
                    logger.info(f"성공: {alt_symbol} - {len(data)}개 데이터")
                    return data
            except Exception as e:
                logger.error(f"2차 시도 실패: {e}")

        # 3차 시도: max 기간으로 시도 (validate_cache=False로 검증 완화)
        try:
            logger.info(f"데이터 로딩 시도 3: {symbol} (max 기간, 검증 비활성화)")
            data = get_stock_data(symbol, period="max", validate_cache=False)

            if data is not None and not data.empty and len(data) > 50:
                logger.info(f"✅ 성공 (max): {symbol} - {len(data)}개 데이터")
                return data
        except Exception as e:
            logger.error(f"3차 시도 실패: {e}")

        # 4차 시도: 단기 데이터 (최후의 수단)
        try:
            logger.info(f"데이터 로딩 시도 4: {symbol} (단기 1mo)")
            data = get_stock_data(symbol, period="1mo", validate_cache=False)

            if data is not None and not data.empty:
                logger.info(f"성공 (단기): {symbol} - {len(data)}개 데이터")
                return data
        except Exception as e:
            logger.error(f"4차 시도 실패: {e}")

        logger.error(f"모든 시도 실패: {symbol}")
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

        # 매매 신호 화살표 표시
        if self.show_signals:
            self._plot_trading_signals(ax, data)

    def _plot_trading_signals(self, ax, data):
        """매매 신호 화살표 그리기 - 강도별 크기 차별화"""
        logger.info(f"화살표 그리기: 매수 {len(self.buy_signals)}개, 매도 {len(self.sell_signals)}개")

        # 강도별 폰트 크기 매핑 (0:없음, 25:10, 50:15, 75:20, 100:25)
        def get_fontsize(strength):
            size_map = {0: 0, 25: 12, 50: 16, 75: 20, 100: 24}
            return size_map.get(strength, 15)

        # 매수 신호 - 빨간색 위쪽 화살표 (강도별 크기)
        buy_count = 0
        for buy_signal in self.buy_signals:
            buy_date, strength = buy_signal
            if buy_date in data.index:
                price = data.loc[buy_date, 'Low'] * 0.98  # 최저가보다 약간 아래
                fontsize = get_fontsize(strength)
                ax.annotate('▲', xy=(buy_date, price),
                           xytext=(0, -15), textcoords='offset points',
                           fontsize=fontsize, color='red', ha='center',
                           weight='bold', alpha=0.8)
                buy_count += 1

        # 매도 신호 - 파란색 아래쪽 화살표 (강도별 크기)
        sell_count = 0
        for sell_signal in self.sell_signals:
            sell_date, strength = sell_signal
            if sell_date in data.index:
                price = data.loc[sell_date, 'High'] * 1.02  # 최고가보다 약간 위
                fontsize = get_fontsize(strength)
                ax.annotate('▼', xy=(sell_date, price),
                           xytext=(0, 15), textcoords='offset points',
                           fontsize=fontsize, color='blue', ha='center',
                           weight='bold', alpha=0.8)
                sell_count += 1

        logger.info(f"화살표 표시 완료: 매수 {buy_count}개, 매도 {sell_count}개 (표시 영역 내)")

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

        # 종합 투자 의견 (기존 방식 - 호환성 유지)
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

        # 🚀 스마트 신호 생성 (새로운 시스템)
        try:
            smart_signal = self.smart_signal_generator.generate_signal(data)
        except Exception as e:
            logger.warning(f"스마트 신호 생성 실패: {e}")
            smart_signal = None

        # ADX 및 ATR 정보 추가
        adx_value = float(current.get('ADX', 0))
        atr_value = float(current.get('ATR', 0))
        plus_di = float(current.get('+DI', 0))
        minus_di = float(current.get('-DI', 0))

        # 스마트 신호 섹션 구성
        if smart_signal:
            smart_section = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🤖 AI 스마트 신호 분석 (NEW!)
    시장 환경: {smart_signal['regime_kr']} (ADX: {smart_signal['adx']:.1f})
    신호: {smart_signal['signal']} | 신뢰도: {smart_signal['confidence']:.0f}%
    종합 의견: {smart_signal['recommendation']}

    📊 점수:
    • 매수 점수: {smart_signal['bullish_score']:.1f}점
    • 매도 점수: {smart_signal['bearish_score']:.1f}점

    📍 리스크 관리 제안:
    • 손절가: {smart_signal['stop_loss']:.2f} (ATR 2배)
    • 목표가: {smart_signal['take_profit']:.2f} (ATR 3배)
    • 손익비: 1:{smart_signal['risk_reward_ratio']:.1f}
    """
        else:
            smart_section = ""

        # 추세 강도 섹션
        adx_section = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 추세 강도 분석 (NEW!)
    ADX: {adx_value:.1f} {"🔥 강한 추세" if adx_value > 25 else "💤 약한 추세 (횡보)"}
    +DI: {plus_di:.1f} | -DI: {minus_di:.1f}
    {"→ 상승 우세" if plus_di > minus_di else "→ 하락 우세"}

    ATR (변동성): {atr_value:.2f}
    → {"높은 변동성" if atr_value > current['Close'] * 0.03 else "낮은 변동성"}
    """

        # 최종 정보 텍스트 구성 (더 상세하고 구조화)
        info_text = f"""
    📊 {self.symbol} ({self.name}) - 현재 상황

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💰 가격 정보
    현재가: {current['Close']:.2f}
    전일대비: {price_change:+.2f} ({price_change_pct:+.2f}%)
    고가: {current['High']:.2f} | 저가: {current['Low']:.2f}
{smart_section}
{adx_section}
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
    💡 기존 투자 의견 (참고용)
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

    def on_mouse_press(self, event):
        """마우스 누를 때 십자선 표시"""
        if event.inaxes is None:
            return
        self.draw_crosshair(event)

    def on_mouse_release(self, event):
        """마우스 뗄 때 십자선 제거"""
        self.remove_crosshair()

    def draw_crosshair(self, event):
        """십자선 그리기 - 모든 subplot에 세로선 표시"""
        if event.inaxes is None:
            return

        ax = event.inaxes
        x, y = event.xdata, event.ydata

        # 기존 십자선 제거
        self.remove_crosshair()

        # 가로선은 클릭한 subplot에만 그리기
        self.crosshair_hline = ax.axhline(y, color='black', linewidth=0.5, linestyle='-', alpha=0.8)

        # 세로선은 모든 subplot에 그리기
        for subplot_ax in self.figure.get_axes():
            vline = subplot_ax.axvline(x, color='black', linewidth=0.5, linestyle='-', alpha=0.8)
            self.crosshair_vlines.append(vline)

        # 값 표시 텍스트
        try:
            # x축이 날짜인 경우 변환
            if hasattr(ax, 'get_xlim'):
                xlim = ax.get_xlim()
                if x >= xlim[0] and x <= xlim[1]:
                    date_str = mdates.num2date(x).strftime('%Y-%m-%d')
                else:
                    date_str = f"X: {x:.2f}"
            else:
                date_str = f"X: {x:.2f}"
        except:
            date_str = f"X: {x:.2f}"

        text = f"{date_str}\nY: {y:.2f}"
        self.crosshair_text = ax.text(0.02, 0.98, text, transform=ax.transAxes,
                                      fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        self.crosshair_visible = True
        self.canvas.draw()

    def remove_crosshair(self):
        """십자선 제거"""
        # 가로선 제거
        if self.crosshair_hline:
            self.crosshair_hline.remove()
            self.crosshair_hline = None

        # 모든 세로선 제거
        for vline in self.crosshair_vlines:
            vline.remove()
        self.crosshair_vlines = []

        # 텍스트 제거
        if self.crosshair_text:
            self.crosshair_text.remove()
            self.crosshair_text = None

        self.crosshair_visible = False
        self.canvas.draw()

    def toggle_trading_signals(self):
        """매매 신호 표시 토글"""
        self.show_signals = self.show_signals_btn.isChecked()

        if self.show_signals:
            # 신호 계산 및 표시
            self.detect_trading_signals()
            self.load_chart_data()  # 차트 다시 그리기
        else:
            # 신호 제거
            self.buy_signals = []
            self.sell_signals = []
            self.load_chart_data()  # 차트 다시 그리기

    def detect_trading_signals(self):
        """매매 신호 감지 - 스크리닝 조건 활용"""
        if not hasattr(self, 'data') or self.data is None:
            logger.warning("데이터가 없어 신호 감지 불가")
            return

        if len(self.data) < 120:
            logger.warning(f"데이터 부족: {len(self.data)}개 (최소 120개 필요)")
            return

        self.buy_signals = []
        self.sell_signals = []

        logger.info(f"신호 감지 시작: {len(self.data)}개 데이터")

        # 매일 조건 체크
        for i in range(120, len(self.data)):
            data_slice = self.data.iloc[:i+1].copy()
            date = self.data.index[i]

            # 매수 조건 체크 (4가지 조건)
            buy_strength = self.check_buy_signal_strength(data_slice)
            if buy_strength > 0:
                self.buy_signals.append((date, buy_strength))
                logger.info(f"매수 신호: {date.strftime('%Y-%m-%d')} (강도: {buy_strength})")

            # 매도 조건 체크 (4가지 조건)
            sell_strength = self.check_sell_signal_strength(data_slice)
            if sell_strength > 0:
                self.sell_signals.append((date, sell_strength))
                logger.info(f"매도 신호: {date.strftime('%Y-%m-%d')} (강도: {sell_strength})")

        logger.info(f"✅ 매수 신호: {len(self.buy_signals)}개, 매도 신호: {len(self.sell_signals)}개")

    def check_buy_signal_strength(self, data):
        """매수 신호 강도 체크 - 4가지 조건 (0/25/50/75/100)"""
        try:
            current = data.iloc[-1]
            strength = 0

            # 조건 1: 60일선이 120일선 상향돌파 + 현재가 > 60일선 (25점)
            if current['MA60'] > current['MA120'] and current['Close'] > current['MA60']:
                for i in range(max(0, len(data)-10), len(data)):
                    if i > 0:
                        prev = data.iloc[i-1]
                        curr = data.iloc[i]
                        if prev['MA60'] <= prev['MA120'] and curr['MA60'] > curr['MA120']:
                            strength += 25
                            break

            # 조건 2: 볼린저밴드 하단 터치 + RSI < 35 (25점)
            # BB_Lower (대문자) 사용
            if 'BB_Lower' in current and 'RSI' in current:
                if current['Close'] <= current['BB_Lower'] * 1.02 and current['RSI'] < 35:
                    strength += 25

            # 조건 3: MACD 골든 크로스 + 거래량 증가 (25점)
            if 'MACD' in current and 'MACD_Signal' in current:
                for i in range(max(0, len(data)-5), len(data)):
                    if i > 0:
                        prev = data.iloc[i-1]
                        curr = data.iloc[i]
                        if prev['MACD'] <= prev['MACD_Signal'] and curr['MACD'] > curr['MACD_Signal']:
                            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                            if current['Volume'] > avg_volume * 1.2:
                                strength += 25
                            break

            # 조건 4: 20일 상대강도 상승 (25점)
            if len(data) >= 20:
                ma20_slope = (current['MA20'] - data['MA20'].iloc[-20]) / data['MA20'].iloc[-20]
                if ma20_slope > 0.02:
                    strength += 25

            return strength
        except Exception as e:
            logger.debug(f"매수 신호 체크 오류: {e}")
            return 0

    def check_sell_signal_strength(self, data):
        """매도 신호 강도 체크 - 4가지 조건 (0/25/50/75/100)"""
        try:
            current = data.iloc[-1]
            strength = 0

            # 조건 1: 데드크로스 + 60일선 3% 하향이탈 (25점)
            if current['MA60'] < current['MA120']:
                for i in range(max(0, len(data)-10), len(data)):
                    if i > 0:
                        prev = data.iloc[i-1]
                        curr = data.iloc[i]
                        if prev['MA60'] >= prev['MA120'] and curr['MA60'] < curr['MA120']:
                            if current['Close'] < current['MA60'] * 0.97:
                                strength += 25
                            break

            # 조건 2: 20% 수익달성 또는 -7% 손절 (25점)
            if len(data) >= 20:
                recent_low = data['Low'].rolling(20).min().iloc[-1]
                gain = (current['Close'] - recent_low) / recent_low
                if gain > 0.20 or gain < -0.07:
                    strength += 25

            # 조건 3: 볼린저 상단 + RSI > 70 (25점)
            # BB_Upper (대문자) 사용
            if 'BB_Upper' in current and 'RSI' in current:
                if current['Close'] >= current['BB_Upper'] * 0.98 and current['RSI'] > 70:
                    strength += 25

            # 조건 4: 거래량 급감 + 모멘텀 약화 (25점)
            if len(data) >= 20:
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                if current['Volume'] < avg_volume * 0.6:
                    ma20_slope = (current['MA20'] - data['MA20'].iloc[-5]) / data['MA20'].iloc[-5]
                    if ma20_slope < -0.01:
                        strength += 25

            return strength
        except Exception as e:
            logger.debug(f"매도 신호 체크 오류: {e}")
            return 0

    def show_stock_search_dialog(self):
        """종목 검색 다이얼로그 표시"""
        dialog = StockSearchDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.get_selected_stock()
            if selected:
                # 새로운 종목으로 차트 변경
                self.symbol = selected['ticker']
                self.name = selected['name']
                self.setWindowTitle(f'📊 {self.symbol} ({self.name}) - Technical Analysis Chart')
                self.load_chart_data()

    def closeEvent(self, event):
        """윈도우 닫을 때 메모리 정리"""
        try:
            # 차트 메모리 정리
            self.chart_manager.close_all()
            logger.info("차트 메모리 정리 완료")
        except Exception as e:
            logger.warning(f"메모리 정리 오류: {e}")
        finally:
            event.accept()


class StockSearchDialog(QDialog):
    """종목 검색 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_stock = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('🔍 종목 검색')
        self.setGeometry(300, 300, 600, 500)

        layout = QVBoxLayout()

        # 검색 입력
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("검색어:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("종목코드 또는 이름 입력 (예: AAPL, 삼성전자, 005930)")
        self.search_input.returnPressed.connect(self.search_stocks)
        search_layout.addWidget(self.search_input)

        search_btn = QPushButton("검색")
        search_btn.clicked.connect(self.search_stocks)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # 결과 레이블
        self.result_label = QLabel("검색어를 입력하세요")
        layout.addWidget(self.result_label)

        # 결과 테이블
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(3)
        self.result_table.setHorizontalHeaderLabels(["종목코드", "종목명", "시장"])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.doubleClicked.connect(self.on_stock_selected)
        layout.addWidget(self.result_table)

        # 버튼
        button_layout = QHBoxLayout()
        select_btn = QPushButton("선택")
        select_btn.clicked.connect(self.on_stock_selected)
        button_layout.addWidget(select_btn)

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        # 포커스
        self.search_input.setFocus()

    def search_stocks(self):
        """종목 검색 실행 - CSV + 온라인"""
        search_term = self.search_input.text().strip()
        logger.info(f"검색 시작: '{search_term}'")

        if not search_term:
            self.result_label.setText("검색어를 입력하세요")
            return

        try:
            # 1. CSV에서 검색 (unified_search 모듈 사용)
            logger.info("CSV 검색 시도...")
            from unified_search import search_stocks
            results = search_stocks(search_term)
            logger.info(f"CSV 검색 결과: {len(results) if results else 0}개")

            self.result_table.setRowCount(0)

            if not results:
                # 2. CSV에서 없으면 온라인 검색 시도
                logger.info("온라인 검색 시도...")
                self.result_label.setText("CSV에서 검색 결과 없음. 온라인 검색 중...")
                QApplication.processEvents()

                online_results = self.try_online_search(search_term)
                if online_results:
                    logger.info(f"온라인 검색 성공: {len(online_results)}개 발견")
                    # 온라인에서 찾은 결과들을 모두 테이블에 추가
                    for row_idx, stock in enumerate(online_results[:20]):  # 최대 20개
                        self.result_table.insertRow(row_idx)
                        self.result_table.setItem(row_idx, 0, QTableWidgetItem(stock['ticker']))
                        self.result_table.setItem(row_idx, 1, QTableWidgetItem(stock['name']))
                        self.result_table.setItem(row_idx, 2, QTableWidgetItem(stock['market']))
                    self.result_table.selectRow(0)
                    self.result_label.setText(f"🌐 온라인에서 {len(online_results)}개 종목 발견")
                else:
                    logger.warning(f"'{search_term}' 검색 결과 없음")
                    self.result_label.setText(f"'{search_term}'에 대한 검색 결과가 없습니다")
                return

            self.result_label.setText(f"📁 CSV에서 {len(results)}개 종목 발견")

            # 테이블에 결과 표시
            for row_idx, stock in enumerate(results[:20]):  # 최대 20개만 표시
                self.result_table.insertRow(row_idx)
                self.result_table.setItem(row_idx, 0, QTableWidgetItem(stock['ticker']))
                self.result_table.setItem(row_idx, 1, QTableWidgetItem(stock['name']))
                self.result_table.setItem(row_idx, 2, QTableWidgetItem(stock['market']))

            # 첫 번째 행 선택
            if self.result_table.rowCount() > 0:
                self.result_table.selectRow(0)
            logger.info("검색 완료")

        except Exception as e:
            logger.error(f"검색 오류: {e}", exc_info=True)
            self.result_label.setText(f"검색 오류: {str(e)}")

    def try_online_search(self, search_term):
        """온라인에서 Yahoo Finance API로 종목 검색"""
        try:
            import urllib.parse
            import requests

            logger.info(f"Yahoo Finance API로 '{search_term}' 검색...")
            query = urllib.parse.quote(search_term)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            res = requests.get(url, headers=headers, timeout=10)
            logger.info(f"API 응답 코드: {res.status_code}")

            if res.ok:
                data = res.json()
                quotes = data.get('quotes', [])
                logger.info(f"API에서 {len(quotes)}개 종목 발견")

                # 모든 결과를 리스트로 반환
                if quotes:
                    results = []
                    for quote in quotes[:20]:  # 최대 20개
                        ticker = quote.get('symbol', '')
                        name = quote.get('longname') or quote.get('shortname') or ticker
                        exchange = quote.get('exchange', 'Online')

                        # 시장 분류
                        market = "Online"
                        if '.KS' in ticker:
                            market = "KOSPI (Online)"
                        elif '.KQ' in ticker:
                            market = "KOSDAQ (Online)"
                        elif '.ST' in ticker:
                            market = "OMX (Online)"
                        elif exchange:
                            market = f"{exchange} (Online)"

                        result = {
                            'ticker': ticker,
                            'name': name,
                            'market': market
                        }
                        results.append(result)

                    logger.info(f"온라인 검색 결과: {len(results)}개 반환")
                    return results

            logger.warning("API 응답 없음")
            return []

        except Exception as e:
            logger.error(f"온라인 검색 오류: {e}", exc_info=True)
            return []

    def on_stock_selected(self):
        """종목 선택"""
        current_row = self.result_table.currentRow()
        if current_row >= 0:
            ticker = self.result_table.item(current_row, 0).text()
            name = self.result_table.item(current_row, 1).text()
            market = self.result_table.item(current_row, 2).text()

            self.selected_stock = {
                'ticker': ticker,
                'name': name,
                'market': market
            }
            self.accept()

    def get_selected_stock(self):
        """선택된 종목 반환"""
        return self.selected_stock