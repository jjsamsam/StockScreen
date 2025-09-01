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

from chart_window import StockChartWindow
from dialogs import CSVEditorDialog, ConditionBuilderDialog, ConditionManagerDialog
#from utils import UpdateThread, TechnicalAnalysis, export_screening_results
from utils import TechnicalAnalysis, export_screening_results
from utils import SmartUpdateThread
from utils import MasterCSVThread, MasterFilterThread

from trend_analysis import TrendTimingAnalyzer
from backtesting_system import BacktestingDialog

class StockScreener(QMainWindow):
    def __init__(self):
        super().__init__()
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
        
        self.initUI()
        self.setup_stock_lists()
        
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
        self.statusbar.showMessage('준비됨 - 샘플 생성 버튼을 클릭하여 시작하세요')
    
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
        
        # 여섯 번째 행: 사용자 정의 조건
        custom_group = QGroupBox("⚙️ 사용자 정의 조건")
        custom_layout = QHBoxLayout()
        
        self.add_condition_btn = QPushButton("➕ 조건 추가")
        self.add_condition_btn.clicked.connect(self.open_condition_builder)
        self.add_condition_btn.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; }")
        custom_layout.addWidget(self.add_condition_btn)
        
        self.manage_conditions_btn = QPushButton("⚙️ 조건 관리")
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
        layout.addWidget(custom_group, 4, 0, 1, 6)  # 행 4에 배치
        
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

        layout.addLayout(button_layout, 5, 0, 1, 6)  # 행 5에 배치

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
        group.setMaximumHeight(60)  # 최대 높이 제한
        group.setMinimumHeight(60)  # 최소 높이도 고정
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
        """테이블 생성 - 정렬 기능 포함"""
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
        
        self.sell_table.doubleClicked.connect(self.show_stock_detail)
        sell_layout.addWidget(self.sell_table)
        sell_group.setLayout(sell_layout)
        
        splitter.addWidget(buy_group)
        splitter.addWidget(sell_group)
        
        return splitter

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
    
    # def update_stocks_online(self):
    #     """온라인에서 종목 리스트 업데이트"""
    #     reply = QMessageBox.question(self, '확인', 
    #                                 '온라인에서 전체 종목 리스트를 업데이트하시겠습니까?\n'
    #                                 '• 한국: KOSPI/KOSDAQ 종목 \n'
    #                                 '• 미국: NASDAQ 종목 \n'
    #                                 '• 스웨덴: OMX Stockholm 종목\n\n'
    #                                 '이 작업은 몇 분 소요될 수 있습니다.',
    #                                 QMessageBox.Yes | QMessageBox.No)
        
    #     if reply == QMessageBox.Yes:
    #         self.update_online_btn.setEnabled(False)
    #         self.statusbar.showMessage('🌐 온라인 종목 업데이트 중...')
            
    #         # 별도 스레드에서 실행
    #         self.update_thread = UpdateThread()
    #         self.update_thread.finished.connect(self.on_update_finished)
    #         self.update_thread.error.connect(self.on_update_error)
    #         self.update_thread.progress.connect(self.on_update_progress)  # 진행상황 연결
    #         self.update_thread.start()
    
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
            
            delete_btn = QPushButton("❌")
            delete_btn.clicked.connect(lambda checked, idx=i: self.delete_custom_condition(idx))
            delete_btn.setMaximumWidth(30)
            delete_btn.setToolTip("조건 삭제")
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
            self.statusbar.showMessage('📁 CSV 파일 로드 완료')
            
        except Exception as e:
            QMessageBox.warning(self, "오류", f"CSV 파일 로드 중 오류: {str(e)}")
    
    def update_stock_count(self):
        """종목 개수 업데이트"""
        korea_count = len(self.stock_lists.get('korea', []))
        usa_count = len(self.stock_lists.get('usa', []))
        sweden_count = len(self.stock_lists.get('sweden', []))
        total_count = korea_count + usa_count + sweden_count
        
        self.korea_count_label.setText(f"🇰🇷 한국: {korea_count}개")
        self.usa_count_label.setText(f"🇺🇸 미국: {usa_count}개")
        self.sweden_count_label.setText(f"🇸🇪 스웨덴: {sweden_count}개")
        self.total_count_label.setText(f"🌍 전체: {total_count}개")
    
    def open_csv_editor(self):
        """CSV 파일 편집 다이얼로그"""
        dialog = CSVEditorDialog(self)
        dialog.exec_()
        self.load_stock_lists()  # 편집 후 새로고침
    
    def get_selected_stocks(self):
        """선택된 시장의 종목들 반환 + 시가총액 필터링"""
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

        # 시가총액 필터링 적용
        if self.use_market_cap_filter.isChecked() and stocks:
            top_count = self.top_stocks_spin.value()
            
            # 시가총액으로 정렬 (내림차순)
            try:
                stocks_with_mcap = []
                for stock in stocks:
                    mcap = stock.get('market_cap', 0)

                    # 문자열 변환 처리
                    if isinstance(mcap, str):
                        # 모든 쉼표, 공백 제거하고 대문자 변환
                        mcap_clean = re.sub(r'[,\s]', '', mcap.upper())
                        
                        try:
                            if mcap_clean.endswith('B'):
                                mcap = float(mcap_clean[:-1]) * 1e9
                            elif mcap_clean.endswith('M'):
                                mcap = float(mcap_clean[:-1]) * 1e6
                            elif mcap_clean.endswith('K'):
                                mcap = float(mcap_clean[:-1]) * 1e3
                            else:
                                mcap = float(mcap_clean)
                        except (ValueError, TypeError):
                            mcap = 0

                    # 숫자 변환 처리
                    if isinstance(mcap, (int, float)) and mcap > 0:
                        # 변환된 숫자 값을 stock에 저장
                        stock_copy = stock.copy()
                        stock_copy['market_cap_numeric'] = mcap  # 숫자 값 저장
                        stocks_with_mcap.append(stock_copy)
                
                # 시가총액 기준 정렬
                stocks_with_mcap.sort(key=lambda x: float(x.get('market_cap_numeric', 0)), reverse=True)
                
                # 상위 N개만 선택
                stocks = stocks_with_mcap[:top_count]
                
                self.statusbar.showMessage(f'💰 시가총액 상위 {len(stocks)}개 종목으로 필터링됨')
                
            except Exception as e:
                print(f"시가총액 필터링 중 오류: {e}")
                # 오류 발생 시 원본 리스트 사용
        
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
        - 2024-08-20 현재, 22일 이내(7월 29일 이후)에 돌파했는지 확인
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
        
        예시: 2024년 8월 10일에 60일선이 120일선을 돌파했다면
        - 체크 기간: 2024년 5월 10일 ~ 2024년 8월 9일 (66거래일)
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
        """안전한 주식 데이터 가져오기"""
        try:
            stock = yf.Ticker(symbol)
            
            # 짧은 타임아웃으로 빠르게 시도
            data = stock.history(start=start_date, end=end_date, timeout=5)
            
            if not data.empty:
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
                # 빠른 기본 정보 체크
                stock = yf.Ticker(symbol)
                info = stock.info
                
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

    def show_stock_detail(self, index):
        """테이블에서 종목 더블클릭시 상세 차트 표시"""
        try:
            # 어느 테이블에서 클릭했는지 확인
            table = self.sender()
            row = index.row()
            
            # 종목 코드와 이름 가져오기
            symbol = table.item(row, 0).text() if table.item(row, 0) else ""
            name = table.item(row, 1).text() if table.item(row, 1) else symbol
            
            if not symbol:
                QMessageBox.warning(self, "경고", "종목 정보를 가져올 수 없습니다.")
                return
            
            # 차트 윈도우 생성 및 표시
            try:
                # chart_window.py에서 StockChartWindow 임포트 시도
                from chart_window import StockChartWindow
                
                chart_window = StockChartWindow(symbol, name, self)
                chart_window.show()
                
                self.statusbar.showMessage(f"📊 {symbol} ({name}) 차트를 열었습니다.")
                
            except ImportError:
                # StockChartWindow를 찾을 수 없는 경우 간단한 메시지 표시
                QMessageBox.information(self, "차트", 
                                    f"종목: {symbol} ({name})\n"
                                    f"차트 기능을 사용하려면 chart_window.py 파일이 필요합니다.")
                                    
            except Exception as chart_error:
                # 차트 생성 중 오류 발생시
                QMessageBox.warning(self, "차트 오류", 
                                f"차트를 불러오는 중 오류가 발생했습니다:\n{str(chart_error)}")
                
        except Exception as e:
            print(f"Error in show_stock_detail: {e}")
            QMessageBox.critical(self, "오류", f"종목 상세 정보를 표시하는 중 오류가 발생했습니다:\n{str(e)}")

    # ========== 추가로 필요한 간단한 차트 기능 (chart_window.py가 없는 경우) ==========

    def show_simple_stock_info(self, symbol, name):
        """간단한 종목 정보 다이얼로그 (차트 대안)"""
        try:
            # yfinance로 기본 정보 가져오기
            import yfinance as yf
            from datetime import datetime, timedelta
            
            stock = yf.Ticker(symbol)
            
            # 최근 1개월 데이터
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = stock.history(start=start_date, end=end_date)
            
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