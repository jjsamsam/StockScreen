"""
prediction_window.py
AI 예측 윈도우 - Enhanced Screener 통합 버전

✅ 변경 사항:
- CPUOptimizedPredictor의 train_and_predict 제거
- enhanced_screener.py의 EnhancedCPUPredictor.predict_stock 사용
- 더 나은 성능과 일관성 제공
- 중복 코드 제거
"""

import yfinance as yf
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtCore import QTimer
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import requests
import urllib.parse
import os
import json

# Enhanced Screener의 예측기 import
try:
    from enhanced_screener import EnhancedCPUPredictor
    ML_AVAILABLE = True
    print("✅ Enhanced Screener 예측기 사용")
except ImportError as e:
    print(f"⚠️ Enhanced Screener 없음: {e}")
    ML_AVAILABLE = False

# 기본 라이브러리 확인
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

print("""
🔧 Prediction Window 업데이트:
• Enhanced Screener 통합 완료
• 중복 예측 함수 제거
• 일관성 있는 예측 결과
• 더 나은 성능과 정확도
""")


class StockPredictionDialog(QDialog):
    """주식 예측 다이얼로그 - Enhanced Screener 통합 버전"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Enhanced Screener의 예측기 사용
        self.predictor = EnhancedCPUPredictor() if ML_AVAILABLE else None
        
        # ✨ 진행률 추적 변수들 추가
        self.prediction_steps = [
            "데이터 수집 중",
            "기술적 지표 계산 중", 
            "특성 생성 중",
            "모델 학습 중",
            "예측 실행 중",
            "결과 처리 중"
        ]
        self.current_step = 0
        self.total_steps = len(self.prediction_steps)

        self.load_current_settings()
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('🤖 AI 주식 예측 (Enhanced)')
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # 상단 입력 패널
        input_panel = self.create_input_panel()
        layout.addWidget(input_panel)
        
        # 결과 표시 영역
        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setFont(QFont("Consolas", 10))
        layout.addWidget(self.result_area)
        
        # 차트 영역
        self.chart_widget = self.create_chart_widget()
        layout.addWidget(self.chart_widget)
        
        # 하단 버튼
        button_layout = self.create_enhanced_button_layout()  # 새로운 함수
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
        # 상태 표시
        if not ML_AVAILABLE:
            self.result_area.setText("""
⚠️ Enhanced Screener가 필요합니다.

enhanced_screener.py 파일이 있는지 확인하고,
다음 라이브러리를 설치해주세요:

pip install scikit-learn xgboost lightgbm statsmodels

📊 Enhanced Screener의 장점:
• 완전한 일관성 보장 (랜덤 시드 고정)
• 30개 이상의 고급 기술적 지표
• 정교한 앙상블 예측
• 실제 현재가와 예측 기준가 분리
• 강력한 데이터 검증 및 오류 처리
            """)

    def load_current_settings(self):
        """✅ 새로 추가: 현재 설정 파일에서 값 로드"""
        self.current_settings = {
            'forecast_days': 7,
            'confidence_threshold': 0.6,
            'batch_delay': 1.0,
            'min_data_days': 300,
            'use_arima_validation': True,
            'models_enabled': {
                'xgboost': True,
                'lightgbm': True,
                'random_forest': True,
                'extra_trees': True,
                'gradient_boosting': True
            }
        }
        
        try:
            if os.path.exists('prediction_settings.json'):
                with open('prediction_settings.json', 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                self.current_settings.update(saved_settings)
                print(f"✅ Prediction Window 설정 로드: {saved_settings.get('forecast_days', 7)}일 예측")
        except Exception as e:
            print(f"⚠️ Prediction Window 설정 로드 실패: {e}")

    def create_enhanced_button_layout(self):
        """향상된 버튼 레이아웃 - 예측 차트 버튼 추가"""
        button_layout = QHBoxLayout()
        
        # 기존 예측 시작 버튼
        self.predict_btn = QPushButton('🚀 AI 예측 시작')
        self.predict_btn.clicked.connect(self.start_prediction_enhanced)  # 새로운 함수 연결
        button_layout.addWidget(self.predict_btn)
        
        # ✨ 새로운 예측 차트 버튼
        self.chart_btn = QPushButton('📈 예측 차트 보기')
        self.chart_btn.clicked.connect(self.show_prediction_chart)
        self.chart_btn.setEnabled(False)  # 예측 완료 후 활성화
        self.chart_btn.setToolTip('현재부터 예측일까지의 주가 변화 차트를 보여줍니다')
        button_layout.addWidget(self.chart_btn)
        
        # 기존 내보내기 버튼
        self.export_btn = QPushButton('📊 결과 내보내기')
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        # 닫기 버튼
        close_btn = QPushButton('닫기')
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        return button_layout

    def create_input_panel(self):
        """입력 패널 생성 - 마스터 CSV 검색 기능 추가"""
        panel = QGroupBox("🎯 예측 설정")
        layout = QGridLayout()
        
        # 종목 코드 입력 및 검색
        layout.addWidget(QLabel("종목 코드:"), 0, 0)
        
        # 종목 입력 레이아웃 (입력창 + 검색 버튼)
        ticker_layout = QHBoxLayout()
        
        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setPlaceholderText("예: AAPL, MSFT, 005930.KS, 삼성")
        ticker_layout.addWidget(self.ticker_input)
        
        # 종목 검색 버튼
        self.search_btn = QPushButton("🔍")
        self.search_btn.setToolTip("종목 검색 (마스터 CSV)")
        self.search_btn.setMaximumWidth(40)
        self.search_btn.clicked.connect(self.show_enhanced_stock_search_dialog)
        ticker_layout.addWidget(self.search_btn)
        
        # 자동완성 기능
        self.ticker_input.textChanged.connect(self.on_ticker_text_changed)
        
        ticker_widget = QWidget()
        ticker_widget.setLayout(ticker_layout)
        layout.addWidget(ticker_widget, 0, 1)
        
        # 예측 기간
        layout.addWidget(QLabel("예측 기간:"), 1, 0)
        days_layout = QHBoxLayout()
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 30)
        # ✅ 설정 파일에서 가져온 값으로 초기화
        self.days_input.setValue(self.current_settings.get('forecast_days', 7))
        self.days_input.setSuffix(" 일")
        days_layout.addWidget(self.days_input)
        
        # ✅ 새로 추가: 설정 정보 표시 라벨
        self.setting_info_label = QLabel(f"(설정파일: {self.current_settings.get('forecast_days', 7)}일)")
        self.setting_info_label.setStyleSheet("color: #666; font-size: 10px;")
        days_layout.addWidget(self.setting_info_label)
        
        # ✅ 새로 추가: 설정 동기화 버튼
        self.sync_settings_btn = QPushButton("⚙️")
        self.sync_settings_btn.setToolTip("설정 파일과 동기화")
        self.sync_settings_btn.setMaximumWidth(30)
        self.sync_settings_btn.clicked.connect(self.sync_with_settings)
        days_layout.addWidget(self.sync_settings_btn)
        
        days_widget = QWidget()
        days_widget.setLayout(days_layout)
        layout.addWidget(days_widget, 1, 1)
        
        # 모델 선택 (Enhanced Screener 정보 표시)
        layout.addWidget(QLabel("사용 모델:"), 2, 0)
        
        model_layout = QVBoxLayout()
        
        # 모델 정보 표시
        if ML_AVAILABLE:
            enabled_models = self.current_settings.get('models_enabled', {})
            active_models = [name for name, enabled in enabled_models.items() if enabled]
            
            self.model_combo = QComboBox()
            self.model_combo.addItems([
                f"🚀 Enhanced Ensemble ({len(active_models)}개 모델 활성화)",
                f"📊 활성 모델: {', '.join(active_models[:3])}" + ("..." if len(active_models) > 3 else ""),
                "🎯 성능 기반 가중치 + 설정 연동",
                "🔒 완전한 일관성 보장"
            ])
            
            # ✅ 새로 추가: 모델별 체크박스 표시 (읽기전용 정보)
            models_info = []
            for model_name, enabled in enabled_models.items():
                status = "✅" if enabled else "❌"
                models_info.append(f"{status} {model_name}")
            
            self.models_info_label = QLabel(" | ".join(models_info))
            self.models_info_label.setStyleSheet("color: #666; font-size: 9px;")
            self.models_info_label.setWordWrap(True)
            
        else:
            self.model_combo = QComboBox()
            self.model_combo.addItems(["❌ Enhanced Screener 필요"])
            self.models_info_label = QLabel("Enhanced Screener를 설치해주세요")
        
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.models_info_label)
        
        model_widget = QWidget()
        model_widget.setLayout(model_layout)
        layout.addWidget(model_widget, 2, 1)
        
        # ✅ 새로 추가: 추가 설정 정보
        layout.addWidget(QLabel("기타 설정:"), 3, 0)
        
        settings_info = f"최소데이터: {self.current_settings.get('min_data_days', 300)}일 | "
        settings_info += f"신뢰도임계값: {self.current_settings.get('confidence_threshold', 0.6)*100:.0f}%"
        
        self.settings_summary_label = QLabel(settings_info)
        self.settings_summary_label.setStyleSheet("color: #444; font-size: 10px;")
        layout.addWidget(self.settings_summary_label, 3, 1)
        
        panel.setLayout(layout)
        return panel

    def sync_with_settings(self):
            """✅ 새로 추가: 설정 파일과 동기화"""
            self.load_current_settings()
            
            # UI 업데이트
            self.days_input.setValue(self.current_settings.get('forecast_days', 7))
            self.setting_info_label.setText(f"(설정파일: {self.current_settings.get('forecast_days', 7)}일)")
            
            # 모델 정보 업데이트
            if ML_AVAILABLE:
                enabled_models = self.current_settings.get('models_enabled', {})
                active_models = [name for name, enabled in enabled_models.items() if enabled]
                
                # 콤보박스 업데이트
                self.model_combo.clear()
                self.model_combo.addItems([
                    f"🚀 Enhanced Ensemble ({len(active_models)}개 모델 활성화)",
                    f"📊 활성 모델: {', '.join(active_models[:3])}" + ("..." if len(active_models) > 3 else ""),
                    "🎯 성능 기반 가중치 + 설정 연동",
                    "🔒 완전한 일관성 보장"
                ])
                
                # 모델 정보 라벨 업데이트
                models_info = []
                for model_name, enabled in enabled_models.items():
                    status = "✅" if enabled else "❌"
                    models_info.append(f"{status} {model_name}")
                self.models_info_label.setText(" | ".join(models_info))
            
            # 기타 설정 정보 업데이트
            settings_info = f"최소데이터: {self.current_settings.get('min_data_days', 300)}일 | "
            settings_info += f"신뢰도임계값: {self.current_settings.get('confidence_threshold', 0.6)*100:.0f}%"
            self.settings_summary_label.setText(settings_info)
            
            QMessageBox.information(self, "설정 동기화", 
                                f"✅ 설정이 동기화되었습니다!\n\n"
                                f"• 예측 기간: {self.current_settings.get('forecast_days', 7)}일\n"
                                f"• 활성 모델: {len(active_models)}개\n"
                                f"• 최소 데이터: {self.current_settings.get('min_data_days', 300)}일")


    def show_enhanced_stock_search_dialog(self):
        """마스터 CSV를 활용한 종목 검색 다이얼로그 표시"""
        dialog = EnhancedStockSearchDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_ticker = dialog.get_selected_ticker()
            if selected_ticker:
                self.ticker_input.setText(selected_ticker)

    def on_ticker_text_changed(self, text):
        """종목 코드 입력 시 간단한 유효성 검사"""
        text = text.strip().upper()
        
        # 자동 대문자 변환
        if text != self.ticker_input.text():
            cursor_pos = self.ticker_input.cursorPosition()
            self.ticker_input.setText(text)
            self.ticker_input.setCursorPosition(cursor_pos)
        
        # 간단한 형식 체크
        if len(text) > 0:
            if text.replace('.', '').replace('-', '').isalnum():
                self.ticker_input.setStyleSheet("")  # 정상
            else:
                self.ticker_input.setStyleSheet("border: 1px solid orange;")  # 경고
        else:
            self.ticker_input.setStyleSheet("")
    
    def create_chart_widget(self):
        """차트 위젯 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        self.figure = Figure(figsize=(10, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        widget.setLayout(layout)
        return widget
    
    def create_button_layout_enhanced(self):
        """향상된 버튼 레이아웃 - 예측 차트 버튼 추가"""
        button_layout = QHBoxLayout()
        
        # 기존 버튼들
        self.predict_btn = QPushButton('🚀 AI 예측 시작')
        self.predict_btn.clicked.connect(self.start_prediction_enhanced)
        button_layout.addWidget(self.predict_btn)
        
        # ✨ 새로운 예측 차트 버튼
        self.chart_btn = QPushButton('📈 예측 차트 보기')
        self.chart_btn.clicked.connect(self.show_prediction_chart)
        self.chart_btn.setEnabled(False)  # 예측 완료 후 활성화
        self.chart_btn.setToolTip('현재부터 예측일까지의 주가 변화 차트를 보여줍니다')
        button_layout.addWidget(self.chart_btn)
        
        # 기존 버튼들
        self.export_btn = QPushButton('📊 결과 내보내기')
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        close_btn = QPushButton('닫기')
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        return button_layout

    def show_prediction_chart(self):
        """예측 차트 다이얼로그 표시"""
        if not hasattr(self, 'last_result'):
            QMessageBox.warning(self, "오류", "먼저 AI 예측을 실행해주세요.")
            return
        
        # 차트 다이얼로그 생성
        chart_dialog = PredictionChartDialog(self.last_result, self)
        chart_dialog.exec_()

    def start_prediction_enhanced(self):
        """진행률 표시가 포함된 Enhanced 예측 시작"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "Enhanced Screener가 설치되지 않았습니다.")
            return
        
        ticker = self.ticker_input.text().strip().upper()
        days = self.days_input.value()

        if not ticker:
            QMessageBox.warning(self, "오류", "종목 코드를 입력해주세요.")
            return

        if days <= 5:
            period_type = "단기"
            description = "빠른 반응, 단기 패턴 포착"
        elif days <= 14:
            period_type = "중기"
            description = "균형잡힌 설정"
        else:
            period_type = "장기"
            description = "추세 중심, 장기 패턴"
        reply = QMessageBox.question(
            self, "예측 모드 확인",
            f"📊 {ticker} 예측\n\n"
            f"• 예측 기간: {days}일\n"
            f"• 모드: {period_type} 최적화\n"
            f"• 특징: {description}\n\n"
            f"이 설정으로 예측하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return

        # UI 비활성화
        self.predict_btn.setEnabled(False)
        if hasattr(self, 'chart_btn'):
            self.chart_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        
        # ✨ 진행률 초기화
        self.current_step = 0
        self.prediction_ticker = ticker
        self.prediction_days = days
        self.prediction_start_time = datetime.now()
        
        # 비동기 예측 시작
        self.start_step_by_step_prediction()

    def on_prediction_finished_enhanced(self, result, error_msg):
        """Enhanced 예측 완료 처리 - 차트 버튼 활성화 추가"""
        self.predict_btn.setEnabled(True)
        
        if error_msg:
            QMessageBox.critical(self, "예측 오류", f"예측 실패:\n{error_msg}")
            return
        
        if result is None:
            QMessageBox.warning(self, "예측 실패", "예측 결과를 받을 수 없습니다.")
            return
        
        # 결과 저장 및 표시
        self.last_result = result
        self.display_results(result)
        
        # 기존 단순 차트도 표시 (기본)
        self.plot_prediction_timeseries(result)
        
        # ✨ 버튼들 활성화
        self.export_btn.setEnabled(True)
        self.chart_btn.setEnabled(True)  # 예측 차트 버튼 활성화
        
        # 성공 메시지
        QMessageBox.information(self, "예측 완료", 
                            f"✅ {result['ticker']} AI 예측이 완료되었습니다!\n\n"
                            f"📈 '예측 차트 보기' 버튼을 눌러 상세 차트를 확인하세요.")

    def run_prediction_step(self, ticker, forecast_days):
        """실제 예측 실행"""
        try:
            # predictor.predict_stock()이 자동으로 forecast_days에 맞게 최적화됨
            result, error = self.predictor.predict_stock(ticker, forecast_days=forecast_days)
            
            # 결과 처리
            self.on_prediction_finished_enhanced(result, error)
            
        except Exception as e:
            self.on_prediction_finished_enhanced(None, str(e))

    def start_step_by_step_prediction(self):
        """단계별 예측 실행 - 진행률 표시와 함께"""
        # self.prediction_timer = QTimer()
        # self.prediction_timer.timeout.connect(self.execute_next_prediction_step)
        # self.prediction_timer.start(300)  # 300ms마다 다음 단계

        """단계별 예측 실행"""
        # 예측 기간 가져오기
        forecast_days = self.days_input.value()
        ticker = self.ticker_input.text().strip().upper()
        
        # 예측 기간 정보 표시
        period_type = "단기" if forecast_days <= 5 else "중기" if forecast_days <= 14 else "장기"
        self.result_area.append(f"\n{'='*50}")
        self.result_area.append(f"📊 {ticker} {period_type} 예측 ({forecast_days}일)")
        self.result_area.append(f"{'='*50}\n")
        
        # 진행률 초기화
        self.current_step = 0
        
        # 비동기 예측 시작 (predictor가 자동으로 최적화)
        QTimer.singleShot(100, lambda: self.run_prediction_step(ticker, forecast_days))

    def execute_next_prediction_step(self):
        """예측의 다음 단계 실행"""
        if self.current_step >= self.total_steps:
            self.prediction_timer.stop()
            self.finalize_prediction()
            return
        
        step_name = self.prediction_steps[self.current_step]
        progress_percent = int((self.current_step / self.total_steps) * 100)
        
        try:
            # ✨ 진행 상태 업데이트
            self.update_progress_display(progress_percent, step_name)
            QApplication.processEvents()
            
            # 각 단계별 작업 (시뮬레이션 + 실제 작업)
            if self.current_step == 0:
                self.step_1_collect_data()
            elif self.current_step == 1:
                self.step_2_calculate_indicators()  
            elif self.current_step == 2:
                self.step_3_generate_features()
            elif self.current_step == 3:
                self.step_4_train_models()
            elif self.current_step == 4:
                self.step_5_make_prediction()  # 실제 예측 실행
            elif self.current_step == 5:
                self.step_6_process_results()
            
            self.current_step += 1
            
        except Exception as e:
            self.prediction_timer.stop()
            self.handle_prediction_error(f"단계 {self.current_step + 1} 오류: {str(e)}")

    def update_progress_display(self, percent, step_name):
        """진행률과 단계 이름으로 UI 업데이트"""
        # 애니메이션 점들
        dots = "." * ((percent // 8) % 4)
        
        # ✨ 버튼 텍스트 업데이트
        self.predict_btn.setText(f"🔄 {step_name} ({percent}%){dots}")
        
        # ✨ 결과 영역에 진행 바 표시
        progress_text = f"""
    🤖 AI 예측 진행 중...

    📊 종목: {self.prediction_ticker}
    📅 예측 기간: {self.prediction_days}일
    ⏱️ 경과 시간: {self.get_elapsed_time()}

    {'='*25} 진행 상황 {'='*25}

    """
        
        # 텍스트 진행률 바
        bar_length = 35
        filled_length = int(bar_length * percent / 100)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        progress_text += f"[{bar}] {percent}%\n\n"
        
        # 단계별 체크 표시
        for i, step in enumerate(self.prediction_steps):
            if i < self.current_step:
                status = "✅"
            elif i == self.current_step:
                status = "🔄"
            else:
                status = "⏳"
            progress_text += f"{status} {step}\n"
        
        progress_text += f"\n💡 현재: {step_name}{dots}"
        
        self.result_area.setText(progress_text)

    def get_elapsed_time(self):
        """예측 시작부터 경과 시간"""
        if not hasattr(self, 'prediction_start_time'):
            return "0초"
        
        elapsed = datetime.now() - self.prediction_start_time
        seconds = int(elapsed.total_seconds())
        
        if seconds < 60:
            return f"{seconds}초"
        else:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes}분 {seconds}초"

    # 각 단계별 작업 함수들 (시뮬레이션)
    def step_1_collect_data(self):
        """1단계: 데이터 수집"""
        import time
        time.sleep(0.2)  # 시각적 효과

    def step_2_calculate_indicators(self):
        """2단계: 기술적 지표 계산"""
        import time
        time.sleep(0.3)

    def step_3_generate_features(self):
        """3단계: 특성 생성"""
        import time
        time.sleep(0.4)

    def step_4_train_models(self):
        """4단계: 모델 학습"""
        import time
        time.sleep(0.6)  # 가장 오래 걸림

    def step_5_make_prediction(self):
        """5단계: 실제 예측 실행"""
        import time
        time.sleep(0.2)
        
        # ✅ 실제 Enhanced Screener 예측 실행
        self.prediction_result, self.prediction_error = self.predictor.predict_stock(
            self.prediction_ticker, 
            forecast_days=self.prediction_days
        )

    def step_6_process_results(self):
        """6단계: 결과 처리"""
        import time
        time.sleep(0.1)
        
        if self.prediction_result and not self.prediction_error:
            self.final_result = self.convert_enhanced_result(
                self.prediction_result, 
                self.prediction_days
            )

    def finalize_prediction(self):
        """예측 완료 후 최종 처리"""
        try:
            # 100% 완료 표시
            self.update_progress_display(100, "완료!")
            
            if hasattr(self, 'prediction_error') and self.prediction_error:
                self.handle_prediction_error(self.prediction_error)
                return
            
            if not hasattr(self, 'final_result') or not self.final_result:
                self.handle_prediction_error("예측 결과를 받을 수 없습니다.")
                return
            
            # ✅ 성공 처리
            self.last_result = self.final_result
            
            # 결과 표시 (기존 함수 사용)
            self.display_results(self.final_result)
            self.plot_prediction_timeseries(self.final_result)
            
            # 버튼 활성화
            self.predict_btn.setEnabled(True)
            if hasattr(self, 'chart_btn'):
                self.chart_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            self.predict_btn.setText("🚀 AI 예측 시작")  # 텍스트 복원
            
            # 성공 메시지
            QMessageBox.information(self, "예측 완료", 
                                f"✅ {self.prediction_ticker} AI 예측이 완료되었습니다!")
                                
        except Exception as e:
            self.handle_prediction_error(f"최종 처리 오류: {str(e)}")

    def handle_prediction_error(self, error_message):
        """예측 오류 처리"""
        # UI 복원
        self.predict_btn.setEnabled(True)
        if hasattr(self, 'chart_btn'):
            self.chart_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.predict_btn.setText("🚀 AI 예측 시작")
        
        # 오류 표시
        self.result_area.setText(f"""
    ❌ 예측 실패

    종목: {getattr(self, 'prediction_ticker', 'N/A')}
    오류: {error_message}

    다시 시도해주세요.
        """)
        
        QMessageBox.critical(self, "예측 오류", f"예측 실패:\n{error_message}")

    def convert_enhanced_result(self, enhanced_result, days):
        """Enhanced Screener 결과를 기존 UI 형식으로 변환"""
        try:
            # Enhanced Screener 결과 구조:
            # {
            #     'ticker': ticker,
            #     'current_price': actual_current_price,
            #     'predicted_price': predicted_price_actual,
            #     'expected_return': predicted_return,
            #     'confidence': confidence,
            #     'successful_models': successful_models,
            #     'model_results': model_results,
            #     'individual_predictions': predictions,
            #     ...
            # }
            
            # 기존 UI가 기대하는 형식으로 변환
            converted = {
                'ticker': enhanced_result.get('ticker', ''),
                'current_price': enhanced_result.get('current_price', 0),
                'predicted_price': enhanced_result.get('predicted_price', 0),
                'expected_return': enhanced_result.get('expected_return', 0),
                'confidence': enhanced_result.get('confidence', 0),
                'days': days,
                'data_points': 600,  # Enhanced Screener는 고정 600일 사용
                'training_samples': enhanced_result.get('training_samples', 0),
                
                # 모델별 결과 변환
                'model_scores': {},
                'individual_predictions': {},
                
                # Enhanced 정보
                'successful_models': enhanced_result.get('successful_models', 0),
                'feature_count': enhanced_result.get('feature_count', 0),
                'prediction_date': enhanced_result.get('prediction_date', ''),
                'method': 'Enhanced Screener'
            }
            
            # 모델별 결과 변환
            model_results = enhanced_result.get('model_results', {})
            individual_predictions = enhanced_result.get('individual_predictions', [])
            
            for i, (model_name, model_data) in enumerate(model_results.items()):
                r2_score = model_data.get('r2_score', 0)
                prediction = model_data.get('prediction', 0)
                
                converted['model_scores'][model_name] = max(0, r2_score)  # R² -> 점수 변환
                converted['individual_predictions'][model_name] = prediction
            
            return converted
            
        except Exception as e:
            print(f"결과 변환 오류: {e}")
            # 최소한의 결과 반환
            return {
                'ticker': enhanced_result.get('ticker', ''),
                'current_price': enhanced_result.get('current_price', 0),
                'predicted_price': enhanced_result.get('predicted_price', 0),
                'expected_return': enhanced_result.get('expected_return', 0),
                'confidence': enhanced_result.get('confidence', 0),
                'days': days,
                'method': 'Enhanced Screener',
                'model_scores': {},
                'individual_predictions': {},
                'data_points': 600,
                'training_samples': 0
            }
    
    def display_results(self, result):
        """✅ 수정: 신뢰도 임계값 정보가 포함된 결과 표시"""
        return_rate = result['expected_return']
        confidence = result['confidence']
        
        # ✅ 신뢰도 임계값 정보 가져오기
        confidence_threshold = result.get('confidence_threshold', 0.6)
        is_high_confidence = result.get('is_high_confidence', True)
        recommendation = result.get('recommendation', '⏸️ 관망')
        confidence_note = result.get('confidence_note', '')
        
        # ✅ 신뢰도에 따른 색상 결정
        if is_high_confidence:
            if return_rate > 0.02:
                color = "🟢"
            elif return_rate < -0.02:
                color = "🔴"
            else:
                color = "⚪"
        else:
            color = "🟡"  # 낮은 신뢰도는 항상 노란색
        
        # ✅ 신뢰도 상태 표시
        confidence_status = f"✅ {confidence*100:.1f}%" if is_high_confidence else f"⚠️ {confidence*100:.1f}%"
        confidence_bar = "█" * min(10, int(confidence * 10)) + "░" * (10 - min(10, int(confidence * 10)))
        
        # 결과 텍스트 생성
        text = f"""
══════════════════════════════════════════════════
🎯 {result['ticker']} Enhanced AI 예측 ({result['days']}일 후)
══════════════════════════════════════════════════

💰 현재 가격: ${result['current_price']:.2f}
🎯 예측 가격: ${result['predicted_price']:.2f}
📊 예상 수익률: {return_rate*100:+.2f}%

🎚️ 신뢰도: {confidence_status}
   [{confidence_bar}] {confidence*100:.1f}% / {confidence_threshold*100:.0f}%
   {confidence_note}

{color} 추천: {recommendation}

──────────────────────────────────────────────────
🔧 신뢰도 분석:
──────────────────────────────────────────────────
• 설정한 임계값: {confidence_threshold*100:.0f}%
• 현재 신뢰도: {confidence*100:.1f}%
• 신뢰도 상태: {'높음 (임계값 이상)' if is_high_confidence else '낮음 (임계값 미만)'}
• 모델 일치도: {'높음' if confidence > 0.8 else '보통' if confidence > 0.6 else '낮음'}

{'✅ 일관된 예측 - 투자 참고 가능' if is_high_confidence else '⚠️ 불일치 예측 - 신중한 판단 필요'}

──────────────────────────────────────────────────
🚀 Enhanced Screener 분석 정보:
──────────────────────────────────────────────────
• 예측 방법: {result.get('method', 'Enhanced Screener')}
• 성공한 모델: {result.get('successful_models', 0)}개
• 사용된 특성: {result.get('feature_count', 30)}개 이상
• 데이터 기간: {result['data_points']}일 (고정)
• 학습 샘플: {result['training_samples']}개
• 예측 완료: {result.get('prediction_date', 'N/A')}

──────────────────────────────────────────────────
📈 모델별 성능 및 예측:
──────────────────────────────────────────────────
"""
        
        # 기존 모델별 결과 표시 (그대로 유지)
        model_scores = result.get('model_scores', {})
        individual_predictions = result.get('individual_predictions', {})
        
        if model_scores:
            for model_name in model_scores.keys():
                score = model_scores.get(model_name, 0)
                pred = individual_predictions.get(model_name, 0)
                text += f"{model_name:15}: R² {score:.3f} | 예측 {pred*100:+.2f}%\n"
        else:
            text += "앙상블 예측 결과만 사용됨\n"
        
        text += f"""
──────────────────────────────────────────────────
🔧 Enhanced 기술 정보:
──────────────────────────────────────────────────
• 랜덤 시드 고정: 완전한 일관성 보장
• 현재가 분리: 실제 vs 예측 기준가
• 고급 특성: RSI, MACD, 볼린저 밴드 등
• 시퀀스 학습: 30일 패턴 분석
• 앙상블 방식: 성능 기반 가중 평균
• 신뢰도 필터링: 임계값 {confidence_threshold*100:.0f}% 적용

💡 참고: {'신뢰도가 높아 투자 참고 가능합니다.' if is_high_confidence else '신뢰도가 낮아 추가 검토가 필요합니다.'}
        """
        
        self.result_area.setText(text)
    
    def plot_prediction_timeseries(self, result):
        """시계열 예측 차트 그리기 - 마커 오류 수정 버전"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        forecast_days = result['days']
        
        try:
            # 📊 1. 과거 데이터 가져오기 (최근 30일)
            ticker = result['ticker']
            import yfinance as yf
            from datetime import datetime, timedelta
            
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=45)
            historical_data = stock.history(start=start_date, end=end_date)
            
            if len(historical_data) == 0:
                print("⚠️ 과거 데이터 없음 - 단순 차트로 대체")
                self.plot_prediction_simple(result)
                return
            
            # 📈 2. 과거 주가 데이터 준비 (최근 30일)
            historical_dates = historical_data.index[-30:]
            historical_prices = historical_data['Close'].iloc[-30:].values
            
            # 📊 3. 미래 날짜 생성 (영업일 기준)
            import pandas as pd
            last_date = historical_dates[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                        periods=forecast_days)
            
            # 📈 4. 예측 가격 생성 (부드러운 곡선)
            current_price = result['current_price']
            target_price = result['predicted_price']
            
            predicted_prices = []
            for i in range(forecast_days):
                progress = (i + 1) / forecast_days
                # 시그모이드 함수로 부드러운 변화
                smooth_progress = 1 / (1 + np.exp(-5 * (progress - 0.5)))
                predicted_price = current_price + (target_price - current_price) * smooth_progress
                predicted_prices.append(predicted_price)
            
            predicted_prices = np.array(predicted_prices)
            
            # 🎨 5. 차트 그리기 - 호환성 개선된 마커 사용
            # 5-1. 과거 데이터 (파란색 실선)
            ax.plot(historical_dates, historical_prices, 'b-', 
                    label='과거 주가', linewidth=2, alpha=0.8)
            
            # 5-2. 예측 데이터 (빨간색 점선) - 표준 마커 사용
            ax.plot(future_dates, predicted_prices, 'r--', 
                    label='AI 예측', linewidth=2.5, marker='o', markersize=4)
            
            # 5-3. 연결선
            ax.plot([historical_dates[-1], future_dates[0]], 
                    [historical_prices[-1], predicted_prices[0]], 
                    'g:', linewidth=1.5, alpha=0.7, label='연결선')
            
            # 5-4. 현재가 강조 (원형 마커)
            ax.scatter([historical_dates[-1]], [current_price], 
                    color='orange', s=100, zorder=5, marker='o', 
                    edgecolors='black', linewidth=2, label='현재가')
            
            # 5-5. 목표가 강조 - ★ 대신 호환성 높은 마커 사용
            try:
                # 첫 번째 시도: 별 마커 (최신 matplotlib)
                ax.scatter([future_dates[-1]], [target_price], 
                        color='red', s=150, zorder=5, marker='*', 
                        edgecolors='darkred', linewidth=2, label='예측가')
            except Exception:
                try:
                    # 두 번째 시도: 다이아몬드 마커
                    ax.scatter([future_dates[-1]], [target_price], 
                            color='red', s=120, zorder=5, marker='D', 
                            edgecolors='darkred', linewidth=2, label='예측가')
                except Exception:
                    # 마지막 대안: 사각형 마커
                    ax.scatter([future_dates[-1]], [target_price], 
                            color='red', s=120, zorder=5, marker='s', 
                            edgecolors='darkred', linewidth=2, label='예측가')
            
            # 📊 6. 신뢰도 구간 표시 (선택적)
            confidence = result.get('confidence', 0.7)
            if confidence < 0.9:  # 신뢰도가 낮을 때만 구간 표시
                confidence_range = predicted_prices * (1 - confidence) * 0.05  # 범위 축소
                ax.fill_between(future_dates, 
                            predicted_prices - confidence_range,
                            predicted_prices + confidence_range,
                            alpha=0.15, color='red', label=f'신뢰구간 ({confidence*100:.0f}%)')
            
            # 🎯 7. 차트 스타일링
            return_pct = result.get('expected_return', 0) * 100
            title = f"{ticker} AI 주가 예측 ({forecast_days}일)"
            subtitle = f"현재: ${current_price:.2f} → 예측: ${target_price:.2f} ({return_pct:+.1f}%)"
            
            ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('날짜', fontsize=12)
            ax.set_ylabel('주가 ($)', fontsize=12)
            
            # 범례 위치 최적화
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                    bbox_to_anchor=(0.02, 0.98))
            
            # 격자 스타일
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Y축 포맷팅 (달러 표시)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # X축 날짜 포맷팅 - 오류 방지
            try:
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(historical_dates)//8)))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            except Exception as e:
                print(f"⚠️ 날짜 포맷팅 오류 (무시됨): {e}")
            
            # 📈 8. 추가 정보 텍스트 박스 (오류 방지)
            try:
                info_text = f"신뢰도: {confidence*100:.1f}%\n"
                info_text += f"예측 모델: {result.get('method', 'Enhanced AI')}\n"
                info_text += f"데이터: {len(historical_dates)}일"
                
                ax.text(0.02, 0.75, info_text, transform=ax.transAxes, 
                        verticalalignment='top', 
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8), 
                        fontsize=9)
            except Exception as e:
                print(f"⚠️ 정보 텍스트 박스 오류 (무시됨): {e}")
            
            # 레이아웃 조정
            plt.tight_layout()
            
        except Exception as e:
            print(f"⚠️ 고급 시계열 차트 생성 실패: {e}")
            print("📊 단순 차트로 대체합니다...")
            # 모든 오류에 대해 백업 차트 사용
            self.plot_prediction_simple(result)
            return
        
        # 캔버스 업데이트
        try:
            self.canvas.draw()
        except Exception as e:
            print(f"⚠️ 캔버스 그리기 오류: {e}")
            # 캔버스 오류시에도 백업 차트 시도
            self.plot_prediction_simple(result)

    def plot_prediction_simple(self, result):
        """기존 단순 막대 차트 (백업용) - 안정성 개선"""
        try:
            ax = self.figure.add_subplot(111)
            
            # 간단한 가격 예측 차트
            days = ['현재', f'{result["days"]}일 후']
            prices = [result['current_price'], result['predicted_price']]
            
            # 색상 결정
            expected_return = result.get('expected_return', 0)
            colors = ['steelblue', 'green' if expected_return > 0 else 'red']
            
            # 막대 차트
            bars = ax.bar(days, prices, color=colors, alpha=0.7, edgecolor='black')
            
            # 수익률 표시
            return_pct = expected_return * 100
            ax.text(1, result['predicted_price'], f'{return_pct:+.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            # 차트 스타일링
            ax.set_title(f"{result['ticker']} AI 예측 ({result['days']}일)", 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel("주가 ($)", fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Y축 포맷팅
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
            
            # 신뢰도 정보 추가
            confidence_pct = result.get('confidence', 0.7) * 100
            ax.text(0.5, max(prices) * 0.9, f'신뢰도: {confidence_pct:.1f}%', 
                    ha='center', fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # 레이아웃 조정
            plt.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"❌ 단순 차트도 실패: {e}")
            # 최후의 수단: 텍스트만 표시
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"차트 생성 오류\n\n{result['ticker']}\n"
                    f"현재: ${result['current_price']:.2f}\n"
                    f"예측: ${result['predicted_price']:.2f}", 
                    ha='center', va='center', fontsize=14, 
                    transform=ax.transAxes)
            self.canvas.draw()

    # 추가: 마커 호환성 테스트 함수
    def test_marker_compatibility():
        """matplotlib 마커 호환성 테스트"""
        import matplotlib.pyplot as plt
        
        test_markers = ['*', '★', 'D', 's', 'o', '^', 'v', '<', '>']
        compatible_markers = []
        
        fig, ax = plt.subplots()
        
        for i, marker in enumerate(test_markers):
            try:
                ax.scatter([i], [i], marker=marker, s=100)
                compatible_markers.append(marker)
                print(f"✅ 마커 '{marker}' 호환됨")
            except Exception as e:
                print(f"❌ 마커 '{marker}' 호환되지 않음: {e}")
        
        plt.close(fig)
        return compatible_markers

    # 사용 예시:
    # compatible_markers = test_marker_compatibility()
    # print(f"호환 가능한 마커들: {compatible_markers}")
    
    def export_results(self):
        """결과 내보내기"""
        if not hasattr(self, 'last_result'):
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_prediction_{self.last_result['ticker']}_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.result_area.toPlainText())
            
            QMessageBox.information(self, "저장 완료", f"Enhanced 예측 결과가 {filename}에 저장되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"파일 저장 중 오류: {str(e)}")

# ===============================================
# 기존 검색 다이얼로그들 (변경 없음)
# ===============================================

class StockSearchDialog(QDialog):
    """기본 종목 검색 다이얼로그"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('종목 검색')
        self.setGeometry(300, 300, 400, 300)
        self.selected_ticker = None
        
        layout = QVBoxLayout()
        
        # 검색 입력
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("종목 코드 또는 회사명 입력...")
        layout.addWidget(self.search_input)
        
        # 결과 리스트
        self.results_list = QListWidget()
        layout.addWidget(self.results_list)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        select_btn = QPushButton("선택")
        select_btn.clicked.connect(self.select_ticker)
        button_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.last_search_results = []  # 마지막 검색 결과 저장용
        
        # CSV 내보내기 버튼 추가 (UI에)
        self.add_csv_export_button()
    
    def add_csv_export_button(self):
        """CSV 내보내기 버튼을 UI에 추가"""
        # 기존 버튼 레이아웃에 추가
        csv_btn = QPushButton("📄 CSV 보기")
        csv_btn.setToolTip("검색 결과를 CSV 형태로 보기/내보내기")
        csv_btn.clicked.connect(self.show_csv_export_dialog)
        
        # 기존 버튼 레이아웃에 추가 (search_btn 옆에)
        # button_layout.addWidget(csv_btn)  # 실제 UI 레이아웃에 맞게 위치 조정 필요
        
        self.csv_export_btn = csv_btn  # 참조 저장

    def select_ticker(self):
        current_item = self.results_list.currentItem()
        if current_item:
            self.selected_ticker = current_item.text().split()[0]  # 첫 번째 단어가 티커
            self.accept()
    
    def get_selected_ticker(self):
        return self.selected_ticker


class EnhancedStockSearchDialog(QDialog):
    """Enhanced 종목 검색 다이얼로그 (마스터 CSV 활용)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('🔍 Enhanced 종목 검색 (Master CSV)')
        self.setGeometry(300, 300, 700, 500)
        self.selected_ticker = None
        self.search_cache = {}  # 캐시 추가

        # ✅ 디바운스 타이머 추가
        self.search_timer = QTimer()
        self.search_timer.setSingleShot(True)  # 한 번만 실행
        self.search_timer.timeout.connect(self.perform_search)
        
        self.initUI()

        # 초기 인기 종목 표시
        self.show_popular_stocks()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 상단 정보
        info_label = QLabel("💡 종목을 검색합니다")
        info_label.setStyleSheet("color: #2196F3; font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # 검색 입력
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("종목 코드, 회사명, 또는 섹터 입력 (예: AAPL, 삼성, 005930)")
        self.search_input.textChanged.connect(self.on_search_text_changed)
        self.search_input.returnPressed.connect(self.perform_search)
        search_layout.addWidget(self.search_input)
        
        search_btn = QPushButton("🔍 검색")
        search_btn.clicked.connect(self.perform_search)
        search_layout.addWidget(search_btn)
        
        layout.addLayout(search_layout)
        
        # 빠른 검색 버튼들
        quick_layout = QHBoxLayout()
        quick_layout.addWidget(QLabel("빠른 검색:"))
        
        popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', '005930.KS', '000660.KS']
        for ticker in popular_tickers:
            btn = QPushButton(ticker)
            btn.setMaximumWidth(80)
            btn.clicked.connect(lambda checked, t=ticker: self.quick_search(t))
            quick_layout.addWidget(btn)
        
        quick_layout.addStretch()
        layout.addLayout(quick_layout)
        
        # 결과 테이블
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(['종목코드', '회사명', '시장', '섹터', '시가총액', '매치점수'])
        self.results_table.doubleClicked.connect(self.select_from_table)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        layout.addWidget(self.results_table)
        
        # 상태 레이블
        self.status_label = QLabel("검색어를 입력하거나 빠른 검색 버튼을 클릭하세요")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 새로고침")
        refresh_btn.setToolTip("마스터 CSV 다시 로드")
        refresh_btn.clicked.connect(self.refresh_search)
        button_layout.addWidget(refresh_btn)
        
        button_layout.addStretch()
        
        select_btn = QPushButton("✅ 선택")
        select_btn.clicked.connect(self.select_ticker)
        button_layout.addWidget(select_btn)
        
        cancel_btn = QPushButton("❌ 취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def on_search_text_changed(self, text):
        """검색어 변경 시 디바운싱 적용"""
        # 기존 타이머 중지
        self.search_timer.stop()
        
        if len(text) >= 3:
            # 200ms 후 검색
            self.search_timer.start(200)
        else:
            # 1-2자 입력 중이면 결과만 지우기
            self.results_table.setRowCount(0)
            if hasattr(self, 'status_label'):
                self.status_label.setText("검색어를 더 입력하세요 (최소 3자)")
    
    def quick_search(self, ticker):
        """빠른 검색"""
        self.search_input.setText(ticker)
        self.perform_search()
    
    def show_popular_stocks(self):
        """인기 종목들 표시"""
        popular_search_terms = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', '005930.KS', '000660.KS']
        
        try:
            all_results = []
            for term in popular_search_terms:
                results = self.search_master_csv(term)
                if results:
                    all_results.append(results[0])  # 각 검색의 최고 결과만
            
            self.display_results(all_results)
            self.status_label.setText("💡 인기 종목들을 표시했습니다")
                
        except Exception as e:
            self.status_label.setText(f"⚠️ 인기 종목 로드 오류: {str(e)}")
            print(f"인기 종목 로드 오류: {e}")
    
    def perform_search(self):
        """마스터 CSV에서 검색 수행"""
        query = self.search_input.text().strip()
        
        if query in self.search_cache:
            print(f"💾 캐시 사용: {query}")
            self.display_results(self.search_cache[query])
            self.status_label.setText(f"✅ {len(self.search_cache[query])}개 종목 (캐시)")
            return

        if len(query) < 3:
            self.show_popular_stocks()
            return
        
        try:
            self.status_label.setText(f"'{query}' 검색 중...")
            self.results_table.setRowCount(0)
            QApplication.processEvents()
            
            # 마스터 CSV에서 검색
            results = self.search_stocks_with_api(query)
            self.display_results(results)
            
            if results:
                self.status_label.setText(f"🔍 {len(results)}개 종목 발견")
            else:
                self.status_label.setText("❌ 검색 결과가 없습니다")

            self.search_cache[query] = results
            self.display_results(results)

        except Exception as e:
            self.status_label.setText(f"❌ 검색 오류: {str(e)}")
            print(f"검색 오류: {e}")

    def search_stocks_with_api(self, search_term):
        """API를 사용한 실시간 주식 검색 + 기존 CSV 백업"""
        
        print(f"🔍 API로 '{search_term}' 검색 시작...")
        api_results = []
        
        # 1. 먼저 API로 검색 시도
        try:
            query = urllib.parse.quote(search_term)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            res = requests.get(url, headers=headers, timeout=10)
            print("Status code:", res.status_code)

            if res.ok:
                data = res.json()
                quotes = data.get('quotes', [])
                print(f"📊 API에서 {len(quotes)}개 종목 발견")
                
                # Make csv from json.
                api_results = self.convert_api_to_csv_format(quotes, search_term)

            else:
                print("Request failed:", res.text[:200])  # 에러일 경우 앞부분 출력           

        except Exception as e:
            print(f"API 검색 실패: {e}")
        
        # 2. CSV에서도 검색 (백업용)
        csv_results = self.search_master_csv(search_term)
        
        # 3. 결과 병합
        combined_results = self.merge_search_results(api_results, csv_results)
        
        print(f"✅ 총 {len(combined_results)}개 종목 반환")
        return combined_results

    def convert_api_to_csv_format(self, quotes, search_term):
        """Yahoo Finance API 응답을 기존 CSV 포맷으로 변환"""
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
                market_cap_str = self.format_market_cap(market_cap_raw)
                
                # 거래소 정보
                exchange = quote.get('exchDisp') or quote.get('exchange', 'Unknown')
                
                # 기존 CSV 포맷과 동일하게 구성
                stock_info = {
                    'ticker': ticker,
                    'name': name,
                    'sector': sector,
                    'market_cap': market_cap_str,
                    'market': exchange,
                    'raw_market_cap': market_cap_raw,
                    'match_score': 90 + self.calculate_relevance_bonus(quote, search_term),  # API는 높은 점수
                    'source': 'API'
                }
                
                csv_format_results.append(stock_info)
                
            except Exception as e:
                print(f"⚠️ API 데이터 변환 오류: {e}")
                continue
        
        return csv_format_results

    def format_market_cap(self, market_cap_value):
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

    def calculate_relevance_bonus(self, quote, search_term):
        """API 결과의 관련성 보너스 점수 계산"""
        bonus = 0
        
        # 정확한 타입인지 확인
        if quote.get('typeDisp') == 'Equity':
            bonus += 5
        
        # 검색어와 ticker 매칭도
        ticker = quote.get('symbol', '').upper()
        search_upper = search_term.upper()
        
        if ticker == search_upper:
            bonus += 10
        elif search_upper in ticker:
            bonus += 5
        
        return bonus

    def merge_search_results(self, api_results, csv_results):
        """API 결과와 CSV 결과를 병합하고 중복 제거"""
        combined = {}
        
        # API 결과 우선 추가 (높은 점수 부여)
        for stock in api_results:
            ticker = stock['ticker']
            combined[ticker] = stock
        
        # CSV 결과 추가 (이미 있는 ticker는 건너뛰기)
        for stock in csv_results:
            ticker = stock['ticker']
            if ticker not in combined:
                stock['source'] = 'CSV'
                combined[ticker] = stock
        
        # 매치 점수와 시가총액으로 정렬
        sorted_results = sorted(
            combined.values(), 
            key=lambda x: (-x['match_score'], -x.get('raw_market_cap', 0))
        )
        
        return sorted_results

    def search_stocks_enhanced(self):
        """향상된 검색 - 결과 저장 기능 추가"""
        query = self.search_input.text().strip()
        if len(query) < 1:
            self.show_popular_stocks()
            return
        
        try:
            self.status_label.setText(f"'{query}' 검색 중... (API + CSV)")
            QApplication.processEvents()
            
            # 향상된 검색 함수 사용
            results = self.search_stocks_with_api(query)
            
            # 결과 저장
            self.last_search_results = results
            
            self.display_results(results)
            
            if results:
                api_count = len([r for r in results if r.get('source') == 'API'])
                csv_count = len([r for r in results if r.get('source') == 'CSV'])
                self.status_label.setText(
                    f"🔍 {len(results)}개 종목 발견 (API: {api_count}, CSV: {csv_count}) - 매치점수순"
                )
                
                # CSV 포맷으로도 출력 (콘솔에)
                self.print_results_as_csv(results[:10])  # 상위 10개만
                
                # CSV 내보내기 버튼 활성화
                if hasattr(self, 'csv_export_btn'):
                    self.csv_export_btn.setEnabled(True)
            else:
                self.status_label.setText("❌ 검색 결과가 없습니다")
                if hasattr(self, 'csv_export_btn'):
                    self.csv_export_btn.setEnabled(False)
                
        except Exception as e:
            self.status_label.setText(f"❌ 검색 오류: {str(e)}")
            print(f"검색 오류: {e}")
            if hasattr(self, 'csv_export_btn'):
                self.csv_export_btn.setEnabled(False)

    def print_results_as_csv(self, results):
        """검색 결과를 CSV 포맷으로 콘솔에 출력"""
        print("\n" + "="*80)
        print(f"검색 결과 (상위 {len(results)}개) - CSV 포맷:")
        print("="*80)
        
        # CSV 헤더
        print("ticker,name,sector,market_cap,market,source,match_score")
        
        # 데이터 행들
        for stock in results:
            ticker = stock.get('ticker', '')
            name = stock.get('name', '').replace(',', ';')  # 쉼표를 세미콜론으로 변경
            sector = stock.get('sector', '').replace(',', ';')
            market_cap = stock.get('market_cap', 'N/A')
            market = stock.get('market', '')
            source = stock.get('source', 'CSV')
            match_score = stock.get('match_score', 0)
            
            print(f"{ticker},{name},{sector},{market_cap},{market},{source},{match_score}")
        
        print("="*80)

    def search_master_csv(self, search_term):
        """마스터 CSV 파일들에서 검색"""
        import os
        import pandas as pd
        
        found_stocks = []
        seen_tickers = set()
        search_term_upper = search_term.strip().upper()
        
        # 두 가지 가능한 위치 확인
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
        
        for file_path in master_files:
            if not os.path.exists(file_path):
                continue
                
            try:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                for _, row in df.iterrows():
                    ticker = str(row.get('ticker', '')).strip()
                    name = str(row.get('name', '')).strip()
                    sector = str(row.get('sector', '')).strip()
                    market = str(row.get('market', '')).strip()
                    market_cap = row.get('market_cap', 0)
                    
                    if not ticker or ticker in seen_tickers:
                        continue
                    
                    # 매칭 로직
                    match_score = 0
                    if ticker.upper() == search_term_upper:
                        match_score = 100
                    elif search_term_upper in ticker.upper():
                        match_score = 80
                    elif search_term_upper in name.upper():
                        match_score = 70
                    elif search_term_upper in sector.upper():
                        match_score = 50
                    
                    if match_score > 0:
                        # 시가총액 포맷팅
                        market_cap_str = "N/A"
                        if pd.notna(market_cap) and market_cap > 0:
                            if market_cap >= 1e12:
                                market_cap_str = f"{market_cap/1e12:.1f}T"
                            elif market_cap >= 1e9:
                                market_cap_str = f"{market_cap/1e9:.1f}B"
                            elif market_cap >= 1e6:
                                market_cap_str = f"{market_cap/1e6:.1f}M"
                            else:
                                market_cap_str = f"{market_cap:,.0f}"
                        
                        stock_info = {
                            'ticker': ticker,
                            'name': name,
                            'sector': sector,
                            'market_cap': market_cap_str,
                            'market': market,
                            'match_score': match_score,
                            'raw_market_cap': market_cap
                        }
                        found_stocks.append(stock_info)
                        seen_tickers.add(ticker)
                        
            except Exception as e:
                print(f"⚠️ {file_path} 읽기 오류: {e}")
                continue
        
        # 매치 점수와 시가총액 기준으로 정렬
        found_stocks.sort(key=lambda x: (-x['match_score'], -x.get('raw_market_cap', 0)))
        return found_stocks
    
    # def display_results(self, results):
    #     """검색 결과 표시"""
    #     self.results_table.setRowCount(len(results))
        
    #     for i, stock in enumerate(results):
    #         self.results_table.setItem(i, 0, QTableWidgetItem(stock.get('ticker', '')))
    #         self.results_table.setItem(i, 1, QTableWidgetItem(stock.get('name', '')))
    #         self.results_table.setItem(i, 2, QTableWidgetItem(stock.get('market', '')))
    #         self.results_table.setItem(i, 3, QTableWidgetItem(stock.get('sector', '')))
    #         self.results_table.setItem(i, 4, QTableWidgetItem(stock.get('market_cap', 'N/A')))
            
    #         # 매치점수 표시
    #         match_score = stock.get('match_score', 0)
    #         score_item = QTableWidgetItem(str(match_score))
            
    #         # 매치점수에 따른 색상 구분
    #         if match_score >= 90:
    #             score_item.setBackground(QColor(76, 175, 80, 100))  # 초록
    #         elif match_score >= 70:
    #             score_item.setBackground(QColor(255, 193, 7, 100))  # 노랑
    #         elif match_score >= 50:
    #             score_item.setBackground(QColor(255, 87, 34, 100))  # 주황
                
    #         self.results_table.setItem(i, 5, score_item)
        
    #     # 첫 번째 행 선택
    #     if len(results) > 0:
    #         self.results_table.selectRow(0)
    
    def display_results(self, results):
        """검색 결과 표시 - source 컬럼 추가"""
        self.results_table.setRowCount(len(results))
        
        # 컬럼 개수를 늘려서 source 정보도 표시
        if self.results_table.columnCount() < 6:
            self.results_table.setColumnCount(6)
            self.results_table.setHorizontalHeaderLabels([
                "종목코드", "회사명", "섹터", "시가총액", "거래소", "출처"
            ])
        
        for i, stock in enumerate(results):
            # 기존 컬럼들
            self.results_table.setItem(i, 0, QTableWidgetItem(stock.get('ticker', '')))
            self.results_table.setItem(i, 1, QTableWidgetItem(stock.get('name', '')))
            self.results_table.setItem(i, 2, QTableWidgetItem(stock.get('sector', '')))
            self.results_table.setItem(i, 3, QTableWidgetItem(stock.get('market_cap', '')))
            self.results_table.setItem(i, 4, QTableWidgetItem(stock.get('market', '')))
            
            # 새로운 출처 컬럼
            source = stock.get('source', 'CSV')
            source_item = QTableWidgetItem(source)
            
            # API 결과는 다른 색으로 표시
            if source == 'API':
                source_item.setBackground(QColor(200, 255, 200))  # 연한 초록색
                source_item.setToolTip("Yahoo Finance API에서 실시간 검색된 결과")
            else:
                source_item.setBackground(QColor(255, 255, 200))  # 연한 노란색
                source_item.setToolTip("로컬 마스터 CSV 파일에서 검색된 결과")
            
            self.results_table.setItem(i, 5, source_item)
        
        # 테이블 컬럼 크기 자동 조정
        self.results_table.resizeColumnsToContents()

    def show_csv_export_dialog(self):
        """검색 결과를 CSV 형태로 보여주는 다이얼로그"""
        if not hasattr(self, 'last_search_results') or not self.last_search_results:
            QMessageBox.information(self, "CSV 내보내기", "먼저 검색을 수행해주세요.")
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("검색 결과 - CSV 포맷")
        dialog.resize(800, 500)
        
        layout = QVBoxLayout()
        
        # 정보 레이블
        info_label = QLabel(f"총 {len(self.last_search_results)}개 종목 - CSV 포맷")
        info_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(info_label)
        
        # CSV 텍스트 영역
        text_edit = QTextEdit()
        csv_content = self.generate_csv_content(self.last_search_results)
        text_edit.setPlainText(csv_content)
        text_edit.setReadOnly(True)
        text_edit.setFont(QFont("Courier", 9))  # 고정폭 글꼴
        layout.addWidget(text_edit)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        copy_btn = QPushButton("클립보드 복사")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(csv_content))
        button_layout.addWidget(copy_btn)
        
        save_btn = QPushButton("파일 저장")
        save_btn.clicked.connect(lambda: self.save_csv_file(csv_content))
        button_layout.addWidget(save_btn)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        dialog.setLayout(layout)
        dialog.exec_()

    def generate_csv_content(self, results):
        """검색 결과를 CSV 문자열로 생성"""
        lines = ["ticker,name,sector,market_cap,market,source,match_score"]
        
        for stock in results:
            # CSV에서 쉼표나 특수문자 처리
            ticker = self.clean_csv_value(stock.get('ticker', ''))
            name = self.clean_csv_value(stock.get('name', ''))
            sector = self.clean_csv_value(stock.get('sector', ''))
            market_cap = self.clean_csv_value(stock.get('market_cap', 'N/A'))
            market = self.clean_csv_value(stock.get('market', ''))
            source = self.clean_csv_value(stock.get('source', 'CSV'))
            match_score = stock.get('match_score', 0)
            
            line = f"{ticker},{name},{sector},{market_cap},{market},{source},{match_score}"
            lines.append(line)
        
        return "\n".join(lines)

    def clean_csv_value(self, value):
        """CSV 값에서 특수문자 처리"""
        if not isinstance(value, str):
            value = str(value)
        
        # 쉼표나 따옴표가 있으면 따옴표로 감싸고 내부 따옴표는 이스케이프
        if ',' in value or '"' in value or '\n' in value:
            value = value.replace('"', '""')  # 따옴표 이스케이프
            return f'"{value}"'
        
        return value

    def save_csv_file(self, csv_content):
        """CSV 내용을 파일로 저장"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"stock_search_results_{timestamp}.csv"
            
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "CSV 파일 저장", 
                default_filename,
                "CSV 파일 (*.csv);;모든 파일 (*)"
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8-sig') as f:
                    f.write(csv_content)
                
                QMessageBox.information(self, "저장 완료", f"파일이 저장되었습니다:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"파일 저장 중 오류가 발생했습니다:\n{str(e)}")


    def refresh_search(self):
        """검색 새로고침"""
        self.status_label.setText("🔄 마스터 CSV 새로고침 중...")
        QApplication.processEvents()
        
        try:
            # 현재 검색어로 다시 검색
            current_query = self.search_input.text().strip()
            if current_query:
                self.perform_search()
            else:
                self.show_popular_stocks()
        except Exception as e:
            self.status_label.setText(f"❌ 새로고침 오류: {str(e)}")
    
    def select_from_table(self):
        """테이블에서 더블클릭으로 선택"""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            ticker_item = self.results_table.item(current_row, 0)
            if ticker_item:
                self.selected_ticker = ticker_item.text()
                self.accept()
    
    def select_ticker(self):
        """선택 버튼으로 선택"""
        current_row = self.results_table.currentRow()
        if current_row >= 0:
            ticker_item = self.results_table.item(current_row, 0)
            if ticker_item:
                self.selected_ticker = ticker_item.text()
                self.accept()
        else:
            QMessageBox.warning(self, "선택 오류", "종목을 선택해주세요.")
    
    def get_selected_ticker(self):
        return self.selected_ticker

class PredictionChartDialog(QDialog):
    """예측 차트 전용 다이얼로그"""
    
    def __init__(self, prediction_result, parent=None):
        super().__init__(parent)
        self.result = prediction_result
        self.initUI()
        self.create_chart()
    
    def initUI(self):
        self.setWindowTitle(f'📈 {self.result["ticker"]} 예측 차트')
        self.setGeometry(300, 200, 1000, 700)
        
        layout = QVBoxLayout()
        
        # 차트 위젯
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # 하단 버튼
        button_layout = QHBoxLayout()
        
        # 차트 저장 버튼
        save_btn = QPushButton('💾 차트 저장')
        save_btn.clicked.connect(self.save_chart)
        button_layout.addWidget(save_btn)
        
        # 차트 설정 버튼
        settings_btn = QPushButton('⚙️ 차트 설정')
        settings_btn.clicked.connect(self.show_chart_settings)
        button_layout.addWidget(settings_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton('닫기')
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def create_chart(self):
        """대형 예측 차트 생성"""
        # 위의 plot_prediction_timeseries 함수와 동일한 로직이지만
        # 더 큰 화면에 최적화
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # ... (plot_prediction_timeseries와 동일한 차트 생성 로직)
        # 단, 더 크고 상세한 차트로 구성
        
        forecast_days = self.result['days']
        ticker = self.result['ticker']
        
        try:
            # 과거 데이터 더 많이 표시 (60일)
            import yfinance as yf
            from datetime import datetime, timedelta
            
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            historical_data = stock.history(start=start_date, end=end_date)
            
            if len(historical_data) > 0:
                # 과거 60일 표시
                historical_dates = historical_data.index[-60:]
                historical_prices = historical_data['Close'].iloc[-60:].values
                
                # 미래 예측 차트 (더 상세하게)
                import pandas as pd
                last_date = historical_dates[-1]
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), 
                                             periods=forecast_days)
                
                # 더 자연스러운 예측 곡선 생성
                current_price = self.result['current_price']
                target_price = self.result['predicted_price']
                
                predicted_prices = []
                for i in range(forecast_days):
                    progress = (i + 1) / forecast_days
                    # 3차 베지어 곡선으로 부드러운 변화
                    smooth_progress = 3 * progress**2 - 2 * progress**3
                    predicted_price = current_price + (target_price - current_price) * smooth_progress
                    predicted_prices.append(predicted_price)
                
                predicted_prices = np.array(predicted_prices)
                
                # 고급 차트 스타일
                ax.plot(historical_dates, historical_prices, 'b-', 
                       label='과거 실제 주가', linewidth=2.5, alpha=0.9)
                
                ax.plot(future_dates, predicted_prices, 'r-', 
                       label='AI 예측 주가', linewidth=3, alpha=0.9)
                
                # 더 자세한 꾸미기...
                
        except Exception as e:
            # 기본 차트 표시
            days = list(range(forecast_days + 1))
            prices = [self.result['current_price']] + \
                    [self.result['predicted_price']] * forecast_days
            ax.plot(days, prices, 'r--', linewidth=2, marker='o')
        
        ax.set_title(f"{ticker} AI 주가 예측 상세 차트", fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self.canvas.draw()
    
    def save_chart(self):
        """차트 이미지로 저장"""
        from datetime import datetime
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", 
            f"{self.result['ticker']}_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.png",
            "PNG files (*.png);;All files (*.*)"
        )
        
        if filename:
            try:
                self.figure.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "저장 완료", f"차트가 저장되었습니다:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "저장 실패", f"차트 저장 중 오류:\n{str(e)}")
    
    def show_chart_settings(self):
        """차트 설정 다이얼로그"""
        QMessageBox.information(self, "차트 설정", 
                              "차트 설정 기능은 향후 업데이트에서 제공될 예정입니다.")
