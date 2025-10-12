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

# 최적화 모듈 import
from cache_manager import get_stock_data, get_ticker_info
from unified_search import search_stocks
from matplotlib_optimizer import safe_figure, ChartManager
from utils import format_market_cap_value
from logger_config import get_logger

logger = get_logger(__name__)

# ✅ 예측 작업을 위한 Worker Thread
class PredictionWorker(QThread):
    """백그라운드에서 예측을 실행하는 워커 스레드"""
    finished = pyqtSignal(object, object)  # (result, error)
    progress = pyqtSignal(str, int)  # (message, percent)

    def __init__(self, predictor, ticker, forecast_days):
        super().__init__()
        self.predictor = predictor
        self.ticker = ticker
        self.forecast_days = forecast_days

    def run(self):
        """백그라운드에서 실행"""
        try:
            # 진행 콜백 설정
            def progress_callback(step, message):
                self.progress.emit(message, self.get_progress_percent(step))

            if hasattr(self.predictor, 'set_progress_callback'):
                self.predictor.set_progress_callback(progress_callback)

            # 예측 실행
            result = self.predictor.predict_stock_price(
                self.ticker,
                forecast_days=self.forecast_days,
                show_plot=False
            )

            self.finished.emit(result, None)

        except Exception as e:
            logger.error(f"예측 워커 오류: {e}")
            self.finished.emit(None, str(e))

    def get_progress_percent(self, step):
        """단계를 퍼센트로 변환"""
        progress_map = {
            'data': 20,
            'market_analysis': 30,
            'kalman': 40,
            'ml': 55,  # ML 모델은 시간이 오래 걸림
            'arima': 70,
            'lstm': 80,
            'transformer': 85,
            'ensemble': 92,
            'complete': 100
        }
        return progress_map.get(step, 50)

# ✅ 백테스팅 작업을 위한 Worker Thread
class BacktestWorker(QThread):
    """백그라운드에서 백테스팅을 실행하는 워커 스레드"""
    finished = pyqtSignal(object, object)  # (summary, error)
    progress = pyqtSignal(int, int, str)  # (current, total, message)

    def __init__(self, predictor, ticker, test_periods, forecast_days, use_parallel):
        super().__init__()
        self.predictor = predictor
        self.ticker = ticker
        self.test_periods = test_periods
        self.forecast_days = forecast_days
        self.use_parallel = use_parallel
        self.cancelled = False

    def run(self):
        """백그라운드에서 백테스팅 실행"""
        try:
            # 진행 콜백 설정
            def progress_callback(current, total, message):
                if not self.cancelled:
                    self.progress.emit(current, total, message)

            # 중지 콜백
            def cancel_callback():
                return self.cancelled

            summary, error = self.predictor.backtest_predictions(
                self.ticker,
                test_periods=self.test_periods,
                forecast_days=self.forecast_days,
                progress_callback=progress_callback,
                use_parallel=self.use_parallel,
                cancel_callback=cancel_callback
            )

            self.finished.emit(summary, error)

        except Exception as e:
            logger.error(f"백테스팅 워커 오류: {e}")
            self.finished.emit(None, str(e))

    def cancel(self):
        """백테스팅 중지"""
        self.cancelled = True

# Enhanced Screener의 예측기 import
try:
    from enhanced_screener import EnhancedCPUPredictor
    ENHANCED_AVAILABLE = True
    logger.info("Enhanced Screener 예측기 사용 가능")
except ImportError as e:
    logger.warning(f"Enhanced Screener 없음: {e}")
    ENHANCED_AVAILABLE = False

# 새로운 딥러닝 예측기 import
try:
    from stock_prediction import StockPredictor
    DEEP_LEARNING_AVAILABLE = True
    logger.info("딥러닝 예측기 사용 가능")
except ImportError as e:
    logger.warning(f"딥러닝 예측기 없음: {e}")
    DEEP_LEARNING_AVAILABLE = False

ML_AVAILABLE = ENHANCED_AVAILABLE or DEEP_LEARNING_AVAILABLE

# 기본 라이브러리 확인
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
    import xgboost as xgb
    import lightgbm as lgb
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger.info("""Prediction Window 업데이트:
• Enhanced Screener 통합 완료
• 중복 예측 함수 제거
• 일관성 있는 예측 결과
• 더 나은 성능과 정확도""")


class StockPredictionDialog(QDialog):
    """주식 예측 다이얼로그 - Enhanced Screener 통합 버전"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 예측기 초기화 (딥러닝 우선, Enhanced 대체)
        self.predictor = None
        self.predictor_type = "None"

        # 딥러닝 예측기 우선 사용
        if DEEP_LEARNING_AVAILABLE:
            self.predictor = StockPredictor(
                use_deep_learning=True,      # LSTM, Transformer 사용
                use_optimization=False        # Bayesian Opt (시간이 오래 걸리므로 기본 False)
            )
            self.predictor_type = "DeepLearning"
            logger.info("딥러닝 예측기 활성화 (LSTM + Transformer)")
        elif ENHANCED_AVAILABLE:
            self.predictor = EnhancedCPUPredictor()
            self.predictor_type = "Enhanced"
            logger.info("Enhanced CPU 예측기 활성화")

        # 딥러닝/최적화 옵션
        self.use_deep_learning = True
        self.use_optimization = False
        
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

        # 백테스팅 중지 플래그
        self.backtest_cancelled = False

        self.load_current_settings()
        
        self.initUI()
        
    def initUI(self):
        # 예측기 타입에 따라 제목 변경
        if self.predictor_type == "DeepLearning":
            title = '🧠 AI 주식 예측 (DeepLearning + LSTM + Transformer)'
        else:
            title = '🚀 AI 주식 예측 (Enhanced)'

        self.setWindowTitle(title)
        self.setGeometry(200, 50, 1000, 960)  # 크기 증가 (800x600 -> 1000x700)

        layout = QVBoxLayout()
        layout.setSpacing(8)  # 전체 레이아웃 간격 조정
        
        # 상단 입력 패널
        input_panel = self.create_input_panel()
        layout.addWidget(input_panel)

        # ✅ 진행 상태 표시 (프로그레스 바 + 상태 메시지)
        progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 5, 0, 5)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.status_label.setVisible(False)
        progress_layout.addWidget(self.status_label)

        progress_widget.setLayout(progress_layout)
        layout.addWidget(progress_widget)

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
            'backtest_periods': 30,  # 백테스팅 횟수 기본값
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
                logger.info(f"Prediction Window 설정 로드: {saved_settings.get('forecast_days', 7)}일 예측")
        except Exception as e:
            logger.warning(f"Prediction Window 설정 로드 실패: {e}")

    def create_enhanced_button_layout(self):
        """향상된 버튼 레이아웃 - 예측 차트 버튼 추가"""
        button_layout = QHBoxLayout()
        
        # 기존 예측 시작 버튼
        self.predict_btn = QPushButton('🚀 AI 예측 시작')
        self.predict_btn.clicked.connect(self.start_prediction_enhanced)  # 새로운 함수 연결
        button_layout.addWidget(self.predict_btn)

        # 딥러닝 모델 훈련 버튼 (LSTM/Transformer 저장 .h5)
        self.train_dl_btn = QPushButton('딥러닝 모델 훈련')
        self.train_dl_btn.setToolTip('현재 종목에 대해 LSTM/Transformer를 학습하고 .h5로 저장합니다')
        self.train_dl_btn.clicked.connect(self.train_deep_models)
        self.train_dl_btn.setEnabled(DEEP_LEARNING_AVAILABLE)
        button_layout.addWidget(self.train_dl_btn)

        # 백테스팅 버튼 추가
        self.backtest_btn = QPushButton('🔬 백테스팅')
        self.backtest_btn.setToolTip('과거 데이터로 예측 알고리즘 검증')
        self.backtest_btn.clicked.connect(self.run_backtest)
        button_layout.addWidget(self.backtest_btn)

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
        """입력 패널 생성 - 컴팩트한 레이아웃"""
        panel = QGroupBox("🎯 예측 설정")
        layout = QGridLayout()

        # 간격 조정 - 세로 간격을 더욱 줄임
        layout.setVerticalSpacing(4)  # 세로 간격 줄이기 (8 -> 4)
        layout.setHorizontalSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)  # 패널 내부 여백 줄임

        # === Row 0: 종목 코드 ===
        layout.addWidget(QLabel("종목 코드:"), 0, 0)

        ticker_layout = QHBoxLayout()
        ticker_layout.setSpacing(2)

        self.ticker_input = QLineEdit("AAPL")
        self.ticker_input.setPlaceholderText("예: AAPL, MSFT, 005930.KS, 삼성")
        self.ticker_input.setMaximumWidth(200)  # 가로 길이 제한
        ticker_layout.addWidget(self.ticker_input)

        self.search_btn = QPushButton("🔍 검색")
        self.search_btn.setToolTip("종목 검색")
        self.search_btn.setMinimumWidth(70)  # 검색 버튼 크기 증가
        self.search_btn.clicked.connect(self.show_enhanced_stock_search_dialog)
        ticker_layout.addWidget(self.search_btn)

        ticker_layout.addStretch()  # 남은 공간을 오른쪽으로

        self.ticker_input.textChanged.connect(self.on_ticker_text_changed)

        ticker_widget = QWidget()
        ticker_widget.setLayout(ticker_layout)
        layout.addWidget(ticker_widget, 0, 1)

        # === Row 1: 예측 기간 + 설정 동기화 ===
        layout.addWidget(QLabel("예측 기간:"), 1, 0)

        days_layout = QHBoxLayout()
        days_layout.setSpacing(2)

        self.days_input = QSpinBox()
        self.days_input.setRange(1, 30)
        self.days_input.setValue(self.current_settings.get('forecast_days', 7))
        self.days_input.setSuffix(" 일")
        self.days_input.setMaximumWidth(80)
        days_layout.addWidget(self.days_input)

        self.sync_settings_btn = QPushButton("⚙️")
        self.sync_settings_btn.setToolTip("설정 파일과 동기화")
        self.sync_settings_btn.setMaximumWidth(45)
        self.sync_settings_btn.clicked.connect(self.sync_with_settings)
        days_layout.addWidget(self.sync_settings_btn)

        days_layout.addStretch()

        days_widget = QWidget()
        days_widget.setLayout(days_layout)
        layout.addWidget(days_widget, 1, 1)

        # === Row 2: 딥러닝 설정 (좌우로 배치) ===
        layout.addWidget(QLabel("🧠 AI 모델:"), 2, 0)

        ai_layout = QHBoxLayout()
        ai_layout.setSpacing(5)

        self.deep_learning_checkbox = QCheckBox("딥러닝 (LSTM+Transformer)")
        self.deep_learning_checkbox.setChecked(self.use_deep_learning and DEEP_LEARNING_AVAILABLE)
        self.deep_learning_checkbox.setEnabled(DEEP_LEARNING_AVAILABLE)
        self.deep_learning_checkbox.setToolTip("LSTM과 Transformer 사용 (정확도↑, 시간↑)")
        self.deep_learning_checkbox.stateChanged.connect(self.on_deep_learning_changed)
        ai_layout.addWidget(self.deep_learning_checkbox)

        # 강제 재학습 옵션
        self.force_retrain = False
        self.force_retrain_checkbox = QCheckBox("강제 재학습")
        self.force_retrain_checkbox.setToolTip("기존 저장 모델이 있어도 다시 학습합니다")
        self.force_retrain_checkbox.stateChanged.connect(lambda s: setattr(self, 'force_retrain', s == 2))
        self.force_retrain_checkbox.setEnabled(DEEP_LEARNING_AVAILABLE)
        ai_layout.addWidget(self.force_retrain_checkbox)

        # 훈련 기간 선택
        ai_layout.addWidget(QLabel("훈련 기간:"))
        self.train_period_combo = QComboBox()
        self.train_period_combo.addItem("자동", "auto")
        self.train_period_combo.addItem("2y", "2y")
        self.train_period_combo.addItem("3y", "3y")
        self.train_period_combo.addItem("5y", "5y")
        self.train_period_combo.addItem("10y", "10y")
        self.train_period_combo.addItem("max", "max")
        try:
            idx = self.train_period_combo.findData("5y")
            if idx >= 0:
                self.train_period_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self.train_period_combo.setEnabled(DEEP_LEARNING_AVAILABLE)
        ai_layout.addWidget(self.train_period_combo)

        # 강제 재학습 옵션
        self.force_retrain = False
        self.force_retrain_checkbox = QCheckBox("강제 재학습")
        self.force_retrain_checkbox.setToolTip("기존 저장 모델이 있어도 다시 학습합니다")
        self.force_retrain_checkbox.stateChanged.connect(lambda s: setattr(self, 'force_retrain', s == 2))
        self.force_retrain_checkbox.setEnabled(DEEP_LEARNING_AVAILABLE)
        ai_layout.addWidget(self.force_retrain_checkbox)

        self.optimization_checkbox = QCheckBox("Bayesian 최적화")
        self.optimization_checkbox.setChecked(self.use_optimization)
        self.optimization_checkbox.setToolTip("하이퍼파라미터 자동 조정 (정확도↑↑, 시간↑↑)")
        self.optimization_checkbox.stateChanged.connect(self.on_optimization_changed)
        ai_layout.addWidget(self.optimization_checkbox)

        ai_layout.addStretch()

        ai_widget = QWidget()
        ai_widget.setLayout(ai_layout)
        layout.addWidget(ai_widget, 2, 1)

        # === Row 3: 백테스팅 설정 ===
        layout.addWidget(QLabel("백테스팅:"), 3, 0)

        backtest_layout = QHBoxLayout()
        backtest_layout.setSpacing(2)

        self.backtest_periods_input = QSpinBox()
        self.backtest_periods_input.setRange(5, 100)
        self.backtest_periods_input.setValue(self.current_settings.get('backtest_periods', 30))
        self.backtest_periods_input.setSuffix(" 회")
        self.backtest_periods_input.setMaximumWidth(80)
        self.backtest_periods_input.setToolTip("테스트 횟수 (많을수록 정확, 느림)")
        self.backtest_periods_input.setKeyboardTracking(True)  # 키보드 입력 즉시 반영
        self.backtest_periods_input.setWrapping(False)  # 순환 방지
        self.backtest_periods_input.setFocusPolicy(Qt.StrongFocus)  # 포커스 강화
        backtest_layout.addWidget(self.backtest_periods_input)

        self.parallel_backtest_checkbox = QCheckBox("병렬")
        self.parallel_backtest_checkbox.setChecked(False)  # 기본값: 순차 처리 (안정적)
        self.parallel_backtest_checkbox.setToolTip("병렬 처리 (100회 이상 백테스팅 시 권장)")
        backtest_layout.addWidget(self.parallel_backtest_checkbox)

        backtest_layout.addStretch()

        backtest_widget = QWidget()
        backtest_widget.setLayout(backtest_layout)
        layout.addWidget(backtest_widget, 3, 1)

        panel.setLayout(layout)
        return panel

    def on_deep_learning_changed(self, state):
        """딥러닝 옵션 변경 핸들러"""
        self.use_deep_learning = (state == 2)  # Qt.Checked = 2

        # 예측기 재생성
        if DEEP_LEARNING_AVAILABLE:
            self.predictor = StockPredictor(
                use_deep_learning=self.use_deep_learning,
                use_optimization=self.use_optimization
            )
            self.predictor_type = "DeepLearning"
            logger.info(f"딥러닝 모델: {'활성화' if self.use_deep_learning else '비활성화'}")
            if hasattr(self, 'train_dl_btn'):
                self.train_dl_btn.setEnabled(self.use_deep_learning)

    def on_optimization_changed(self, state):
        """하이퍼파라미터 최적화 옵션 변경 핸들러"""
        self.use_optimization = (state == 2)

        # 예측기 재생성
        if DEEP_LEARNING_AVAILABLE:
            self.predictor = StockPredictor(
                use_deep_learning=self.use_deep_learning,
                use_optimization=self.use_optimization
            )
            logger.info(f"Bayesian Optimization: {'활성화' if self.use_optimization else '비활성화'}")

    def train_deep_models(self):
        """현재 종목에 대해 LSTM/Transformer를 학습하고 저장(.h5)"""
        if not DEEP_LEARNING_AVAILABLE:
            QMessageBox.warning(self, '딥러닝 사용 불가', 'TensorFlow/딥러닝 환경이 활성화되어 있지 않습니다.')
            return

        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, '입력 필요', '종목 코드를 입력하세요 (예: AAPL, 005930.KS).')
            return

        # 학습 기간 결정
        try:
            from optimal_period_config import get_optimal_training_period
            period = get_optimal_training_period(ticker)
            logger.info(f"학습 기간(자동): {period}")
        except Exception:
            period = '5y'
            logger.debug("학습 기간 기본값 5y 사용")

        # 데이터 로드
        try:
            # 사용자 선택 기간이 있으면 우선 적용
            try:
                if hasattr(self, 'train_period_combo') and self.train_period_combo is not None:
                    sel = self.train_period_combo.currentData()
                    if sel not in (None, 'auto'):
                        period = sel
            except Exception:
                pass

            df = get_stock_data(ticker, period=period)
            if df is None or len(df) < 100:
                QMessageBox.warning(self, '데이터 부족', '학습에 충분한 데이터가 없습니다. 기간을 늘려보세요.')
                return
            prices = df['Close'].values
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            QMessageBox.critical(self, '오류', f"데이터 로드 실패: {e}")
            return

        forecast_days = self.days_input.value()

        # 진행 안내
        self.result_area.append(f"\n[딥러닝 훈련] {ticker} ({period}), 재학습={'ON' if self.force_retrain else 'OFF'}")
        QApplication.processEvents()

        trained_any = False
        errors = []

        try:
            # LSTM 훈련
            from stock_prediction import LSTMPredictor
            lstm = LSTMPredictor(ticker=ticker, auto_load=True)
            lstm_result = lstm.fit_predict(prices, forecast_days=forecast_days, force_retrain=self.force_retrain)
            if 'error' in lstm_result:
                errors.append(f"LSTM: {lstm_result['error']}")
            else:
                trained_any = True
                self.result_area.append(f" - LSTM: 학습 완료 (val_loss={lstm_result.get('val_loss','N/A')})")
        except Exception as e:
            errors.append(f"LSTM 오류: {e}")

        try:
            # Transformer 훈련
            from stock_prediction import TransformerPredictor
            tr = TransformerPredictor(ticker=ticker, auto_load=True)
            tr_result = tr.fit_predict(prices, forecast_days=forecast_days, force_retrain=self.force_retrain)
            if 'error' in tr_result:
                errors.append(f"Transformer: {tr_result['error']}")
            else:
                trained_any = True
                self.result_area.append(f" - Transformer: 학습 완료 (val_loss={tr_result.get('val_loss','N/A')})")
        except Exception as e:
            errors.append(f"Transformer 오류: {e}")

        if trained_any:
            QMessageBox.information(self, '훈련 완료', f"모델이 저장되었습니다. models/{ticker} 폴더를 확인하세요.")
        else:
            QMessageBox.warning(self, '훈련 실패', "\n".join(errors) if errors else '훈련에 실패했습니다.')

        # 상태 갱신
        self.chart_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        QApplication.processEvents()

    def sync_with_settings(self):
            """설정 파일과 동기화 - 간소화 버전"""
            self.load_current_settings()

            # UI 업데이트
            self.days_input.setValue(self.current_settings.get('forecast_days', 7))
            self.backtest_periods_input.setValue(self.current_settings.get('backtest_periods', 30))

            # 메시지 표시
            QMessageBox.information(self, "설정 동기화",
                                f"✅ 설정이 동기화되었습니다!\n\n"
                                f"• 예측 기간: {self.current_settings.get('forecast_days', 7)}일\n"
                                f"• 백테스팅: {self.current_settings.get('backtest_periods', 30)}회\n"
                                f"• 신뢰도 임계값: {self.current_settings.get('confidence_threshold', 0.6)*100:.0f}%")


    def show_enhanced_stock_search_dialog(self):
        """마스터 CSV를 활용한 종목 검색 다이얼로그 표시"""
        dialog = EnhancedStockSearchDialog(self)

        # ✅ 입력란에 이미 입력된 내용이 있으면 검색창에 미리 채우기
        current_text = self.ticker_input.text().strip()
        if current_text:
            dialog.search_input.setText(current_text)

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

    def show_progress(self, message, percent):
        """진행 상태 표시"""
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.progress_bar.setValue(percent)
        self.status_label.setText(f"🔄 {message}")

    def hide_progress(self):
        """진행 상태 숨기기"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)

    def on_prediction_finished_enhanced(self, result, error_msg):
        """Enhanced 예측 완료 처리 - 차트 버튼 활성화 추가"""
        self.predict_btn.setEnabled(True)
        self.hide_progress()  # ✅ 진행 상태 숨기기

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
        """실제 예측 실행 - Worker Thread 사용"""
        try:
            # ✅ 진행 상태 표시 시작
            self.show_progress("예측 준비 중...", 10)

            if self.predictor_type == "DeepLearning":
                # Worker Thread로 예측 실행
                logger.info(f"딥러닝 예측기 실행 (백그라운드): {ticker} ({forecast_days}일)")

                self.worker = PredictionWorker(self.predictor, ticker, forecast_days)
                self.worker.progress.connect(self.on_worker_progress)
                self.worker.finished.connect(self.on_worker_finished)
                self.worker.start()

            else:
                # Enhanced 예측기는 기존 방식 유지
                logger.info(f"Enhanced 예측기 실행: {ticker} ({forecast_days}일)")
                result, error = self.predictor.predict_stock(ticker, forecast_days=forecast_days)
                self.on_prediction_finished_enhanced(result, error)

        except Exception as e:
            logger.error(f"예측 오류: {e}")
            self.on_prediction_finished_enhanced(None, str(e))

    def on_worker_progress(self, message, percent):
        """워커 진행 상태 업데이트"""
        logger.info(f"진행률 업데이트: {percent}% - {message}")
        self.show_progress(message, percent)
        QApplication.processEvents()  # UI 즉시 업데이트

    def on_worker_finished(self, result, error):
        """워커 완료 처리"""
        if error:
            self.on_prediction_finished_enhanced(None, error)
        elif result and 'error' in result:
            self.on_prediction_finished_enhanced(None, result['error'])
        else:
            # 결과 형식 변환
            ticker = self.ticker_input.text().strip().upper()
            forecast_days = self.days_input.value()

            converted_result = {
                'ticker': ticker,
                'current_price': result['current_price'],
                'predicted_prices': result['predicted_prices'],
                'predicted_price': result['predicted_prices'][-1],
                'expected_return': result['expected_returns'][0] / 100,
                'expected_returns': result['expected_returns'],
                'future_dates': result['future_dates'],
                'days': forecast_days,
                'confidence': result['confidence_score'],
                'confidence_score': result['confidence_score'],
                'recommendation': result['recommendation'],
                'models_used': result.get('models_used', []),
                'model_weights': result.get('model_weights', {}),
                'market_regime': result.get('market_regime', 'unknown'),
                'predictor_type': 'DeepLearning',
                'is_high_confidence': result['confidence_score'] >= 0.6,
                'data_points': 'N/A',
                'training_samples': 'N/A',
                'market_correlations': result.get('market_correlations', {}),
                'sector_performance': result.get('sector_performance', {}),
                'institutional_flow': result.get('institutional_flow', {})
            }
            self.on_prediction_finished_enhanced(converted_result, None)

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
            logger.error(f"결과 변환 오류: {e}")
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
        """✅ 수정: 딥러닝 정보 포함 결과 표시"""
        # 결과 형식에 따라 다르게 처리
        if 'expected_return' in result:
            return_rate = result['expected_return']
            confidence = result['confidence']
        elif 'expected_returns' in result:
            return_rate = result['expected_returns'][0] / 100  # 첫 번째 예측일 수익률
            confidence = result['confidence_score']
        else:
            return_rate = 0
            confidence = 0.5

        # 예측기 타입 확인
        predictor_info = result.get('predictor_type', 'Enhanced')
        market_regime = result.get('market_regime', 'unknown')

        # ✅ 신뢰도 임계값 정보 가져오기
        confidence_threshold = result.get('confidence_threshold', 0.6)
        is_high_confidence = result.get('is_high_confidence', confidence >= confidence_threshold)
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

        # 예측 기간
        days = result.get('days', len(result.get('predicted_prices', [])))

        # 시장 상황 이모지
        regime_emoji = {'bull': '📈', 'bear': '📉', 'sideways': '↔️'}.get(market_regime, '❓')

        # 결과 텍스트 생성
        predictor_name = "🧠 DeepLearning AI" if predictor_info == "DeepLearning" else "🚀 Enhanced AI"

        # 예측 가격 처리 (단수/복수 형식 모두 지원)
        if 'predicted_price' in result:
            predicted_price = result['predicted_price']
        elif 'predicted_prices' in result:
            predicted_price = result['predicted_prices'][-1]  # 마지막 예측일 가격
        else:
            predicted_price = result['current_price']

        # 실제 수익률 계산 (예측 가격 기준)
        current_price = result['current_price']
        actual_return_rate = (predicted_price - current_price) / current_price

        text = f"""
══════════════════════════════════════════════════
🎯 {result['ticker']} {predictor_name} 예측 ({days}일 후)
══════════════════════════════════════════════════

💰 현재 가격: ${current_price:.2f}
🎯 예측 가격: ${predicted_price:.2f}
📊 예상 수익률: {actual_return_rate*100:+.2f}%

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
🔬 {predictor_name} 분석 정보:
──────────────────────────────────────────────────
• 예측기 타입: {predictor_info}
• 시장 상황: {regime_emoji} {market_regime.upper()}
• 성공한 모델: {result.get('successful_models', len(result.get('models_used', [])))}개
• 사용 모델: {', '.join(result.get('models_used', ['N/A']))}
• 사용된 특성: {result.get('feature_count', 30)}개 이상
• 데이터 기간: {result.get('data_points', 'N/A')}일
• 학습 샘플: {result.get('training_samples', 'N/A')}개
• 예측 완료: {result.get('prediction_date', datetime.now().strftime('%Y-%m-%d %H:%M'))}"""

        # 시장 상관관계 정보 추가
        market_corr = result.get('market_correlations', {})
        if market_corr:
            text += f"""

──────────────────────────────────────────────────
📊 시장 지수 상관관계:
──────────────────────────────────────────────────"""
            for index_name, corr_value in market_corr.items():
                corr_percent = corr_value * 100
                if abs(corr_value) > 0.7:
                    strength = "강한"
                    emoji = "🔴" if corr_value > 0 else "🔵"
                elif abs(corr_value) > 0.4:
                    strength = "보통"
                    emoji = "🟠" if corr_value > 0 else "🟦"
                else:
                    strength = "약한"
                    emoji = "⚪"

                text += f"\n• {index_name}: {emoji} {corr_percent:+.1f}% ({strength} {'양' if corr_value > 0 else '음'}의 상관)"

        # 섹터 성과 정보 추가 (미국 종목)
        sector_perf = result.get('sector_performance', {})
        if sector_perf:
            # 상위 3개 섹터만 표시
            sorted_sectors = sorted(sector_perf.items(), key=lambda x: x[1], reverse=True)[:3]
            text += f"""

──────────────────────────────────────────────────
🏭 섹터 성과 (최근 1개월, Top 3):
──────────────────────────────────────────────────"""
            for sector, perf in sorted_sectors:
                emoji = "📈" if perf > 0 else "📉"
                text += f"\n• {sector}: {emoji} {perf:+.2f}%"

        # 외국인/기관 매매 동향 (한국 종목)
        inst_flow = result.get('institutional_flow', {})
        if inst_flow:
            text += f"""

──────────────────────────────────────────────────
💰 외국인/기관 매매 동향 (최근 30일):
──────────────────────────────────────────────────"""

            foreign_net = inst_flow.get('foreign_net_buy', 0)
            institution_net = inst_flow.get('institution_net_buy', 0)

            foreign_emoji = "🟢" if foreign_net > 0 else "🔴" if foreign_net < 0 else "⚪"
            inst_emoji = "🟢" if institution_net > 0 else "🔴" if institution_net < 0 else "⚪"

            text += f"\n• 외국인: {foreign_emoji} {foreign_net:+,.0f}주 {'순매수' if foreign_net > 0 else '순매도' if foreign_net < 0 else '보합'}"
            text += f"\n• 기관: {inst_emoji} {institution_net:+,.0f}주 {'순매수' if institution_net > 0 else '순매도' if institution_net < 0 else '보합'}"

            if inst_flow.get('foreign_ownership'):
                text += f"\n• 외국인 지분율: {inst_flow['foreign_ownership']:.2f}%"

        text += """

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
            # 📊 1. 과거 데이터 가져오기 (최근 30일) - 캐싱 사용
            ticker = result['ticker']

            historical_data = get_stock_data(ticker, period="45d")

            if len(historical_data) == 0:
                logger.warning("과거 데이터 없음 - 단순 차트로 대체")
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

            # 예측 가격 처리 (단수/복수 형식 모두 지원)
            if 'predicted_price' in result:
                target_price = result['predicted_price']
                predicted_prices_array = None
            elif 'predicted_prices' in result:
                predicted_prices_array = result['predicted_prices']
                target_price = predicted_prices_array[-1]
            else:
                target_price = current_price
                predicted_prices_array = None
            
            # 실제 예측 가격 배열이 있으면 사용, 없으면 부드러운 곡선 생성
            if predicted_prices_array is not None and len(predicted_prices_array) == forecast_days:
                predicted_prices = np.array(predicted_prices_array)
            else:
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
                logger.warning(f"날짜 포맷팅 오류 (무시됨): {e}")
            
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
                logger.warning(f"정보 텍스트 박스 오류 (무시됨): {e}")
            
            # 레이아웃 조정
            plt.tight_layout()

        except Exception as e:
            logger.warning(f"고급 시계열 차트 생성 실패: {e}")
            logger.info("단순 차트로 대체합니다...")
            # 모든 오류에 대해 백업 차트 사용
            self.plot_prediction_simple(result)
            return

        # 캔버스 업데이트
        try:
            self.canvas.draw()
        except Exception as e:
            logger.warning(f"캔버스 그리기 오류: {e}")
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
            logger.error(f"단순 차트도 실패: {e}")
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
                logger.info(f"마커 '{marker}' 호환됨")
            except Exception as e:
                logger.error(f"마커 '{marker}' 호환되지 않음: {e}")
        
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


    def run_backtest(self):
        """백테스팅 실행"""
        ticker = self.ticker_input.text().strip().upper()
        days = self.days_input.value()
        # GUI에서 백테스팅 횟수 가져오기
        test_periods = self.backtest_periods_input.value()

        if not ticker:
            QMessageBox.warning(self, "오류", "종목 코드를 입력해주세요.")
            return

        # 병렬 처리 여부 확인
        use_parallel = self.parallel_backtest_checkbox.isChecked()

        reply = QMessageBox.question(
            self, "백테스팅",
            f"{ticker} 예측 알고리즘을 과거 데이터로 검증합니다.\n\n"
            f"• 예측 기간: {days}일\n"
            f"• 테스트 횟수: {test_periods}회\n"
            f"• 처리 방식: {'🚀 병렬 처리 (빠름)' if use_parallel else '⏳ 순차 처리'}\n\n"
            f"시간이 다소 걸릴 수 있습니다. 계속하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        # 중지 플래그 초기화
        self.backtest_cancelled = False

        # Progress bar 생성
        if not hasattr(self, 'backtest_progress_bar'):
            self.backtest_progress_bar = QProgressBar()
            self.backtest_progress_label = QLabel("")
            self.backtest_cancel_btn = QPushButton("⏹ 중지")
            self.backtest_cancel_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px; }")
            self.backtest_cancel_btn.clicked.connect(self.cancel_backtest)

            # 버튼 레이아웃 위에 progress bar 추가
            layout = self.layout()
            layout.insertWidget(layout.count() - 1, self.backtest_progress_bar)

            # Progress label과 중지 버튼을 가로로 배치
            progress_control_layout = QHBoxLayout()
            progress_control_layout.addWidget(self.backtest_progress_label)
            progress_control_layout.addWidget(self.backtest_cancel_btn)
            progress_control_widget = QWidget()
            progress_control_widget.setLayout(progress_control_layout)
            layout.insertWidget(layout.count() - 1, progress_control_widget)

        # Progress bar 표시
        self.backtest_progress_bar.setVisible(True)
        self.backtest_progress_label.setVisible(True)
        self.backtest_cancel_btn.setVisible(True)
        self.backtest_cancel_btn.setEnabled(True)
        self.backtest_progress_bar.setMaximum(test_periods)
        self.backtest_progress_bar.setValue(0)
        self.backtest_progress_label.setText("백테스팅 준비 중...")

        # UI 비활성화
        self.backtest_btn.setEnabled(False)
        self.result_area.setText("백테스팅 진행 중...\n")
        QApplication.processEvents()

        # 백테스팅 실행 (Worker Thread 사용)
        try:
            # 병렬 처리 옵션 가져오기
            use_parallel = self.parallel_backtest_checkbox.isChecked()

            # Worker 생성 및 시작
            self.backtest_worker = BacktestWorker(
                self.predictor,
                ticker,
                test_periods,
                days,
                use_parallel
            )

            # 시그널 연결
            self.backtest_worker.progress.connect(self.on_backtest_progress)
            self.backtest_worker.finished.connect(self.on_backtest_finished)

            # 워커 시작
            self.backtest_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "오류", f"백테스팅 시작 실패:\n{str(e)}")
            self.backtest_btn.setEnabled(True)
            self.backtest_progress_bar.setVisible(False)
            self.backtest_progress_label.setVisible(False)
            self.backtest_cancel_btn.setVisible(False)

    def cancel_backtest(self):
        """백테스팅 중지"""
        if hasattr(self, 'backtest_worker') and self.backtest_worker.isRunning():
            self.backtest_worker.cancel()
            self.backtest_cancel_btn.setEnabled(False)
            self.backtest_progress_label.setText("중지 중... 현재 작업 완료 대기")
            logger.info("백테스팅 중지 요청됨")

    def is_backtest_cancelled(self):
        """백테스팅 중지 여부 확인 (콜백용)"""
        return self.backtest_cancelled

    def on_backtest_progress(self, current, total, message):
        """백테스팅 진행률 업데이트 (Worker 시그널용)"""
        self.backtest_progress_bar.setValue(current)
        self.backtest_progress_label.setText(f"{message} - {current}/{total}")
        QApplication.processEvents()

    def on_backtest_finished(self, summary, error):
        """백테스팅 완료 핸들러 (Worker 시그널용)"""
        try:
            # 중지되었는지 확인
            if hasattr(self.backtest_worker, 'cancelled') and self.backtest_worker.cancelled:
                self.result_area.setText("⏹ 백테스팅이 사용자에 의해 중지되었습니다.")
                QMessageBox.information(self, "중지됨", "백테스팅이 중지되었습니다.")
                return

            if error:
                QMessageBox.critical(self, "오류", f"백테스팅 실패:\n{error}")
                return

            # 결과 표시
            if summary:
                self.display_backtest_results(summary)

        except Exception as e:
            QMessageBox.critical(self, "오류", f"백테스팅 결과 처리 중 오류:\n{str(e)}")
        finally:
            # UI 복원
            self.backtest_btn.setEnabled(True)
            self.backtest_progress_bar.setVisible(False)
            self.backtest_progress_label.setVisible(False)
            self.backtest_cancel_btn.setVisible(False)

    def update_backtest_progress(self, current, total, message):
        """백테스팅 진행률 업데이트 (레거시 - 삭제 예정)"""
        self.backtest_progress_bar.setValue(current)
        self.backtest_progress_label.setText(f"{message} - {current}/{total}")
        QApplication.processEvents()

    def _format_model_accuracies(self, model_accuracies):
        """모델별 적중률 포맷팅"""
        import logging
        logger = logging.getLogger(__name__)

        logger.debug(f"🎨 _format_model_accuracies 호출됨")
        logger.debug(f"🎨 model_accuracies 타입: {type(model_accuracies)}")
        logger.debug(f"🎨 model_accuracies 내용: {model_accuracies}")
        logger.debug(f"🎨 model_accuracies bool 값: {bool(model_accuracies)}")

        if not model_accuracies:
            logger.warning(f"🎨 model_accuracies가 비어있음!")
            return "    • 모델별 데이터 없음"

        lines = []
        # 적중률 순으로 정렬
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        logger.debug(f"🎨 정렬된 모델 수: {len(sorted_models)}")

        for model_name, accuracy in sorted_models:
            # 이모지 선택
            if accuracy >= 60:
                emoji = "🏆"
            elif accuracy >= 50:
                emoji = "✅"
            else:
                emoji = "⚠️"

            line = f"    • {emoji} {model_name}: {accuracy:.1f}%"
            logger.debug(f"🎨 추가된 라인: {line}")
            lines.append(line)

        result = "\n".join(lines)
        logger.debug(f"🎨 최종 결과: {result}")
        return result

    def display_backtest_results(self, summary):
        """백테스팅 결과 표시"""
        # 예측 편향 분석
        pred_bull = summary.get('pred_bull', 0)
        pred_bear = summary.get('pred_bear', 0)
        total = summary['test_periods']

        bias_text = ""
        if pred_bull > total * 0.7:
            bias_text = "⚠️ 상승 편향 (낙관적 예측)"
        elif pred_bear > total * 0.7:
            bias_text = "⚠️ 하락 편향 (비관적 예측)"
        else:
            bias_text = "✅ 균형잡힌 예측"

        result_text = f"""
    {'='*60}
    🔬 {summary['ticker']} 백테스팅 결과
    {'='*60}

    📊 전체 통계:
    • 테스트 횟수: {summary['test_periods']}회
    • 방향 정확도: {summary['direction_accuracy']:.1f}%
    • 평균 MAE: {summary['avg_mae']:.2f}
    • 평균 MAPE: {summary['avg_mape']:.2f}%
    • 상관계수: {summary['correlation']:.3f}

    🎯 상세 분석:
    • 📈 상승장 적중률: {summary.get('bull_accuracy', 0):.1f}% ({summary.get('bull_total', 0)}회 중)
    • 📉 하락장 적중률: {summary.get('bear_accuracy', 0):.1f}% ({summary.get('bear_total', 0)}회 중)
    • 🎲 예측 분포: 상승 {pred_bull}회 / 하락 {pred_bear}회
    • {bias_text}

    🤖 모델별 적중률:
{self._format_model_accuracies(summary.get('model_accuracies', {}))}

    📈 개별 결과:
    """
        
        for i, r in enumerate(summary['results'], 1):
            direction = "✅" if r['direction_match'] else "❌"
            result_text += f"""
    {i}. {r['date'].strftime('%Y-%m-%d')}
        예측: {r['predicted_return']:+.2f}% → 실제: {r['actual_return']:+.2f}%
        {direction} 방향 {'정확' if r['direction_match'] else '틀림'}
    """
        
        result_text += f"\n{'='*60}"
        
        self.result_area.setText(result_text)

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
            logger.error(f"인기 종목 로드 오류: {e}")
    
    def perform_search(self):
        """마스터 CSV에서 검색 수행"""
        query = self.search_input.text().strip()

        if query in self.search_cache:
            logger.debug(f"캐시 사용: {query}")
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
            logger.error(f"검색 오류: {e}")

    def search_stocks_with_api(self, search_term):
        """API를 사용한 실시간 주식 검색 + 기존 CSV 백업"""

        logger.info(f"API로 '{search_term}' 검색 시작...")
        api_results = []
        
        # 1. 먼저 API로 검색 시도
        try:
            query = urllib.parse.quote(search_term)
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={query}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            res = requests.get(url, headers=headers, timeout=10)
            logger.debug(f"Status code: {res.status_code}")

            if res.ok:
                data = res.json()
                quotes = data.get('quotes', [])
                logger.info(f"API에서 {len(quotes)}개 종목 발견")

                # Make csv from json.
                api_results = self.convert_api_to_csv_format(quotes, search_term)

            else:
                logger.warning(f"Request failed: {res.text[:200]}")  # 에러일 경우 앞부분 출력

        except Exception as e:
            logger.error(f"API 검색 실패: {e}")
        
        # 2. CSV에서도 검색 (백업용)
        csv_results = self.search_master_csv(search_term)
        
        # 3. 결과 병합
        combined_results = self.merge_search_results(api_results, csv_results)

        logger.info(f"총 {len(combined_results)}개 종목 반환")
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
                logger.warning(f"API 데이터 변환 오류: {e}")
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
            logger.error(f"검색 오류: {e}")
            if hasattr(self, 'csv_export_btn'):
                self.csv_export_btn.setEnabled(False)

    def print_results_as_csv(self, results):
        """검색 결과를 CSV 포맷으로 콘솔에 출력"""
        logger.info("\n" + "="*80)
        logger.info(f"검색 결과 (상위 {len(results)}개) - CSV 포맷:")
        logger.info("="*80)

        # CSV 헤더
        logger.info("ticker,name,sector,market_cap,market,source,match_score")

        # 데이터 행들
        for stock in results:
            ticker = stock.get('ticker', '')
            name = stock.get('name', '').replace(',', ';')  # 쉼표를 세미콜론으로 변경
            sector = stock.get('sector', '').replace(',', ';')
            market_cap = stock.get('market_cap', 'N/A')
            market = stock.get('market', '')
            source = stock.get('source', 'CSV')
            match_score = stock.get('match_score', 0)

            logger.info(f"{ticker},{name},{sector},{market_cap},{market},{source},{match_score}")

        logger.info("="*80)

    def search_master_csv(self, search_term):
        """마스터 CSV 파일들에서 검색 - 통합 검색 모듈 사용"""
        # ✅ 최적화: unified_search 사용 (96줄 → 3줄)
        results = search_stocks(search_term)

        # 기존 형식에 맞춰 변환 (match_score 추가)
        for result in results:
            ticker_upper = result['ticker'].upper()
            name_upper = result['name'].upper()
            search_upper = search_term.strip().upper()

            # 매칭 점수 계산
            if ticker_upper == search_upper:
                match_score = 100
            elif search_upper in ticker_upper:
                match_score = 80
            elif search_upper in name_upper:
                match_score = 70
            else:
                match_score = 50

            result['match_score'] = match_score
            result['raw_market_cap'] = result.get('market_cap', 0)

            # 시가총액 포맷팅
            market_cap = result.get('market_cap', 0)
            if pd.notna(market_cap) and market_cap > 0:
                if market_cap >= 1e12:
                    result['market_cap'] = f"{market_cap/1e12:.1f}T"
                elif market_cap >= 1e9:
                    result['market_cap'] = f"{market_cap/1e9:.1f}B"
                elif market_cap >= 1e6:
                    result['market_cap'] = f"{market_cap/1e6:.1f}M"
                else:
                    result['market_cap'] = f"{market_cap:,.0f}"
            else:
                result['market_cap'] = "N/A"

        # 매치 점수와 시가총액으로 정렬
        results.sort(key=lambda x: (-x.get('match_score', 0), -x.get('raw_market_cap', 0)))
        return results
    
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

            # market_cap을 포맷팅 (OverflowError 방지)
            market_cap_raw = stock.get('market_cap', '')
            if isinstance(market_cap_raw, (int, float)):
                market_cap_str = format_market_cap_value(market_cap_raw)
            else:
                market_cap_str = str(market_cap_raw) if market_cap_raw else 'N/A'

            self.results_table.setItem(i, 3, QTableWidgetItem(market_cap_str))
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
            # 과거 데이터 가져오기 (캐싱 사용)
            historical_data = get_stock_data(ticker, period="90d")
            
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

                # 예측 가격 처리 (단수/복수 형식 모두 지원)
                if 'predicted_price' in self.result:
                    target_price = self.result['predicted_price']
                    predicted_prices_array = None
                elif 'predicted_prices' in self.result:
                    predicted_prices_array = self.result['predicted_prices']
                    target_price = predicted_prices_array[-1]
                else:
                    target_price = current_price
                    predicted_prices_array = None

                # 실제 예측 가격 배열이 있으면 사용, 없으면 부드러운 곡선 생성
                if predicted_prices_array is not None and len(predicted_prices_array) == forecast_days:
                    predicted_prices = np.array(predicted_prices_array)
                else:
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

            # 예측 가격 처리
            if 'predicted_price' in self.result:
                final_price = self.result['predicted_price']
            elif 'predicted_prices' in self.result:
                final_price = self.result['predicted_prices'][-1]
            else:
                final_price = self.result['current_price']

            prices = [self.result['current_price']] + [final_price] * forecast_days
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
