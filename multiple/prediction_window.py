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
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        button_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton('🚀 AI 예측 시작')
        self.predict_btn.clicked.connect(self.start_prediction)
        button_layout.addWidget(self.predict_btn)
        
        self.export_btn = QPushButton('📊 결과 내보내기')
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        close_btn = QPushButton('닫기')
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
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
        self.days_input = QSpinBox()
        self.days_input.setRange(1, 30)
        self.days_input.setValue(7)
        self.days_input.setSuffix(" 일")
        layout.addWidget(self.days_input, 1, 1)
        
        # 모델 선택 (Enhanced Screener 정보 표시)
        layout.addWidget(QLabel("사용 모델:"), 2, 0)
        self.model_combo = QComboBox()
        if ML_AVAILABLE:
            self.model_combo.addItems([
                "🚀 Enhanced Ensemble (XGBoost + LightGBM + RF + ET + GB)",
                "📊 모든 모델 자동 앙상블",
                "🎯 성능 기반 가중치",
                "🔒 완전한 일관성 보장"
            ])
        else:
            self.model_combo.addItems(["❌ Enhanced Screener 필요"])
        layout.addWidget(self.model_combo, 2, 1)
        
        panel.setLayout(layout)
        return panel

    def show_stock_search_dialog(self):
        """종목 검색 다이얼로그 표시"""
        dialog = StockSearchDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            selected_ticker = dialog.get_selected_ticker()
            if selected_ticker:
                self.ticker_input.setText(selected_ticker)

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
    
    def start_prediction(self):
        """Enhanced Screener의 predict_stock 사용한 예측 시작"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "Enhanced Screener가 설치되지 않았습니다.")
            return
        
        ticker = self.ticker_input.text().strip().upper()
        days = self.days_input.value()
        
        if not ticker:
            QMessageBox.warning(self, "오류", "종목 코드를 입력해주세요.")
            return
        
        # UI 비활성화
        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("🔄 예측 중...")
        
        # 예측 실행 - Enhanced Screener의 predict_stock 사용
        QApplication.processEvents()
        
        try:
            # ✅ Enhanced Screener의 통합된 predict_stock 사용
            result, error = self.predictor.predict_stock(ticker, forecast_days=days)
            
            # UI 복구
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("🚀 AI 예측 시작")
            
            if error:
                QMessageBox.critical(self, "예측 오류", error)
                return
            
            if result:
                # Enhanced Screener 결과를 UI에 맞게 변환
                converted_result = self.convert_enhanced_result(result, days)
                self.display_results(converted_result)
                self.plot_prediction(converted_result)
                self.export_btn.setEnabled(True)
                self.last_result = converted_result
        
        except Exception as e:
            # UI 복구
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("🚀 AI 예측 시작")
            QMessageBox.critical(self, "예측 오류", f"예측 중 오류가 발생했습니다:\n{str(e)}")
    
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
        """결과 표시 - Enhanced Screener 정보 포함"""
        # 추천 결정
        return_rate = result['expected_return']
        confidence = result['confidence']
        
        if return_rate > 0.02 and confidence > 0.7:
            recommendation = "📈 강력 매수"
            color = "🟢"
        elif return_rate > 0.005 and confidence > 0.6:
            recommendation = "📈 매수"
            color = "🟡"
        elif return_rate < -0.02 and confidence > 0.7:
            recommendation = "📉 강력 매도"
            color = "🔴"
        elif return_rate < -0.005 and confidence > 0.6:
            recommendation = "📉 매도"
            color = "🟠"
        else:
            recommendation = "⏸️ 관망"
            color = "⚪"
        
        # 결과 텍스트 생성
        text = f"""
══════════════════════════════════════════════════
🎯 {result['ticker']} Enhanced AI 예측 ({result['days']}일 후)
══════════════════════════════════════════════════

💰 현재 가격: ${result['current_price']:.2f}
🎯 예측 가격: ${result['predicted_price']:.2f}
📊 예상 수익률: {return_rate*100:+.2f}%
🎚️ 신뢰도: {confidence*100:.1f}%

{color} 추천: {recommendation}

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
        
        # 개별 모델 결과 (Enhanced 버전)
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

💡 참고: Enhanced Screener는 더 정확하고 일관성 있는
   예측을 제공합니다. 투자 결정 시 다른 요소들도 함께 고려하세요.
        """
        
        self.result_area.setText(text)
    
    def plot_prediction(self, result):
        """예측 차트 그리기"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # 간단한 가격 예측 차트
        days = ['현재', f'{result["days"]}일 후']
        prices = [result['current_price'], result['predicted_price']]
        
        colors = ['blue', 'green' if result['expected_return'] > 0 else 'red']
        bars = ax.bar(days, prices, color=colors, alpha=0.7)
        
        # 수익률 표시
        return_pct = result['expected_return'] * 100
        ax.text(1, result['predicted_price'], f'{return_pct:+.1f}%', 
                ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f"{result['ticker']} Enhanced AI 예측 ({result['days']}일)", fontsize=14)
        ax.set_ylabel("가격 ($)")
        ax.grid(True, alpha=0.3)
        
        # 신뢰도 정보 추가
        confidence_pct = result['confidence'] * 100
        ax.text(0.5, max(prices) * 0.9, f'신뢰도: {confidence_pct:.1f}%', 
                ha='center', fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        self.canvas.draw()
    
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


class QuickPredictionWidget(QWidget):
    """빠른 예측 위젯 - Enhanced Screener 사용"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.predictor = EnhancedCPUPredictor() if ML_AVAILABLE else None
        self.initUI()
    
    def initUI(self):
        layout = QHBoxLayout()
        
        # 종목 입력
        self.ticker_input = QLineEdit()
        self.ticker_input.setPlaceholderText("종목 코드 (예: AAPL)")
        self.ticker_input.setMaximumWidth(100)
        layout.addWidget(self.ticker_input)
        
        # 예측 버튼
        self.predict_btn = QPushButton("🚀 Enhanced 예측")
        self.predict_btn.clicked.connect(self.quick_predict)
        layout.addWidget(self.predict_btn)
        
        # 결과 라벨
        self.result_label = QLabel("Enhanced 예측 결과가 여기에 표시됩니다")
        layout.addWidget(self.result_label)
        
        # 상세 보기 버튼
        self.detail_btn = QPushButton("📊 상세 분석")
        self.detail_btn.clicked.connect(self.show_detail)
        self.detail_btn.setEnabled(False)
        layout.addWidget(self.detail_btn)
        
        self.setLayout(layout)
    
    def quick_predict(self):
        """Enhanced Screener로 빠른 예측"""
        if not ML_AVAILABLE:
            QMessageBox.warning(self, "오류", "Enhanced Screener가 필요합니다.")
            return
        
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "오류", "종목 코드를 입력하세요.")
            return
        
        self.predict_btn.setEnabled(False)
        self.predict_btn.setText("🔄 예측 중...")
        
        try:
            # Enhanced Screener 사용
            result, error = self.predictor.predict_stock(ticker, forecast_days=7)
            
            if error:
                self.result_label.setText(f"❌ {error}")
            elif result:
                return_pct = result['expected_return'] * 100
                confidence_pct = result['confidence'] * 100
                
                if return_pct > 2:
                    icon = "📈"
                elif return_pct < -2:
                    icon = "📉"
                else:
                    icon = "⏸️"
                
                self.result_label.setText(
                    f"{icon} {ticker}: {return_pct:+.1f}% (신뢰도: {confidence_pct:.0f}%)"
                )
                self.detail_btn.setEnabled(True)
                self.last_result = result
            
        except Exception as e:
            self.result_label.setText(f"❌ 오류: {str(e)[:50]}...")
        
        finally:
            self.predict_btn.setEnabled(True)
            self.predict_btn.setText("🚀 Enhanced 예측")
    
    def show_detail(self):
        """상세 분석 창 표시"""
        if hasattr(self, 'last_result'):
            dialog = StockPredictionDialog(self)
            if hasattr(dialog, 'ticker_input'):
                dialog.ticker_input.setText(self.last_result['ticker'])
            dialog.exec_()


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
        self.initUI()
        
        # 초기 인기 종목 표시
        self.show_popular_stocks()
    
    def initUI(self):
        layout = QVBoxLayout()
        
        # 상단 정보
        info_label = QLabel("💡 마스터 CSV에서 종목을 검색합니다 (한국, 미국, 스웨덴 전체)")
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
        """텍스트 변경 시 자동 검색 (3글자 이상)"""
        if len(text.strip()) >= 3:
            self.perform_search()
        elif len(text.strip()) == 0:
            self.show_popular_stocks()
    
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
        if len(query) < 1:
            self.show_popular_stocks()
            return
        
        try:
            self.status_label.setText(f"'{query}' 검색 중...")
            QApplication.processEvents()
            
            # 마스터 CSV에서 검색
            results = self.search_master_csv(query)
            self.display_results(results)
            
            if results:
                self.status_label.setText(f"🔍 {len(results)}개 종목 발견 (매치점수순)")
            else:
                self.status_label.setText("❌ 검색 결과가 없습니다")
                
        except Exception as e:
            self.status_label.setText(f"❌ 검색 오류: {str(e)}")
            print(f"검색 오류: {e}")
    
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
    
    def display_results(self, results):
        """검색 결과 표시"""
        self.results_table.setRowCount(len(results))
        
        for i, stock in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(stock.get('ticker', '')))
            self.results_table.setItem(i, 1, QTableWidgetItem(stock.get('name', '')))
            self.results_table.setItem(i, 2, QTableWidgetItem(stock.get('market', '')))
            self.results_table.setItem(i, 3, QTableWidgetItem(stock.get('sector', '')))
            self.results_table.setItem(i, 4, QTableWidgetItem(stock.get('market_cap', 'N/A')))
            
            # 매치점수 표시
            match_score = stock.get('match_score', 0)
            score_item = QTableWidgetItem(str(match_score))
            
            # 매치점수에 따른 색상 구분
            if match_score >= 90:
                score_item.setBackground(QColor(76, 175, 80, 100))  # 초록
            elif match_score >= 70:
                score_item.setBackground(QColor(255, 193, 7, 100))  # 노랑
            elif match_score >= 50:
                score_item.setBackground(QColor(255, 87, 34, 100))  # 주황
                
            self.results_table.setItem(i, 5, score_item)
        
        # 첫 번째 행 선택
        if len(results) > 0:
            self.results_table.selectRow(0)
    
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

# 사용 예제 및 테스트
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    
    print("🧪 Prediction Window - Enhanced Screener 통합 테스트")
    
    if ML_AVAILABLE:
        print("✅ Enhanced Screener 사용 가능")
        
        # 예제 1: 메인 예측 다이얼로그 테스트
        dialog = StockPredictionDialog()
        dialog.show()
        
        # 예제 2: 빠른 예측 위젯 테스트
        quick_widget = QuickPredictionWidget()
        quick_widget.show()
        
    else:
        print("⚠️ Enhanced Screener 설치 필요")
        print("enhanced_screener.py 파일과 ML 라이브러리가 필요합니다")
        
        # 오류 다이얼로그 표시
        error_dialog = StockPredictionDialog()
        error_dialog.show()
    
    sys.exit(app.exec_())