"""
dialogs.py
각종 다이얼로그 클래스들
"""

import pandas as pd
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class CSVEditorDialog(QDialog):
    """CSV 파일 편집 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('📝 CSV 파일 편집')
        self.setGeometry(200, 200, 900, 700)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        info_label = QLabel("💡 종목 정보를 직접 편집할 수 있습니다. 편집 후 '저장' 버튼을 클릭하세요.")
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 파일 선택
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("편집할 파일:"))
        
        self.file_combo = QComboBox()
        self.file_combo.addItems(["korea_stocks.csv", "usa_stocks.csv", "sweden_stocks.csv"])
        self.file_combo.currentTextChanged.connect(self.load_csv_file)
        file_layout.addWidget(self.file_combo)
        
        self.load_btn = QPushButton("🔄 파일 로드")
        self.load_btn.clicked.connect(self.load_csv_file)
        file_layout.addWidget(self.load_btn)
        
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # 테이블
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.add_row_btn = QPushButton("➕ 행 추가")
        self.add_row_btn.clicked.connect(self.add_row)
        self.add_row_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        button_layout.addWidget(self.add_row_btn)
        
        self.delete_row_btn = QPushButton("➖ 행 삭제")
        self.delete_row_btn.clicked.connect(self.delete_row)
        self.delete_row_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        button_layout.addWidget(self.delete_row_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("💾 저장")
        self.save_btn.clicked.connect(self.save_csv_file)
        self.save_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        button_layout.addWidget(self.save_btn)
        
        self.cancel_btn = QPushButton("❌ 취소")
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
            reply = QMessageBox.question(self, '확인', 
                                       f'{current_row + 1}번째 행을 삭제하시겠습니까?',
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.table.removeRow(current_row)
        else:
            QMessageBox.information(self, "알림", "삭제할 행을 선택해주세요.")
    
    def save_csv_file(self):
        """CSV 파일 저장"""
        filename = self.file_combo.currentText()
        filepath = f'stock_data/{filename}'
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

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
            
            QMessageBox.information(self, "완료", f"✅ {filename} 파일이 저장되었습니다.")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"파일 저장 실패: {str(e)}")


class ConditionBuilderDialog(QDialog):
    """조건 빌더 다이얼로그"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('🔧 사용자 정의 조건 생성')
        self.setGeometry(300, 300, 600, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        info_label = QLabel("💡 나만의 매수/매도 조건을 생성할 수 있습니다.")
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 조건 이름
        name_group = QGroupBox("조건 이름")
        name_layout = QVBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("예: RSI 과매도 반등")
        name_layout.addWidget(self.name_edit)
        name_group.setLayout(name_layout)
        layout.addWidget(name_group)
        
        # 매수/매도 선택
        action_group = QGroupBox("조건 유형")
        action_layout = QHBoxLayout()
        self.action_combo = QComboBox()
        self.action_combo.addItems(["BUY (매수)", "SELL (매도)"])
        action_layout.addWidget(self.action_combo)
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # 지표 선택
        indicator_group = QGroupBox("기술적 지표")
        indicator_layout = QVBoxLayout()
        
        self.indicator_combo = QComboBox()
        indicators = [
            ('RSI', 'RSI (상대강도지수)'),
            ('MACD', 'MACD'),
            ('MACD_Signal', 'MACD Signal'),
            ('%K', '스토캐스틱 %K'),
            ('%D', '스토캐스틱 %D'),
            ('Williams_R', '윌리엄스 %R'),
            ('MA20', '20일 이동평균선'),
            ('MA60', '60일 이동평균선'),
            ('MA120', '120일 이동평균선'),
            ('BB_Upper', '볼린저 상단선'),
            ('BB_Lower', '볼린저 하단선'),
            ('CCI', 'CCI (상품채널지수)'),
            ('Volume_Ratio', '거래량 비율'),
            ('Close', '종가'),
            ('High', '고가'),
            ('Low', '저가')
        ]
        
        for value, text in indicators:
            self.indicator_combo.addItem(text, value)
        
        indicator_layout.addWidget(self.indicator_combo)
        indicator_group.setLayout(indicator_layout)
        layout.addWidget(indicator_group)
        
        # 연산자 선택
        operator_group = QGroupBox("연산자")
        operator_layout = QVBoxLayout()
        
        self.operator_combo = QComboBox()
        operators = [
            ('>', '> (초과)'),
            ('<', '< (미만)'),
            ('>=', '>= (이상)'),
            ('<=', '<= (이하)'),
            ('==', '== (같음)'),
            ('cross_above', '상향 돌파'),
            ('cross_below', '하향 돌파')
        ]
        
        for value, text in operators:
            self.operator_combo.addItem(text, value)
        
        operator_layout.addWidget(self.operator_combo)
        operator_group.setLayout(operator_layout)
        layout.addWidget(operator_group)
        
        # 값 입력
        value_group = QGroupBox("비교값")
        value_layout = QVBoxLayout()
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("예: 30, 70, 0 등")
        value_layout.addWidget(self.value_edit)
        value_group.setLayout(value_layout)
        layout.addWidget(value_group)
        
        # 예시 설명
        example_group = QGroupBox("📝 작성 예시")
        example_layout = QVBoxLayout()
        
        examples = """
🔹 RSI > 70: RSI가 70보다 클 때 (과매수)
🔹 MACD cross_above 0: MACD가 0선을 상향돌파할 때
🔹 Close < MA20: 종가가 20일선 아래일 때
🔹 Volume_Ratio >= 1.5: 거래량이 평균의 1.5배 이상일 때
🔹 %K cross_above %D: 스토캐스틱에서 %K가 %D를 상향돌파할 때
        """
        
        example_label = QLabel(examples)
        example_label.setWordWrap(True)
        example_label.setStyleSheet("color: #555; font-size: 11px;")
        example_layout.addWidget(example_label)
        example_group.setLayout(example_layout)
        layout.addWidget(example_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        preview_btn = QPushButton("🔍 미리보기")
        preview_btn.clicked.connect(self.preview_condition)
        button_layout.addWidget(preview_btn)
        
        button_layout.addStretch()
        
        ok_btn = QPushButton("✅ 확인")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("❌ 취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def preview_condition(self):
        """조건 미리보기"""
        try:
            condition = self.get_condition()
            if condition:
                preview_text = f"""
생성될 조건:

📌 조건명: {condition['name']}
📌 유형: {condition['action']}
📌 내용: {condition['indicator']} {condition['operator']} {condition['value']}

이 조건이 만족되면 해당 종목이 후보로 선정됩니다.
                """
                QMessageBox.information(self, "조건 미리보기", preview_text)
        except:
            QMessageBox.warning(self, "오류", "조건을 완전히 입력해주세요.")
    
    def get_condition(self):
        """생성된 조건 반환"""
        try:
            name = self.name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "오류", "조건 이름을 입력해주세요.")
                return None
            
            action = self.action_combo.currentData() or self.action_combo.currentText().split()[0]
            indicator = self.indicator_combo.currentData()
            operator = self.operator_combo.currentData()
            value = float(self.value_edit.text())
            
            return {
                'name': name,
                'action': action,
                'indicator': indicator,
                'operator': operator,
                'value': value
            }
        except ValueError:
            QMessageBox.warning(self, "오류", "비교값은 숫자여야 합니다.")
            return None
        except Exception as e:
            QMessageBox.warning(self, "오류", f"조건 생성 중 오류: {str(e)}")
            return None


class ConditionManagerDialog(QDialog):
    """조건 관리 다이얼로그"""
    def __init__(self, conditions, parent=None):
        super().__init__(parent)
        self.conditions = conditions.copy()
        self.setWindowTitle('📋 사용자 정의 조건 관리')
        self.setGeometry(300, 300, 700, 500)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # 상단 설명
        info_label = QLabel("💡 생성된 사용자 정의 조건들을 관리할 수 있습니다.")
        info_label.setStyleSheet("color: #666; padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # 조건 리스트
        list_group = QGroupBox(f"📝 등록된 조건 ({len(self.conditions)}개)")
        list_layout = QVBoxLayout()
        
        self.condition_list = QListWidget()
        self.condition_list.setAlternatingRowColors(True)
        self.update_condition_list()
        list_layout.addWidget(self.condition_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 버튼들
        button_layout = QHBoxLayout()
        
        self.edit_btn = QPushButton("📝 편집")
        self.edit_btn.clicked.connect(self.edit_condition)
        self.edit_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        button_layout.addWidget(self.edit_btn)
        
        self.delete_btn = QPushButton("🗑️ 삭제")
        self.delete_btn.clicked.connect(self.delete_condition)
        self.delete_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        button_layout.addWidget(self.delete_btn)
        
        button_layout.addStretch()
        
        self.ok_btn = QPushButton("✅ 확인")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("❌ 취소")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def update_condition_list(self):
        """조건 리스트 업데이트"""
        self.condition_list.clear()
        
        for i, condition in enumerate(self.conditions):
            action_icon = "💰" if condition['action'] == 'BUY' else "📉"
            item_text = f"{action_icon} [{condition['action']}] {condition['name']}: {condition['indicator']} {condition['operator']} {condition['value']}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # 인덱스 저장
            self.condition_list.addItem(item)
        
        # 그룹박스 제목 업데이트
        list_group = self.findChild(QGroupBox)
        if list_group:
            list_group.setTitle(f"📝 등록된 조건 ({len(self.conditions)}개)")
    
    def edit_condition(self):
        """조건 편집"""
        current_item = self.condition_list.currentItem()
        if current_item:
            QMessageBox.information(self, "알림", 
                                  "편집 기능은 현재 개발 중입니다.\n"
                                  "삭제 후 새로 추가해주세요.")
        else:
            QMessageBox.information(self, "알림", "편집할 조건을 선택해주세요.")
    
    def delete_condition(self):
        """조건 삭제"""
        current_item = self.condition_list.currentItem()
        if current_item:
            index = current_item.data(Qt.UserRole)
            condition_name = self.conditions[index]['name']
            
            reply = QMessageBox.question(self, '확인', 
                                       f"'{condition_name}' 조건을 삭제하시겠습니까?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                del self.conditions[index]
                self.update_condition_list()
        else:
            QMessageBox.information(self, "알림", "삭제할 조건을 선택해주세요.")
    
    def get_conditions(self):
        """조건 리스트 반환"""
        return self.conditions