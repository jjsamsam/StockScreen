#!/usr/bin/env python3
"""
Advanced Global Stock Screener
메인 실행 파일

사용법:
    python main.py

필요한 라이브러리:
    pip install PyQt5 pandas yfinance numpy matplotlib
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from logger_config import get_logger

logger = get_logger(__name__)

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from screener import StockScreener
except ImportError as e:
    logger.error(f"모듈 import 오류: {e}")
    logger.error("필요한 파일들이 같은 폴더에 있는지 확인해주세요:")
    logger.error("- main.py")
    logger.error("- screener.py")
    logger.error("- chart_window.py")
    logger.error("- dialogs.py")
    logger.error("- utils.py")
    sys.exit(1)

def check_dependencies():
    """필요한 라이브러리들이 설치되어 있는지 확인"""
    required_packages = {
        'PyQt5': 'PyQt5',
        'pandas': 'pandas', 
        'yfinance': 'yfinance',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.error("다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            logger.error(f"   - {package}")
        logger.error("\n설치 명령어:")
        logger.error(f"pip install {' '.join(missing_packages)}")
        return False

    logger.info("모든 필요한 패키지가 설치되어 있습니다.")
    return True

def apply_application_style(app):
    """애플리케이션 전체 스타일 적용"""
    style = """
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
    QScrollArea {
        border: 1px solid #cccccc;
        border-radius: 4px;
        background-color: white;
    }
    """
    app.setStyleSheet(style)

def main():
    """메인 함수 - 프로그램 시작점"""
    logger.info("=" * 50)
    logger.info("Advanced Global Stock Screener")
    logger.info("고급 글로벌 주식 스크리너를 시작합니다...")
    logger.info("=" * 50)
    
    # 의존성 체크
    if not check_dependencies():
        input("\n엔터를 눌러 종료...")
        return
    
    # PyQt5 애플리케이션 생성
    app = QApplication(sys.argv)
    app.setApplicationName("Advanced Stock Screener")
    app.setApplicationVersion("1.0")
    
    # 스타일 적용
    apply_application_style(app)
    
    try:
        logger.info("GUI 초기화 중...")

        # 메인 스크리너 윈도우 생성
        screener = StockScreener()
        screener.show()

        logger.info("프로그램이 성공적으로 시작되었습니다!")
        logger.info("\n사용 방법:")
        logger.info("1. '샘플 생성' 버튼으로 기본 종목 리스트 생성")
        logger.info("2. '온라인 종목 업데이트'로 최신 종목 리스트 확보")
        logger.info("3. 매수/매도 조건 선택 후 '스크리닝 시작'")
        logger.info("4. 결과 테이블에서 종목 더블클릭으로 차트 확인")
        
        # 이벤트 루프 시작
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()

        # GUI로 오류 메시지 표시
        try:
            QMessageBox.critical(None, "오류",
                               f"프로그램 실행 중 오류가 발생했습니다:\n{str(e)}\n\n"
                               f"콘솔 창에서 자세한 오류 정보를 확인하세요.")
        except:
            pass

        input("\n엔터를 눌러 종료...")

if __name__ == '__main__':
    main()